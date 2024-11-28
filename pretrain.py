import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch import nn
import logging
import copy
from dataset import MultimodalDataset
from modal import MultiModalModel
import torchvision.transforms as transforms
from PIL import ImageFilter
import random
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
def get_hard_negatives(text_embeddings, top_k=1):
    # 计算相似度矩阵
    similarity = F.cosine_similarity(text_embeddings.unsqueeze(1), text_embeddings.unsqueeze(0), dim=-1)
    # 排除自身
    similarity.fill_diagonal_(0)
    # 选择相似度最高的负样本
    hard_negatives = torch.topk(similarity, k=top_k, dim=1).indices
    return hard_negatives

def parse_args():
    parser = argparse.ArgumentParser(description="Train a multimodal model with image and text inputs.")

    parser.add_argument('--csv_path', type=str, default='../processed_all_data.csv', help='Path to the CSV file with descriptions.')
    parser.add_argument('--image_dir', type=str, default='../images', help='Directory path where images are stored.')
    parser.add_argument('--text_tokenizer_path', type=str, default='allenai/scibert_scivocab_uncased', help='Path to the pretrained text tokenizer.')
    parser.add_argument('--image_encoder_pretrained_path', type=str, default='../vit/google', help='Path to the pretrained image encoder.')
    parser.add_argument('--text_encoder_pretrained_path', type=str, default='../scibert_scivocab_uncased', help='Path to the pretrained text encoder.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning_rate_pretrained', type=float, default=1e-5, help='Learning rate for pretrained models.')
    parser.add_argument('--learning_rate_new', type=float, default=1e-4, help='Learning rate for new layers.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0, 1], help='List of device IDs for multi-GPU training.')
    parser.add_argument('--max_len', type=int, default=300, help='Maximum length for text sequences.')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='Probability of masking tokens for MLM.')
    parser.add_argument('--imageSize', type=int, default=224, help='Size of images after preprocessing.')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients.')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision for training.')
    parser.add_argument('--use_warmup', action='store_true', help='Use learning rate warm-up at the beginning of training.')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training steps for learning rate warm-up.')

    return parser.parse_args()


def train_model(model, device, optimizer, scheduler, dataloader, num_epochs, patience, accumulation_steps, use_amp):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    no_improve_epoch = 0
    all_loss = []

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        model.train()
        running_loss = 0.0
        running_mlm_loss = 0.0
        running_cmm_loss = 0.0
        running_clip_loss = 0.0
        total_batches = 0  # 记录批次数

        loop = tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs - 1}')
        optimizer.zero_grad()

        for i, data in enumerate(loop):
            imgs, text_input_ids, text_attention_mask, mlm_labels = data

            # 将数据移动到设备
            imgs = imgs.to(device, non_blocking=True)
            text_input_ids = text_input_ids.to(device, non_blocking=True)
            text_attention_mask = text_attention_mask.to(device, non_blocking=True)
            mlm_labels = mlm_labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 直接访问 SciBERT
                text_embeddings = model.SciBERT(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state[:, 0, :]  # 使用 [CLS] token
                hard_neg_indices = get_hard_negatives(text_embeddings, top_k=1).to(device)  # 形状: [batch_size, 1]

                # 根据索引选择负样本并移除多余的维度
                negative_text_input_ids = text_input_ids[hard_neg_indices].squeeze(1)  # 形状: [batch_size, sequence_length]
                negative_text_attention_mask = text_attention_mask[hard_neg_indices].squeeze(1)  # 形状: [batch_size, sequence_length]

                # 确保负样本不与正样本相同
                if torch.equal(negative_text_input_ids, text_input_ids):
                    perm = torch.randperm(text_input_ids.size(0)).to(device)
                    negative_text_input_ids = text_input_ids[perm]  # 假设 perm 是一维的
                    negative_text_attention_mask = text_attention_mask[perm]

                # 前向传播
                total_loss, mlm_loss, cmm_loss, clip_loss = model(
                    imgs,
                    text_input_ids,
                    text_attention_mask,
                    mlm_labels=mlm_labels,
                    negative_text_input_ids=negative_text_input_ids,
                    negative_text_attention_mask=negative_text_attention_mask
                )
                total_loss = total_loss / accumulation_steps  # 梯度累积

            # 反向传播
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # 梯度累积步数
            if (i + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

                # 更新学习率调度器
                if scheduler is not None:
                    scheduler.step()

            # 累加损失
            running_loss += total_loss.item() * accumulation_steps
            running_mlm_loss += mlm_loss.item()
            running_cmm_loss += cmm_loss.item()
            running_clip_loss += clip_loss.item()
            total_batches += 1  # 记录批次数

            # 更新进度条描述
            loop.set_postfix(batch_loss=(total_loss.item() * accumulation_steps),
                            mlm_loss=mlm_loss.item(),
                            cmm_loss=cmm_loss.item(),
                            clip_loss=clip_loss.item())

        # 计算 epoch 的平均损失
        epoch_loss = running_loss / total_batches
        epoch_mlm_loss = running_mlm_loss / total_batches
        epoch_cmm_loss = running_cmm_loss / total_batches
        epoch_clip_loss = running_clip_loss / total_batches

        # 获取可学习参数的值
        if hasattr(model, 'log_alpha_mlm'):
            log_alpha_mlm = model.log_alpha_mlm.item()
            log_alpha_cmm = model.log_alpha_cmm.item()
            log_alpha_clip = model.log_alpha_clip.item()
        else:
            log_alpha_mlm = 'N/A'
            log_alpha_cmm = 'N/A'
            log_alpha_clip = 'N/A'

        # 记录日志
        logging.info(f'Epoch {epoch} Total Loss: {epoch_loss:.4f} | MLM Loss: {epoch_mlm_loss:.4f} | CMM Loss: {epoch_cmm_loss:.4f} | CLIP Loss: {epoch_clip_loss:.4f}')
        logging.info(f'Log Alphas - MLM: {log_alpha_mlm}, CMM: {log_alpha_cmm}, CLIP: {log_alpha_clip}')

        # Early stopping 逻辑
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epoch = 0

            # 保存当前最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, './best_model.pth')
            logging.info(f'Best model saved to ./best_model.pth')
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                logging.info(f"No improvement for {patience} consecutive epochs, stopping early.")
                break

    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {(time_elapsed % 60):.0f}s')
    model.load_state_dict(best_model_wts)
    return model, all_loss


def main():
    args = parse_args()

    # 设置主设备
    device = torch.device(f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu")

    img_transformer = [
        transforms.Resize((args.imageSize, args.imageSize)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=360),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]

    datasets_train = MultimodalDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        text_tokenizer_path=args.text_tokenizer_path,
        transform=transforms.Compose(img_transformer),
        max_len=args.max_len,
        mask_prob=args.mask_prob
    )
    data_loader_train = DataLoader(datasets_train, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = MultiModalModel(
        image_encoder_pretrained_path=args.image_encoder_pretrained_path,
        text_encoder_model_path=args.text_encoder_pretrained_path
    )

    # 将模型移动到设备
    if torch.cuda.is_available():
        model = model.to(device)
        # 移除 DataParallel 封装
        # if len(args.device_ids) > 1:
        #     model = nn.DataParallel(model, device_ids=args.device_ids)

    # 参数分组
    pretrained_param_names = set()
    for name, param in model.named_parameters():
        if 'ViT' in name or 'SciBERT' in name:
            pretrained_param_names.add(name)

    param_groups = [
        {
            'params': [param for name, param in model.named_parameters() if name in pretrained_param_names],
            'lr': args.learning_rate_pretrained
        },
        {
            'params': [param for name, param in model.named_parameters() if name not in pretrained_param_names],
            'lr': args.learning_rate_new
        }
    ]

    optimizer_ft = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = (len(data_loader_train) // args.accumulation_steps) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_proportion)

    scheduler_cosine = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_ft,
        T_0=total_steps - warmup_steps,
        T_mult=1,
        eta_min=1e-6
    )

    if args.use_warmup:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                return 1.0

        warmup_scheduler = lr_scheduler.LambdaLR(optimizer_ft, lr_lambda)

        scheduler = lr_scheduler.SequentialLR(
            optimizer_ft,
            schedulers=[warmup_scheduler, scheduler_cosine],
            milestones=[warmup_steps]
        )
    else:
        scheduler = scheduler_cosine

    model_trained, all_loss = train_model(
        model,
        device,
        optimizer_ft,
        scheduler,
        data_loader_train,
        args.num_epochs,
        args.patience,
        args.accumulation_steps,
        args.use_amp
    )

    torch.save(model_trained.state_dict(), './final_model.pth')
    logging.info(f'Final model saved to ./final_model.pth')

if __name__ == "__main__":
    main()