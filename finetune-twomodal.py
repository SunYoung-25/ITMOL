import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
import os
import warnings
from dataloader.all_dataloder import MultiModalDataset, MultiModalClassifier, train_one_epoch_multitask, evaluate_on_multitask
from models.train.dual_model_utils import load_pretrained_weights2
from utils.public_utils import is_left_better_right, get_tqdm_desc
from utils.splitter import get_split_data
from dataloader.image_dataloader import get_datasets
from models.train_utils import fix_train_random_seed, load_smiles
from splitter import split_train_val_test_idx, split_train_val_test_idx_stratified, scaffold_split_train_val_test, \
    random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test,enhanced_stratified_scaffold_split
import pandas as pd

warnings.filterwarnings("ignore")


def get_optimizer_parameters(model, args, no_decay=['bias', 'LayerNorm.weight']):
    optimizer_grouped_parameters = []
    model_module = model.module if hasattr(model, 'module') else model

    # 1. 预训练模型
    pretrained_layers = ['ViT', 'SciBERT']
    for layer_name in pretrained_layers:
        layer = getattr(model_module, layer_name)
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': args.lr,
                'weight_decay': 10 ** args.weight_decay
            },
            {
                'params': [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': args.lr,
                'weight_decay': 0.0
            }
        ]

    # 2. 分类器
    classifier = model_module.fc
    optimizer_grouped_parameters += [
        {
            'params': [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': args.lr_new_layers,
            'weight_decay': 10 ** args.weight_decay
        },
        {
            'params': [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': args.lr_new_layers,
            'weight_decay': 0.0
        },
    ]

    return optimizer_grouped_parameters

def plot_losses(epoch_losses, args):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses['train'], label='Training Loss')
    plt.plot(epoch_losses['valid'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    plt.savefig(os.path.join(args.save_path, 'training_validation_loss.png'))
    plt.close()

def set_seed(seed):
    print(f"Setting random seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_current_lr(optimizer):
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list if len(lr_list) > 1 else lr_list[0]


def load_labels(txt_file, task_type="classification"):
    """
    从文件加载标签列表。

    :param file_path: 标签文件的路径。
    :return: 标签列表。
    """
    df = pd.read_csv(txt_file)
    index = df["index"].values.astype(int)
    labels = np.array(df.label.apply(lambda x: str(x).split(' ')).tolist())
    labels = labels.astype(int) if task_type == "classification" else labels.astype(float)
    assert len(index) == labels.shape[0]
    return labels

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate model")
    # Basic parameters
    parser.add_argument('--dataset', type=str, default="bace", help='Dataset name, e.g., bbbp, tox21, etc.')
    parser.add_argument('--dataroot', type=str, default="./Data/", help='Path to data root')
    parser.add_argument('--save_path', type=str, default='./results', help='Directory to save results and plots')
    parser.add_argument('--resume', default='10.11-model.pth', type=str, metavar='PATH', help='Path to checkpoint (default: None)')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 2)')
    parser.add_argument('--device_ids', type=int, nargs='+', default=[2], help='List of device IDs for CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators')
    parser.add_argument('--patience', type=int, default=10, help='Patience for learning rate reduction')
    parser.add_argument('--early_stop_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--epochs', type=int, default=100, help='Number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int, help='Manual epoch number (useful on restarts)')
    # Optimizer parameters
    parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate for pretrained models (default: 1e-5)')
    parser.add_argument('--lr_new_layers', default=1e-4, type=float, help='Learning rate for new layers (default: 1e-4)')
    parser.add_argument('--weight_decay', default=-5, type=float, help='Weight decay exponent (default: -5)')
    parser.add_argument('--batch', default=32, type=int, help='Mini-batch size (default: 32)')
    # Other parameters
    parser.add_argument('--split', default="random_scaffold", type=str,
                        choices=['random', 'stratified', 'scaffold', 'random_scaffold', 'scaffold_balanced','scaffold_stratified','enhanced_scaffold_stratified'],
                        help='Data splitting method')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='Task type')
    parser.add_argument('--imageSize', type=int, default=224, help='Input image size for the network')
    parser.add_argument('--image_aug', action='store_true', default=False, help='Whether to use data augmentation')
    parser.add_argument('--save_finetune_ckpt', type=int, default=0, choices=[0, 1],
                        help='1 to save best checkpoint, 0 to not save')
    parser.add_argument('--log_dir', default=' ', help='Path to log directory')

    return parser.parse_args()

def main(args):
    # Set the device
    device = torch.device(f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed")

    if args.task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
    elif args.task_type == "regression":
        eval_metric = "rmse"
        valid_select = "min"
        min_value = np.inf
    else:
        raise ValueError(f"Task type {args.task_type} is not supported.")

    print(f"Evaluation metric: {eval_metric}; Validation selection: {valid_select}")

    stopwords_file_path = 'stopwords-en.txt'
    dataset = MultiModalDataset(args.txt_file, args.image_folder, 'scibert_scivocab_uncased', stopwords_file_path)
    num_tasks = dataset.num_tasks
    print(f"Number of tasks: {num_tasks}")

    # if args.split == "random":
    #     train_idx, val_idx, test_idx = split_train_val_test_idx(list(range(0, len(dataset))), frac_train=0.8,
    #                                                             frac_valid=0.1, frac_test=0.1, seed=args.seed)
    # elif args.split == "stratified":
    #     train_idx, val_idx, test_idx = split_train_val_test_idx_stratified(list(range(0, len(names))), labels,
    #                                                                        frac_train=0.8, frac_valid=0.1,
    #                                                                        frac_test=0.1, seed=args.seed)
    # elif args.split == "scaffold":
    #     smiles = load_smiles(args.txt_file)
    #     train_idx, val_idx, test_idx = scaffold_split_train_val_test(list(range(0, len(dataset))), smiles, frac_train=0.8,
    #                                                                  frac_valid=0.1, frac_test=0.1)
    # elif args.split == "random_scaffold":
    #     smiles = load_smiles(args.txt_file)
    #     train_idx, val_idx, test_idx = random_scaffold_split_train_val_test(list(range(0, len(dataset))), smiles,
    #                                                                         frac_train=0.8, frac_valid=0.1,
    #                                                                         frac_test=0.1, seed=114514)
    # elif args.split == "scaffold_balanced":
    #     smiles = load_smiles(args.txt_file)
    #     train_idx, val_idx, test_idx = scaffold_split_balanced_train_val_test(list(range(0, len(dataset))), smiles,
    #                                                                           frac_train=0.8, frac_valid=0.1,
    #                                                                           frac_test=0.1, seed=42,
    #                                                                           balanced=True)
    # elif args.split == "scaffold_stratified":
    #     smiles = load_smiles(args.txt_file)
    #     train_idx, val_idx, test_idx = scaffold_split_balanced_train_val_test(
    #         list(range(0, len(dataset))),
    #         smiles,
    #         frac_train=0.8,
    #         frac_valid=0.1,
    #         frac_test=0.1,
    #         seed=0,
    #         balanced=True
    #     )
    # elif args.split == "enhanced_scaffold_stratified":
    #     # 加载 SMILES 和标签数据
    #     smiles = load_smiles(args.txt_file)
    #     labels = load_labels(args.txt_file)  # 确保标签文件与 SMILES 文件对应
    #
    #     # 定义数据集索引
    #     dataset_size = len(smiles)
    #     index = list(range(dataset_size))
    #
    #     # 使用 enhanced_stratified_scaffold_split 进行数据切分
    #     train_idx, val_idx, test_idx = enhanced_stratified_scaffold_split(
    #         index=index,
    #         smiles_list=smiles,
    #         labels=labels,
    #         frac_train=0.8,
    #         frac_valid=0.1,
    #         frac_test=0.1,
    #         seed=args.seed
    #     )
    #
    #     print(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}, Test size: {len(test_idx)}")
    # Get train, validation, and test indices
    train_idx, val_idx, test_idx = get_split_data(args.dataset, args.dataroot)
    train_idx, val_idx, test_idx = train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
    train_subset = Subset(dataset, train_idx)
    valid_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_subset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_subset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    print(f"Number of training samples: {len(train_subset)}")
    print(f"Number of validation samples: {len(valid_subset)}")
    print(f"Number of test samples: {len(test_subset)}")

    # Initialize the model
    model = MultiModalClassifier(num_tasks)

    # Load pretrained weights if available
    load_flag, desc = load_pretrained_weights2(model.ViT, args.resume, device=device, submodule_key='ViT')
    if load_flag:
        print(desc)
    load_flag, desc = load_pretrained_weights2(model.SciBERT, args.resume, device=device, submodule_key='SciBERT')
    if load_flag:
        print(desc)

    # Move model to device
    model.to(device)

    # Wrap the model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    # 使用动态分组函数定义优化器
    optimizer = optim.AdamW(get_optimizer_parameters(model, args))

    # Define criterion
    if args.task_type == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Task type {args.task_type} is not supported.")

    # Initialize tracking variables
    results = {
        'highest_valid': float('-inf') if valid_select == "max" else float('inf'),
        'train_metric_at_highest_valid': None,
        'valid_metric_at_highest_valid': None,
        'test_metric_at_highest_valid': None,
        'highest_test': float('-inf') if valid_select == "max" else float('inf'),
        'train_metric_at_highest_test': None,
        'valid_metric_at_highest_test': None,
        'test_metric_at_highest_test': None
    }

    early_stop_counter = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min' if valid_select == "min" else 'max',
                                  patience=args.patience, factor=0.5, verbose=True)

    for epoch in range(args.start_epoch, args.epochs):
        # Get current learning rates
        current_lr = get_current_lr(optimizer)
        if isinstance(current_lr, list):
            print(f"Epoch {epoch}/{args.epochs - 1} - Current Learning Rates: {current_lr}")
        else:
            print(f"Epoch {epoch}/{args.epochs - 1} - Current Learning Rate: {current_lr}")

        tqdm_train_desc, tqdm_eval_train_desc, tqdm_eval_val_desc, tqdm_eval_test_desc = get_tqdm_desc(args.dataset, epoch)

        # Training
        train_one_epoch_multitask(model=model, optimizer=optimizer, data_loader=train_loader, criterion=criterion,
                                  device=device, epoch=epoch, task_type=args.task_type, tqdm_desc=tqdm_train_desc)

        # Evaluation
        train_loss, train_results = evaluate_on_multitask(model=model, data_loader=train_loader, criterion=criterion,
                                                          device=device, epoch=epoch, task_type=args.task_type,
                                                          tqdm_desc=tqdm_eval_train_desc, type="train")
        val_loss, val_results = evaluate_on_multitask(model=model, data_loader=valid_loader, criterion=criterion,
                                                      device=device, epoch=epoch, task_type=args.task_type,
                                                      tqdm_desc=tqdm_eval_val_desc, type="valid")
        test_loss, test_results = evaluate_on_multitask(model=model, data_loader=test_loader, criterion=criterion,
                                                        device=device, epoch=epoch, task_type=args.task_type,
                                                        tqdm_desc=tqdm_eval_test_desc, type="test")

        train_metric = train_results[eval_metric.upper()]
        valid_metric = val_results[eval_metric.upper()]
        test_metric = test_results[eval_metric.upper()]

        # Print all information
        print(f"Epoch: {epoch}")
        print(f"Dataset: {args.dataset}")
        print(f"Training Loss: {train_loss:.4f}, Training {eval_metric}: {train_metric:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation {eval_metric}: {valid_metric:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test {eval_metric}: {test_metric:.4f}")
        print("-" * 50)

        # Update learning rate scheduler
        scheduler.step(val_loss if valid_select == "min" else valid_metric)

        # Early stopping logic based on validation metric
        if is_left_better_right(valid_metric, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_metric
            results['train_metric_at_highest_valid'] = train_metric
            results['valid_metric_at_highest_valid'] = valid_metric
            results['test_metric_at_highest_valid'] = test_metric
            early_stop_counter = 0  # Reset early stopping counter

            # Save the best model if required
            if args.save_finetune_ckpt == 1:
                save_finetune_ckpt(model, optimizer, round(val_loss, 4), epoch, args.log_dir, "valid_best",
                                   lr_scheduler=None, result_dict=results)
        else:
            early_stop_counter += 1
            if early_stop_counter > args.early_stop_patience:
                print("Early stopping triggered based on validation metric.")
                break

        # Update the best test metric
        if is_left_better_right(test_metric, results['highest_test'], standard=valid_select):
            results['highest_test'] = test_metric
            results['train_metric_at_highest_test'] = train_metric
            results['valid_metric_at_highest_test'] = valid_metric
            results['test_metric_at_highest_test'] = test_metric

    # Print final results
    print("Final Results:")
    print(f"Highest Validation {eval_metric}: {results['highest_valid']:.4f}")
    print(f"Metrics at Highest Validation:")
    print(f"Train {eval_metric}: {results['train_metric_at_highest_valid']:.4f}")
    print(f"Valid {eval_metric}: {results['valid_metric_at_highest_valid']:.4f}")
    print(f"Test {eval_metric}: {results['test_metric_at_highest_valid']:.4f}")
    print("-" * 50)
    print(f"Highest Test {eval_metric}: {results['highest_test']:.4f}")
    print(f"Metrics at Highest Test:")
    print(f"Train {eval_metric}: {results['train_metric_at_highest_test']:.4f}")
    print(f"Valid {eval_metric}: {results['valid_metric_at_highest_test']:.4f}")
    print(f"Test {eval_metric}: {results['test_metric_at_highest_test']:.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
