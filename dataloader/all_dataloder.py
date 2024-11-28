import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig,BertModel,ViTModel
from torchvision import transforms


class MultiModalDataset(Dataset):
    def __init__(self, csv_file, image_dir, text_tokenizer_path, stopwords_file_path, image_transform=None, max_len=300):
        self.df = pd.read_csv(csv_file)
        self.text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.image_dir = image_dir
        self.max_len = max_len
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = image_transform

        with open(stopwords_file_path, 'r') as file:
            self.stop_words = set(file.read().splitlines())

        #self.preprocessed_texts = [self.preprocess_text(text) for text in self.df['description']]
        self.preprocessed_texts = self.df['description'].tolist()
        self.labels = self.get_label(self.df)
        self.num_tasks = self.labels.shape[1]


    # def preprocess_text(self, text):
    #     words = text.split()
    #     filtered_words = [word for word in words if word not in self.stop_words]
    #     return ' '.join(filtered_words)

    def get_label(self, df):
        return np.array(df["label"].apply(lambda x: np.array(str(x).split()).astype(int).tolist()).tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Handling image paths
        file_index = self.df.iloc[idx]['index']
        image_path = os.path.join(self.image_dir, f"{file_index}.png")
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)

        processed_text = self.preprocessed_texts[idx]
        text_inputs = self.text_tokenizer(processed_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        text_input_ids = text_inputs['input_ids'].squeeze(0)
        text_attention_mask = text_inputs['attention_mask'].squeeze(0)

        label = self.labels[idx]

        return image, text_input_ids, text_attention_mask, label


# class MultiModalClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(MultiModalClassifier, self).__init__()
#         # 加载预训练模型
#         self.ViT = ViTModel.from_pretrained('./vit/google')
#         self.SciBERT = BertModel.from_pretrained('scibert_scivocab_uncased', return_dict=True)
#
#         # 跨模态注意力层
#         self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
#
#         # 分类器
#         self.fc = nn.Linear(768, num_classes)
#
#         # 其他层
#         self.norm = nn.LayerNorm(768)
#         self.dropout = nn.Dropout(0.1)
#         self.relu = nn.GELU()
#
#         # 参数初始化
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_uniform_(module.weight)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.ones_(module.weight)
#             nn.init.zeros_(module.bias)
#
#     def forward(self, images, input_ids, attention_mask):
#         # 文本特征提取
#         text_outputs = self.SciBERT(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
#
#         # 图像特征提取
#         image_outputs = self.ViT(images, return_dict=True)
#         image_features = image_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
#
#         # 跨模态注意力
#         # 将文本特征作为 Query，图像特征作为 Key 和 Value
#         attn_output, _ = self.cross_attention(query=text_features, key=image_features, value=image_features)
#         attn_output = self.norm(attn_output + text_features)
#         attn_output = self.relu(attn_output)
#         attn_output = self.dropout(attn_output)
#
#         # 方法1：使用 [CLS] token
#         pooled_output = attn_output[:, 0, :]  # [batch_size, hidden_size]
#         # 池化（取平均）
#         #pooled_output = torch.mean(attn_output, dim=1)  # [batch_size, hidden_size]
#
#         # 分类器
#         logits = self.fc(pooled_output)
#
#         return logits

class MultiModalClassifier(nn.Module):#不用cross-attention
    def __init__(self, num_classes):
        super(MultiModalClassifier, self).__init__()
        # 加载预训练模型
        self.ViT = ViTModel.from_pretrained('./vit/google')
        self.SciBERT = BertModel.from_pretrained('scibert_scivocab_uncased', return_dict=True)

        # 归一化层
        self.norm_text = nn.LayerNorm(768)
        self.norm_image = nn.LayerNorm(768)

        # 激活函数
        self.relu = nn.GELU()

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # 分类器
        self.fc = nn.Linear(768 + 768, num_classes)

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images, input_ids, attention_mask):
        # 文本特征提取
        text_outputs = self.SciBERT(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_features = text_outputs.pooler_output  # [batch_size, hidden_size]

        # 图像特征提取
        image_outputs = self.ViT(images, return_dict=True)
        image_features = image_outputs.pooler_output  # [batch_size, hidden_size]

        # 归一化
        text_features = self.norm_text(text_features)
        image_features = self.norm_image(image_features)

        # 激活和 Dropout
        text_features = self.relu(text_features)
        text_features = self.dropout(text_features)

        image_features = self.relu(image_features)
        image_features = self.dropout(image_features)

        # 特征融合：拼接文本和图像特征
        pooled_output = torch.cat((text_features, image_features), dim=-1)  # [batch_size, hidden_size * 2]

        # 分类器
        logits = self.fc(pooled_output)  # [batch_size, num_classes]

        return logits

import sys

import torch
from sklearn import metrics
from tqdm import tqdm

from eval.evaluate import metric as utils_evaluate_metric
from eval.evaluate import metric_multitask as utils_evaluate_metric_multitask
from eval.evaluate import metric_reg as utils_evaluate_metric_reg
from eval.evaluate import metric_reg_multitask as utils_evaluate_metric_reg_multitask


def metric(y_true, y_pred, y_prob):
    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob)
    return {
        "accuracy": acc,
        "ROCAUC": auc
    }


def train_one_epoch_multitask(model, optimizer, data_loader, criterion, device, epoch, task_type, tqdm_desc=""):
    assert task_type in ["classification", "regression"]

    model.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, desc=tqdm_desc)
    for step, data in enumerate(data_loader):
        image, text_input_ids, text_attention_mask, labels = data
        image, text_input_ids, text_attention_mask, labels = image.to(device), text_input_ids.to(
            device), text_attention_mask.to(
            device), labels.to(device)
        sample_num += text_input_ids.size(0)

        pred = model(image, text_input_ids, text_attention_mask)
        # print(pred.shape)
        # print(labels)
        labels = labels.view(pred.shape).to(torch.float64)
        if task_type == "classification":
            is_valid = labels != -1
            loss_mat = criterion(pred.double(), labels)
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
        elif task_type == "regression":
            loss = criterion(pred.double(), labels)

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate_on_multitask(model, data_loader, criterion, device, epoch, task_type="classification", tqdm_desc="", type="valid"):
    assert task_type in ["classification", "regression"]

    model.eval()

    accu_loss = torch.zeros(1).to(device)

    y_scores, y_true, y_pred, y_prob = [], [], [], []
    sample_num = 0
    data_loader = tqdm(data_loader, desc=tqdm_desc)
    for step, data in enumerate(data_loader):
        image,text_input_ids, text_attention_mask, labels = data
        image,text_input_ids, text_attention_mask, labels = image.to(device),text_input_ids.to(device), text_attention_mask.to(
            device), labels.to(device)
        sample_num += text_input_ids.size(0)

        with torch.no_grad():
            pred = model(image, text_input_ids, text_attention_mask)

            labels = labels.view(pred.shape).to(torch.float64)
            if task_type == "classification":
                is_valid = labels != -1
                loss_mat = criterion(pred.double(), labels)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            elif task_type == "regression":
                loss = criterion(pred.double(), labels)
            accu_loss += loss.detach()
            data_loader.desc = "[evaluation epoch {}] {} loss: {:.3f}".format(epoch, type, accu_loss.item() / (step + 1))

        y_true.append(labels.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if y_true.shape[1] == 1:
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            return accu_loss.item() / (step + 1), utils_evaluate_metric(y_true, y_pred, y_pro, empty=-1)
        #y_true:label  y_pred:根据阈值置1或0 y_pro：根据预测分数使用sigmoid函数确定概率
        elif task_type == "regression":
            return accu_loss.item() / (step + 1), utils_evaluate_metric_reg(y_true, y_scores)
    elif y_true.shape[1] > 1:  # multi-task
        if task_type == "classification":
            y_pro = torch.sigmoid(torch.Tensor(y_scores))
            y_pred = torch.where(y_pro > 0.5, torch.Tensor([1]), torch.Tensor([0])).numpy()
            return accu_loss.item() / (step + 1), utils_evaluate_metric_multitask(y_true, y_pred, y_pro, num_tasks=y_true.shape[1], empty=-1)
        elif task_type == "regression":
            return accu_loss.item() / (step + 1), utils_evaluate_metric_reg_multitask(y_true, y_scores, num_tasks=y_true.shape[1])
    else:
        raise Exception("error in the number of task.")
