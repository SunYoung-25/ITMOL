import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel, AutoConfig

class SciBERTClassifier(nn.Module):
    def __init__(self, model_name, config, num_labels):
        super(SciBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # 取出[CLS]标记的输出，通常用于分类任务
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
class TextDataset(Dataset):
    def __init__(self, root, text_tokenizer_path, stopwords_file_path, max_len=300):
        self.df = pd.read_csv(root)  # 加载CSV文件
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
        self.max_len = max_len

        # 处理标签
        self.labels = self.get_label(self.df)
        self.num_tasks = self.labels.shape[1]

        # 读取并存储停用词
        with open(stopwords_file_path, 'r') as file:
            self.stop_words = set(file.read().splitlines())

        # 预处理所有文本
        self.preprocessed_texts = [self.preprocess_text(text) for text in self.df['description']]

    def preprocess_text(self, text):
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def get_label(self, df):
        return np.array(df["label"].apply(lambda x: np.array(str(x).split()).astype(int).tolist()).tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        processed_text = self.preprocessed_texts[idx]
        text_inputs = self.text_tokenizer(processed_text, max_length=self.max_len,
                                          padding="max_length", truncation=True,
                                          return_tensors="pt")
        text_input_ids = text_inputs['input_ids'].squeeze(0)
        text_attention_mask = text_inputs['attention_mask'].squeeze(0)

        # 获取标签
        labels = self.labels[idx]

        return text_input_ids, text_attention_mask, labels

# Example usage:
# dataset = TextDataset('data.csv', 'bert-base-uncased', 'stopwords.txt')
# from torch.utils.data import DataLoader
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
# for input_ids, attention_mask, labels in loader:
#     print(input_ids, attention_mask, labels)
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
        text_input_ids, text_attention_mask, labels = data
        text_input_ids, text_attention_mask,labels = text_input_ids.to(device),text_attention_mask.to(device), labels.to(device)
        sample_num += text_input_ids.size(0)

        pred = model(input_ids=text_input_ids, attention_mask=text_attention_mask)
        #print(pred.shape)
        #print(labels)
        labels = labels.view(pred.shape).to(torch.float64)
        if task_type == "classification":
            is_valid = labels != -1
            loss_mat = criterion(pred.double(), labels)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
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
        text_input_ids, text_attention_mask, labels = data
        text_input_ids, text_attention_mask, labels = text_input_ids.to(device), text_attention_mask.to(
            device), labels.to(device)
        sample_num += text_input_ids.size(0)

        with torch.no_grad():
            pred = model(input_ids=text_input_ids, attention_mask=text_attention_mask)

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
