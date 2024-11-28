import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttentionLayer, self).__init__()
        # 自注意力和跨注意力层分别为 x 和 y
        self.self_attn_x = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn_x = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.self_attn_y = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn_y = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # 前馈网络
        self.linear1_x = nn.Linear(d_model, d_model)
        self.linear2_x = nn.Linear(d_model, d_model)
        self.linear1_y = nn.Linear(d_model, d_model)
        self.linear2_y = nn.Linear(d_model, d_model)

        # 归一化层
        self.norm1_x = nn.LayerNorm(d_model)
        self.norm2_x = nn.LayerNorm(d_model)
        self.norm3_x = nn.LayerNorm(d_model)
        self.norm1_y = nn.LayerNorm(d_model)
        self.norm2_y = nn.LayerNorm(d_model)
        self.norm3_y = nn.LayerNorm(d_model)

        # 丢弃层
        self.dropout1_x = nn.Dropout(0.1)
        self.dropout2_x = nn.Dropout(0.1)
        self.dropout3_x = nn.Dropout(0.1)
        self.dropout1_y = nn.Dropout(0.1)
        self.dropout2_y = nn.Dropout(0.1)
        self.dropout3_y = nn.Dropout(0.1)

        # 新增：用于存储注意力权重
        self.attn_weights = None

    def forward(self, x, y):
        # 处理 x 的自注意力
        x2 = self.self_attn_x(x, x, x)[0]
        x_sa = x + self.dropout1_x(x2)
        x_sa = self.norm1_x(x_sa)

        # 处理 y 的自注意力
        y2 = self.self_attn_y(y, y, y)[0]
        y_sa = y + self.dropout1_y(y2)
        y_sa = self.norm1_y(y_sa)

        # 处理 x 的跨注意力，使用 y_sa
        # 获取 cross-attention 的输出和注意力权重
        x2, attn_weights = self.cross_attn_x(
            x_sa, y_sa, y_sa, need_weights=True, average_attn_weights=False
        )

        # 保留 attn_weights 的梯度
        attn_weights.retain_grad()

        # 存储 attn_weights 以便在自定义损失中访问
        self.attn_weights = attn_weights

        # 将 attn_weights 与 x2 分开，继续正常前向传播
        x = x_sa + self.dropout2_x(x2)
        x = self.norm2_x(x)

        # 处理 y 的跨注意力，使用 x_sa（无需更改）
        y2 = self.cross_attn_y(y_sa, x_sa, x_sa)[0]
        y = y_sa + self.dropout2_y(y2)
        y = self.norm2_y(y)

        # 处理 x 的前馈网络
        x2 = self.linear2_x(self.dropout3_x(F.relu(self.linear1_x(x))))
        x = x + x2
        x = self.norm3_x(x)

        # 处理 y 的前馈网络
        y2 = self.linear2_y(self.dropout3_y(F.relu(self.linear1_y(y))))
        y = y + y2
        y = self.norm3_y(y)

        return x, y




class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.08):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        # 归一化图像和文本特征
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # 计算相似度矩阵
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        # 为了数值稳定性，减去最大值
        logits_per_image_max, _ = logits_per_image.max(dim=1, keepdim=True)
        logits_per_image = logits_per_image - logits_per_image_max.detach()

        logits_per_text_max, _ = logits_per_text.max(dim=1, keepdim=True)
        logits_per_text = logits_per_text - logits_per_text_max.detach()

        # 创建正确的标签
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device, dtype=torch.long)

        # 计算交叉熵损失
        loss_i2t = F.cross_entropy(logits_per_image, labels, reduction='mean')
        loss_t2i = F.cross_entropy(logits_per_text, labels, reduction='mean')

        # 返回平均损失
        return (loss_i2t + loss_t2i) / 2


class MultiModalModel(nn.Module):
    def __init__(self, image_encoder_pretrained_path, text_encoder_model_path):
        super(MultiModalModel, self).__init__()

        # 加载 Vision Transformer 模型
        self.ViT = ViTModel.from_pretrained(image_encoder_pretrained_path)

        # 加载 BERT 模型
        self.SciBERT = BertModel.from_pretrained(text_encoder_model_path)

        # 添加用于 MLM 的输出层
        self.mlm_head = nn.Linear(self.SciBERT.config.hidden_size, self.SciBERT.config.vocab_size)

        # 交叉注意力层
        self.cross_modal_image_text_layers = nn.ModuleList(
            [CrossAttentionLayer(d_model=768, nhead=8) for _ in range(3)]
        )

        self.last_cross_attention_layer = self.cross_modal_image_text_layers[-1]

        # 投影层
        self.projector_image = nn.Linear(768, 768)
        self.projector_text = nn.Linear(768, 768)

        # 用于 CMM 的分类器
        self.cmm_classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        # 初始化权重参数
        self.log_alpha_mlm = nn.Parameter(torch.tensor(-0.5))  # 初始值为负数，表示权重<1
        self.log_alpha_cmm = nn.Parameter(torch.tensor(0.0))  # 初始值为0，表示权重=1
        self.log_alpha_clip = nn.Parameter(torch.tensor(-1.0))  # 初始值为负数，表示权重<1

        # CLIP loss
        self.clip_loss_fn = CLIPLoss()

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
            self,
            images,
            text_input_ids,
            text_attention_mask,
            mlm_labels=None,
            negative_text_input_ids=None,
            negative_text_attention_mask=None
    ):
        # 获取当前设备
        device = images.device

        # 图像编码
        image_outputs = self.ViT(images)
        image_features = image_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 文本编码
        text_outputs = self.SciBERT(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_dict=True
        )
        text_embedding = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 提取 [CLS] token 的特征（在跨模态注意力之前）
        cls_feats_image = image_features[:, 0, :]  # [batch_size, hidden_size]
        cls_feats_text = text_embedding[:, 0, :]  # [batch_size, hidden_size]

        # 经过投影层并归一化
        image_projected = F.normalize(self.projector_image(cls_feats_image), dim=-1, eps=1e-8)
        text_projected = F.normalize(self.projector_text(cls_feats_text), dim=-1, eps=1e-8)

        # 计算 CLIP loss（在跨模态注意力之前）
        clip_loss = self.clip_loss_fn(image_projected, text_projected)

        # 处理负样本（用于 CMM 任务）
        if negative_text_input_ids is not None:
            negative_text_outputs = self.SciBERT(
                input_ids=negative_text_input_ids,
                attention_mask=negative_text_attention_mask,
                return_dict=True
            )
            negative_text_embedding = negative_text_outputs.last_hidden_state
        else:
            negative_text_embedding = None

        # 交叉注意力处理（使用原始的图像和文本特征）
        img_feats, txt_feats = image_features, text_embedding
        for layer in self.cross_modal_image_text_layers:
            img_feats, txt_feats = layer(img_feats, txt_feats)

        # 提取跨模态后的 [CLS] token 特征
        cls_feats_image_cmm = img_feats[:, 0, :]  # [batch_size, hidden_size]
        cls_feats_text_cmm = txt_feats[:, 0, :]  # [batch_size, hidden_size]

        # 经过投影层并归一化（用于 CMM 任务）
        image_projected_cmm = F.normalize(self.projector_image(cls_feats_image_cmm), dim=-1, eps=1e-8)
        text_projected_cmm = F.normalize(self.projector_text(cls_feats_text_cmm), dim=-1, eps=1e-8)

        # 计算 MLM 任务的损失
        if mlm_labels is not None:
            mlm_logits = self.mlm_head(txt_feats)  # 使用跨模态后的文本特征
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, self.SciBERT.config.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100
            )
        else:
            mlm_loss = torch.tensor(0.0, device=device)

        # 计算 CMM 任务的损失（使用二分类方法）
        if negative_text_embedding is not None:
            # 负样本也经过跨模态注意力层
            neg_img_feats, neg_txt_feats = image_features, negative_text_embedding
            for layer in self.cross_modal_image_text_layers:
                neg_img_feats, neg_txt_feats = layer(neg_img_feats, neg_txt_feats)

            cls_feats_negative_text = neg_txt_feats[:, 0, :]
            negative_text_projected_cmm = F.normalize(
                self.projector_text(cls_feats_negative_text), dim=-1, eps=1e-8
            )

            # 拼接正样本和负样本的特征
            positive_pair = torch.cat([image_projected_cmm, text_projected_cmm], dim=-1)
            negative_pair = torch.cat([image_projected_cmm, negative_text_projected_cmm], dim=-1)

            # 通过分类器进行二分类
            positive_logits = self.cmm_classifier(positive_pair)
            negative_logits = self.cmm_classifier(negative_pair)

            # 构造标签（正样本标签为1，负样本标签为0）
            labels = torch.cat([
                torch.ones_like(positive_logits),
                torch.zeros_like(negative_logits)
            ], dim=0).to(device)  # 确保标签在正确的设备上

            # 计算二分类的 BCE 损失
            cmm_loss = F.binary_cross_entropy_with_logits(
                torch.cat([positive_logits, negative_logits], dim=0),
                labels
            )
        else:
            cmm_loss = torch.tensor(0.0, device=device)

        # 计算总损失，调整权重
        total_loss = (
                torch.exp(-self.log_alpha_mlm) * mlm_loss +
                torch.exp(-self.log_alpha_cmm) * cmm_loss +
                torch.exp(-self.log_alpha_clip) * clip_loss +
                0.5 * (self.log_alpha_mlm + self.log_alpha_cmm + self.log_alpha_clip)
        )

        return total_loss, mlm_loss, cmm_loss, clip_loss

