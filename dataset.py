import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer,BertTokenizer
from torchvision import transforms

class MultimodalDataset(Dataset):  # 多模态数据集
    def __init__(self, csv_path, image_dir, text_tokenizer_path, transform=None, max_len=300, mask_prob=0.15):
        """
        初始化多模态数据集。
        Args:
            csv_path (str): 包含描述的 CSV 文件的路径。
            image_dir (str): 保存图片的目录路径。
            text_tokenizer_path (str): 文本分词器的预训练模型路径。
            transform (callable, optional): 应用于图片的变换。
            max_len (int): 文本处理的最大长度。
            mask_prob (float): 掩蔽的概率，用于生成 MLM 任务的标签。
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取文本描述
        text_data = self.df.iloc[idx]['description']

        # 获取图片路径并加载图片
        img_path = os.path.join(self.image_dir, f"{idx}.png")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 对文本进行 tokenization
        text_inputs = self.text_tokenizer(text_data, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        text_input_ids = text_inputs['input_ids'].squeeze(0)
        text_attention_mask = text_inputs['attention_mask'].squeeze(0)

        # 生成 MLM 任务的标签
        input_ids, mlm_labels = self.create_mlm_labels(text_input_ids)

        return image, input_ids, text_attention_mask, mlm_labels

    def create_mlm_labels(self, input_ids):
        """
        生成 MLM 任务的标签，并在 input_ids 中引入 [MASK] 标记。
        """
        labels = input_ids.clone()

        # 获取特殊token的mask，确保特殊token不被掩蔽
        special_tokens_mask = self.text_tokenizer.get_special_tokens_mask(
            input_ids.tolist(),
            already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        # 创建掩蔽的概率矩阵
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = masked_indices & ~special_tokens_mask  # 不掩蔽特殊token

        labels[~masked_indices] = -100  # 只计算被掩蔽部分的损失

        # 80% 的时间，将 masked_indices 替换为 [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.text_tokenizer.mask_token_id

        # 10% 的时间，将 masked_indices 替换为随机 token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.text_tokenizer), labels.shape, dtype=input_ids.dtype)
        input_ids[indices_random] = random_words[indices_random]

        # 剩下的 10% 保持原样

        return input_ids, labels

