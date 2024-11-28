import torch
from model.model import ImageMol
from dataloader.image_dataloader import Smiles2Img
from model.cnn_model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt
from torchvision import transforms
import os

smiles_list = "CCO"
image = Smiles2Img(smiles_list, size=224, savePath=None)

# 指定使用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义预处理转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])
def createVisualModel():
# 定义模型并将其移动到指定的设备
    model = load_model()

    model.to(device)
    # 模型预训练权重的路径
    model_path = 'datasets/pretraining/data/ImageMol.pth.tar'

    # 检查预训练模型文件是否存在并加载
    if os.path.isfile(model_path):
        print(f"=> loading checkpoint '{model_path}'")
        # 确保在加载时也指定了正确的设备
        checkpoint = torch.load(model_path, map_location=device)

        ckp_keys = list(checkpoint['state_dict'])
        cur_keys = list(model.state_dict())
        model_sd = model.state_dict()

        ckp_keys = ckp_keys[:120]
        cur_keys = cur_keys[:120]

        for ckp_key, cur_key in zip(ckp_keys, cur_keys):
            model_sd[cur_key] = checkpoint['state_dict'][ckp_key]

        model.load_state_dict(model_sd)
        arch = checkpoint['arch']
        print("resume model info: arch: {}".format(arch))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    return model.eval()

 # 应用预处理转换
image = transform(image).unsqueeze(0).to(device)  # 增加批次维度

# 确保模型处于评估模式
model.eval()

# 无梯度环境下进行预测
with torch.no_grad():
    output = model(image)
    print(f"Output shape: {output.shape}")



