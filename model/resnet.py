import torch
import torchvision
import timm
from transformers import BertModel, ViTModel
def get_support_visual_model_names():
    return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",'VIT']


def load_model(modelname="ResNet18", num_classes=2):
    assert modelname in get_support_visual_model_names()
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if modelname == "ResNet34":
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if modelname == "ResNet50":
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if modelname == "ResNet101":
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if modelname == "ResNet152":
        model = torchvision.models.resnet152(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if modelname == "VIT":
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)

    return model