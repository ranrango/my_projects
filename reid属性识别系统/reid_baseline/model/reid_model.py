import torch
import torch.nn as nn
import torchvision.models as models

class ReIDModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, pretrained=True):
        super(ReIDModel, self).__init__()
        if hasattr(models, "ResNet50_Weights"):
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        else:
            backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # 去除 avgpool & fc

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(2048, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.bn = nn.BatchNorm1d(embedding_dim)
        self.bn.bias.requires_grad_(False)  # 不训练偏置

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        feat = self.embedding(x)
        feat_bn = self.bn(feat)
        logits = self.classifier(feat_bn)

        return feat_bn, logits
