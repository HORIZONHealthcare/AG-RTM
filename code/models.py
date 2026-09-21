import timm
import torch
import torch.nn as nn


MODEL_REGISTRY = {
    # DINOv2 (patch14)
    ("dinov2", "small"): "vit_small_patch14_dinov2.lvd142m",
    ("dinov2", "base"): "vit_base_patch14_dinov2.lvd142m",
    ("dinov2", "large"): "vit_large_patch14_dinov2.lvd142m",
    ("dinov2", "giant"): "vit_giant_patch14_dinov2.lvd142m",
    # DINOv3 (patch16)
    ("dinov3", "small"): "vit_small_patch16_dinov3.lvd1689m",
    ("dinov3", "base"): "vit_base_patch16_dinov3.lvd1689m",
    ("dinov3", "large"): "vit_large_patch16_dinov3.lvd1689m",
    ("dinov3", "huge"): "vit_huge_plus_patch16_dinov3.lvd1689m",
}


class BiomarkerModel(nn.Module):
    def __init__(self, version="dinov2", model="large", pretrained=True, img_size=224,
                 head_hidden_dim=32, head_dropout=0.5):
        super().__init__()
        model_name = MODEL_REGISTRY[(version, model)]
        self.backbone = timm.create_model(model_name, pretrained=pretrained, img_size=img_size)
        self.head = PredictionHead(
            embed_dim=self.backbone.embed_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    def set_training_mode(self, mode, k=0):
        if mode == 'linear_probe':
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif mode == 'finetune_last_k':
            for param in self.backbone.parameters():
                param.requires_grad = False
            for block in self.backbone.blocks[-k:]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, 'fc_norm'):
                for param in self.backbone.fc_norm.parameters():
                    param.requires_grad = True
        # finetune_all: all params require grad by default

    def no_weight_decay(self):
        nwd = set()
        if hasattr(self.backbone, 'no_weight_decay'):
            for name in self.backbone.no_weight_decay():
                nwd.add(f'backbone.{name}')
        return nwd

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = x[:, self.backbone.num_prefix_tokens:, :].mean(dim=1)
        x = self.backbone.fc_norm(x)
        return self.head(x)


class PredictionHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1, bias=True)
        )

    def forward(self, x):
        return self.layers(x)
