# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone='efficientnet_b0', pretrained=True):
        super().__init__()
        if backbone.startswith('resnet'):
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Identity()
            self.out_dim = 512
        else:
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier = nn.Identity()
            self.out_dim = 1280

    def forward(self, x):
        # x: (B, C, H, W)
        return self.backbone(x)  # (B, feat_dim)

class CNN_LSTM_Attention(nn.Module):
    def __init__(self, feat_dim=1280, hidden_dim=256, num_layers=1, num_classes=2, dropout=0.3):
        super().__init__()
        self.feat_dim = feat_dim
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(hidden_dim*2, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, feats_seq):
        # feats_seq: (B, seq_len, feat_dim)
        out, _ = self.lstm(feats_seq)  # (B, seq_len, hidden*2)
        # Attention weights:
        attn_scores = self.attn_fc(out).squeeze(-1)  # (B, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, seq_len, 1)
        context = (out * attn_weights).sum(dim=1)  # (B, hidden*2)
        logits = self.classifier(context)
        return logits, attn_weights
