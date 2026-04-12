# src/evaluate.py
import torch
from torch.utils.data import DataLoader
from src.data_loader import DeepfakeSequenceDataset
from src.model import CNNFeatureExtractor, CNN_LSTM_Attention
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate(checkpoint_path, data_root, seq_len=8, batch_size=4, device='cuda'):
    dataset = DeepfakeSequenceDataset(data_root, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    feat_extractor = CNNFeatureExtractor().to(device)
    model = CNN_LSTM_Attention(feat_dim=feat_extractor.out_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_extractor.load_state_dict(ckpt['feat_state'])
    model.load_state_dict(ckpt['model_state'])
    feat_extractor.eval(); model.eval()
    ys = []; ys_pred = []
    with torch.no_grad():
        for seqs, labels in loader:
            B, S, C, H, W = seqs.shape
            seqs = seqs.view(B*S, C, H, W).to(device)
            features = feat_extractor(seqs)
            features = features.view(B, S, -1)
            logits, _ = model(features)
            preds = logits.argmax(dim=1).cpu().numpy()
            ys_pred.extend(preds.tolist())
            ys.extend(labels.numpy().tolist())
    print(classification_report(ys, ys_pred, target_names=['real','fake']))
    print("Confusion matrix:\n", confusion_matrix(ys, ys_pred))
