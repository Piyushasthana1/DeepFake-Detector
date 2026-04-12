# src/data_loader.py
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as T
import glob
import numpy as np

class DeepfakeSequenceDataset(Dataset):
    """
    Expects folder structure:
    data/
      real/
        video1/
          frame_000001.jpg ...
      fake/
        video2/
    Each sample yields a tensor of shape (seq_len, C, H, W)
    """
    def __init__(self, root_dir, seq_len=8, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.samples = []
        for label, cls in enumerate(['real','fake']):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for vid in os.listdir(cls_dir):
                frames = sorted(glob.glob(os.path.join(cls_dir, vid, '*.jpg')))
                if len(frames) >= seq_len:
                    self.samples.append((frames, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]
        # sample uniformly seq_len frames
        total = len(frames)
        indices = np.linspace(0, total-1, self.seq_len).astype(int)
        imgs = []
        for i in indices:
            img = Image.open(frames[i]).convert('RGB')
            imgs.append(self.transform(img))
        seq = torch.stack(imgs)  # shape: (seq_len, C, H, W)
        return seq, torch.tensor(label, dtype=torch.long)
