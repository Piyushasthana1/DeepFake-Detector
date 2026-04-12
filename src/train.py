# src/train.py
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_loader import DeepfakeSequenceDataset
from src.model import CNNFeatureExtractor, CNN_LSTM_Attention
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def train_epoch(feat_extractor, model, dataloader, criterion, optimizer, device):
    feat_extractor.train(); model.train()
    losses = []
    ys = []; ypred = []
    for seqs, labels in dataloader:
        B,S,C,H,W = seqs.shape
        seqs = seqs.view(B*S, C, H, W).to(device)
        with torch.set_grad_enabled(True):
            feats = feat_extractor(seqs)  # (B*S, feat_dim)
            feats = feats.view(B, S, -1).to(device)
            logits, _ = model(feats)
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        ys.extend(labels.numpy().tolist())
        ypred.extend(preds.tolist())
    acc = accuracy_score(ys, ypred) if len(ys)>0 else 0.0
    return np.mean(losses) if losses else 0.0, acc

@torch.no_grad()
def validate(feat_extractor, model, dataloader, criterion, device):
    feat_extractor.eval(); model.eval()
    losses = []
    ys = []; ypred = []
    attn_weights = []
    for seqs, labels in dataloader:
        B,S,C,H,W = seqs.shape
        seqs = seqs.view(B*S, C, H, W).to(device)
        feats = feat_extractor(seqs)
        feats = feats.view(B, S, -1).to(device)
        logits, attn = model(feats)
        loss = criterion(logits, labels.to(device))
        losses.append(loss.item())
        preds = logits.argmax(dim=1).cpu().numpy()
        ys.extend(labels.numpy().tolist())
        ypred.extend(preds.tolist())
        attn_weights.extend(attn.squeeze(-1).cpu().numpy().tolist())
    acc = accuracy_score(ys, ypred) if len(ys)>0 else 0.0
    return np.mean(losses) if losses else 0.0, acc, ys, ypred, attn_weights

def set_requires_grad_backbone(feat_extractor, requires_grad):
    for p in feat_extractor.parameters():
        p.requires_grad = requires_grad

def main(args):
    device = 'cuda' if (torch.cuda.is_available() and args.device=='cuda') else 'cpu'
    print("Using device:", device)
    # dataset expects data/<real|fake>/<video_id> with frames inside
    train_root = os.path.join(args.data_root, 'train') if args.use_split_csv else args.data_root
    # If you used create_splits.csv, the dataset class adaptation is needed. For simplicity we reuse full folders.
    dataset = DeepfakeSequenceDataset(args.data_root, seq_len=args.seq_len)
    # NOTE: if you created splits.csv and want train/val, modify dataset class or use separate roots
    # For now we do an 80/20 split here:
    n = len(dataset)
    idxs = list(range(n))
    split = int(0.8 * n)
    train_idx = idxs[:split]
    val_idx = idxs[split:]
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    feat_extractor = CNNFeatureExtractor().to(device)
    model = CNN_LSTM_Attention(feat_dim=feat_extractor.out_dim).to(device)

    # Optionally freeze backbone initially
    if args.freeze_backbone:
        print("Freezing backbone parameters")
        set_requires_grad_backbone(feat_extractor, False)

    params = list(model.parameters()) + [p for p in feat_extractor.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    # create scheduler — handle compatibility for 'verbose' kwarg across torch versions
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    except TypeError:
    # older/newer torch versions may not accept 'verbose' argument
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = []
    start_epoch = 0

    # resume if provided
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        feat_extractor.load_state_dict(ckpt['feat_state'])
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        # try to load optimizer state if compatible
        try:
            optim_state = ckpt.get('optim_state', None)
            if optim_state is not None:
                optimizer.load_state_dict(optim_state)
                print("Optimizer state loaded from checkpoint.")
            else:
                print("No optimizer state found in checkpoint; starting with fresh optimizer.")
        except ValueError as e:
            # optimizer param groups mismatch (common when unfreezing or changing params)
            print("Warning: could not load optimizer state from checkpoint (mismatched parameter groups).")
            print("Reason:", e)
            print("Continuing with a freshly initialized optimizer (optimizer state skipped).")
        print("Resumed model weights from", args.resume, "starting epoch", start_epoch)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(feat_extractor, model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, ys, ypred, _ = validate(feat_extractor, model, val_loader, criterion, device)
        scheduler.step(val_loss)
        t1 = time.time()
        elapsed = t1 - t0
        print(f"Epoch {epoch+1}/{args.epochs}  TrainLoss {train_loss:.4f} TrainAcc {train_acc:.4f}  ValLoss {val_loss:.4f} ValAcc {val_acc:.4f}  Time {elapsed:.1f}s")
        row = {'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'lr': optimizer.param_groups[0]['lr']}
        history.append(row)
        # save CSV log
        os.makedirs('experiments/logs', exist_ok=True)
        df = pd.DataFrame(history)
        df.to_csv('experiments/logs/train_log.csv', index=False)

        # checkpoint
        ckpt_path = f"experiments/checkpoints/epoch_{epoch+1}.pth"
        save_checkpoint({
            'epoch': epoch,
            'feat_state': feat_extractor.state_dict(),
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }, ckpt_path)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'feat_state': feat_extractor.state_dict(),
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, "experiments/checkpoints/best.pth")

        # optionally unfreeze backbone mid-training
        if args.unfreeze_epoch and (epoch+1) == args.unfreeze_epoch:
            print("Unfreezing backbone at epoch", epoch+1)
            set_requires_grad_backbone(feat_extractor, True)
            # rebuild optimizer to include backbone params
            params = list(model.parameters()) + [p for p in feat_extractor.parameters() if p.requires_grad]
            optimizer = optim.Adam(params, lr=args.lr*0.5)
            # reattach scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    print("Training finished. Best val acc:", best_val_acc)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=str, help="root data folder (contains real/ and fake/ subfolders)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze_backbone", action='store_true')
    p.add_argument("--unfreeze_epoch", type=int, default=0, help="epoch number (1-indexed) to unfreeze backbone; 0 = never")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use_split_csv", action='store_true', help="(future) use splits.csv")
    args = p.parse_args()
    main(args)
