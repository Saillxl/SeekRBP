import os
import argparse
import random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score
from torch.utils.data import DataLoader
from model import *  


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PtFeatureDataset(torch.utils.data.Dataset):

    def __init__(self, pos_txt: str, neg_txt: str, pos_dir: str, neg_dir: str):
        self.items = []
        with open(pos_txt, 'r') as f:
            pos_samples = [line.strip() for line in f.readlines()]
        with open(neg_txt, 'r') as f:
            neg_samples = [line.strip() for line in f.readlines()]

        # Based on name matching, .pt file
        for sample in pos_samples:
            pos_path = os.path.join(pos_dir, f"{sample}.pt")
            if os.path.exists(pos_path):
                self.items.append((pos_path, 1))

        for sample in neg_samples:
            neg_path = os.path.join(neg_dir, f"{sample}.pt")
            if os.path.exists(neg_path):
                self.items.append((neg_path, 0))

        probe = torch.load(self.items[0][0], map_location="cpu")

        self.seq_len = probe.shape[0]
        self.embed_dim = probe.shape[1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        try:
            x = torch.load(path, map_location="cpu").float()  # [L, D]
            return x, torch.tensor(y, dtype=torch.long)
        except FileNotFoundError:
            return None  


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None  
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys)


def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    y_score = [] 
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            x, y = batch
            x, y = x.to(device), y.to(device)

            outputs = model(x)  # [B, C]
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.cpu().numpy())


    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    if save_probs:
        return avg_loss, acc, f1, prec, y_true, y_pred, y_score
    else:
        return avg_loss, acc, f1, prec, y_true, y_pred, None


def main():
    parser = argparse.ArgumentParser(description="Test Transformer classifier (binary/multi) on ESM features.")
    parser.add_argument('--pos_txt',
                        default='./dataset/pos_trainval_sets/test_set.txt',
                        help='positive samples txt')
    parser.add_argument('--neg_txt',
                        default='./dataset/neg_trainval_sets/test_set.txt',
                        help='negative samples txt')
    parser.add_argument('--pos_dir', default='./pos_esm2_fea',
                        help='positive .pt features directory')
    parser.add_argument('--neg_dir', default='./neg_esm2_fea',
                        help='negative .pt features directory')
    parser.add_argument('--task', default='binary', choices=['binary', 'multi'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default=None, help='cpu / cuda / cuda:0 ...')
    parser.add_argument('--load_model', default='./results/train_sequence/transformer_binary.pth', help='Path to pre-trained model') 
    parser.add_argument('--out', default='out', help='directory to save outputs')

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_dataset = PtFeatureDataset(
        pos_txt=args.pos_txt, neg_txt=args.neg_txt,
        pos_dir=args.pos_dir, neg_dir=args.neg_dir
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none
    )

    num_classes = 2 if args.task == 'binary' else len(set([y for _, y in test_dataset]))
    print(f"seq_len={test_dataset.seq_len}, embed_dim={test_dataset.embed_dim}, num_classes={num_classes}")

    model = Transformer(
        src_vocab_size=test_dataset.embed_dim,
        src_pad_idx=0,
        device=device,
        max_length=test_dataset.seq_len,
        dropout=args.dropout,
        out_dim=num_classes,
        embed_size=args.embed_size,
        heads=args.heads,
        num_layers=1,
        forward_expansion=4,
    )

    state_dict = torch.load(args.load_model)

    if 'module.' in next(iter(state_dict.keys())):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)

    avg_loss, acc, f1, prec, y_true, y_pred, y_score = evaluate(
    model, test_loader, device
    )

    print(f"Test Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f}")
    print("Test Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    with open(os.path.join(args.out, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f}\n")
        f.write("Test Classification Report:\n")
        f.write(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()