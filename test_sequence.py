import os
import argparse
import random
import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score
from torch.utils.data import DataLoader
from model import *


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PtFeatureDataset(torch.utils.data.Dataset):
    """
    Build dataset by scanning .pt files in pos_dir and neg_dir.
    pos_dir/*.pt -> label 1
    neg_dir/*.pt -> label 0
    """

    def __init__(self, pos_dir: str, neg_dir: str):
        self.items = []

        # collect all .pt files
        if pos_dir and os.path.isdir(pos_dir):
            pos_files = sorted([f for f in os.listdir(pos_dir) if f.endswith(".pt")])
            for fname in pos_files:
                self.items.append((os.path.join(pos_dir, fname), 1))

        if neg_dir and os.path.isdir(neg_dir):
            neg_files = sorted([f for f in os.listdir(neg_dir) if f.endswith(".pt")])
            for fname in neg_files:
                self.items.append((os.path.join(neg_dir, fname), 0))

        if len(self.items) == 0:
            raise RuntimeError(
                f"No .pt files found. pos_dir={pos_dir}, neg_dir={neg_dir}"
            )

        # probe to get shape
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

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    return avg_loss, acc, f1, prec, y_true, y_pred, None


def main():
    parser = argparse.ArgumentParser(description="Test Transformer classifier (binary/multi) on ESM features.")

    # removed pos_txt / neg_txt
    parser.add_argument('--pos_dir', default='./sequence_features/pos_esm2_fea',
                        help='positive .pt features directory')
    parser.add_argument('--neg_dir', default='./sequence_features/neg_esm2_fea',
                        help='negative .pt features directory')

    parser.add_argument('--task', default='binary', choices=['binary', 'multi'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default=None, help='cpu / cuda / cuda:0 ...')
    parser.add_argument('--load_model', default='./checkpoint/sequence/best.pth', help='Path to pre-trained model')
    parser.add_argument('--out', default='results', help='directory to save outputs')

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_dataset = PtFeatureDataset(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none
    )

    num_classes = 2 if args.task == 'binary' else len(set([y for _, y in test_dataset.items]))
    print(f"seq_len={test_dataset.seq_len}, embed_dim={test_dataset.embed_dim}, num_classes={num_classes}")
    print(f"Loaded items: {len(test_dataset)} (pos+neg)")

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

    state_dict = torch.load(args.load_model, map_location="cpu")

    if len(state_dict) > 0 and 'module.' in next(iter(state_dict.keys())):
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
