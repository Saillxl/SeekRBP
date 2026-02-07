import os
import argparse
import random
from glob import glob
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as Data
from torch import nn, optim
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score
from model import *  


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_all_paths(pos_dir: str, neg_dir: str):
    pos_files = sorted(glob(str(Path(pos_dir) / "*.npy")))
    neg_files = sorted(glob(str(Path(neg_dir) / "*.npy")))
    items = [(p, 1) for p in pos_files] + [(n, 0) for n in neg_files]
    if not items:
        raise RuntimeError(".npy feature file was not found in feature_pos / feature_neg.")
    return items


def stratified_split(items, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.Random(seed).shuffle(items)

    pos = [it for it in items if it[1] == 1]
    neg = [it for it in items if it[1] == 0]

    def split_one(cls_items):
        n = len(cls_items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        return (cls_items[:n_train], cls_items[n_train:n_train+n_val], cls_items[n_train+n_val:])

    pos_tr, pos_va, pos_te = split_one(pos)
    neg_tr, neg_va, neg_te = split_one(neg)

    train = pos_tr + neg_tr
    val   = pos_va + neg_va
    test  = pos_te + neg_te

    random.Random(seed + 1).shuffle(train)
    random.Random(seed + 2).shuffle(val)
    random.Random(seed + 3).shuffle(test)
    return train, val, test


class NpyDirDataset(Data.Dataset):
    def __init__(self, items):
        self.items = items
        probe = np.load(self.items[0][0], mmap_mode="r")
        if probe.ndim == 3 and probe.shape[0] == 1:
            L, D = probe.shape[1], probe.shape[2]
        elif probe.ndim == 2:
            L, D = probe.shape
        else:
            raise ValueError(f"Expected feature shape [1, L, D] or [L, D], got {probe.shape} @ {self.items[0][0]}")
        self.seq_len = int(L)
        self.embed_dim = int(D)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        arr = np.load(path)  # -> (1,L,D) or (L,D)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]  # -> (L,D)
        x = torch.from_numpy(arr).float()
        return x, torch.tensor(y, dtype=torch.long)


def build_loaders_from_dirs(pos_dir,
                            neg_dir,
                            batch_size=32,
                            num_workers=0,
                            seed=42):
    items = load_all_paths(pos_dir, neg_dir)
    train_items, val_items, test_items = stratified_split(items, 0.8, 0.1, seed)
    train_ds = NpyDirDataset(train_items)
    val_ds   = NpyDirDataset(val_items)
    test_ds  = NpyDirDataset(test_items)

    assert train_ds.seq_len == val_ds.seq_len == test_ds.seq_len, "L is not consistent, please confirm that the features have been unified to a fixed length"
    assert train_ds.embed_dim == val_ds.embed_dim == test_ds.embed_dim, "D is not consistent, please confirm that the feature dimensions are consistent"

    train_loader = Data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True
    )
    val_loader = Data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )
    test_loader = Data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )
    return train_loader, val_loader, test_loader, train_ds.seq_len, train_ds.embed_dim


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        outputs = model(x)           # Expected model to receive [B, L, D], output [B, C]
        loss = criterion(outputs, y)
        total_loss += loss.item()

        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average='macro')

    return avg_loss, acc, f1, prec, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train Transformer classifier on .npy features from feature_pos/feature_neg")
    parser.add_argument('--pos_dir', default='./structure_features/pos_train', help='positive .npy features directory')
    parser.add_argument('--neg_dir', default='./structure_features/neg_train', help='negative .npy features directory')
    parser.add_argument('--task', default='binary', choices=['binary', 'multi'])
    parser.add_argument('--nepoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--embed_size', type=int, default=128) 
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--save_name', default='transformer_binary.pth')
    parser.add_argument('--device', default=None, help='cpu / cuda / cuda:0 ...')
    parser.add_argument('--out', default='./results/train_structure', help='directory for saving logs and models')

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
  
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, seq_len, embed_dim = build_loaders_from_dirs(
        args.pos_dir, args.neg_dir,
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed
    )

    num_classes = 2 if args.task == 'binary' else len(set([y for _, y in train_loader.dataset]))
    print(f"seq_len={seq_len}, embed_dim={embed_dim}, num_classes={num_classes}")
    print(f"train/val/test = {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    # model
    model = Transformer(
        src_vocab_size=embed_dim,
        src_pad_idx=0,
        device=device,
        max_length=seq_len,
        dropout=args.dropout,
        out_dim=num_classes,
        embed_size=args.embed_size,
        heads=args.heads,
        num_layers=1,
        forward_expansion=4,
    )

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f'Use {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)

    model.to(device)

    # loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp))

    best_metric = 0.0
    save_path = os.path.join(args.out, args.save_name)

    # train
    for epoch in range(1, args.nepoch + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)  # [B, L, D]
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and args.amp)):
                logits = model(xb)       # [B, C]
                loss = criterion(logits, yb)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_acc, val_f1, val_prec, y_true, y_pred = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_prec={val_prec:.4f}")

        # save every 10 epochs
        if epoch % 10 == 0:
            path = os.path.join(args.out, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), path)

        # save the best
        if val_f1 > best_metric:
            best_metric = val_f1
            torch.save(model.state_dict(), save_path)
            print(f" -> Saved best to {save_path} (F1={best_metric:.4f})")
            # save the current validation set report
            with open(os.path.join(args.out, "val_report.txt"), "w") as fw:
                fw.write(classification_report(y_true, y_pred, digits=4))

    # after training, evaluate on the test set
    test_loss, test_acc, test_f1, test_prec, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} prec={test_prec:.4f}")
    print("Test report:")
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    with open(os.path.join(args.out, "test_report.txt"), "w") as fw:
        fw.write(report)
    print("Done.")


if __name__ == "__main__":
    main()