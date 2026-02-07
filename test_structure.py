import os
import argparse
from glob import glob
from pathlib import Path
from collections import Counter
import random
import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score
from model import *  


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_all_as_test(pos_dir: str, neg_dir: str):

    pos_files = sorted(glob(str(Path(pos_dir) / "*.npy"))) if pos_dir else []
    neg_files = sorted(glob(str(Path(neg_dir) / "*.npy"))) if neg_dir else []
    items = [(p, 1) for p in pos_files] + [(n, 0) for n in neg_files]
    if not items:
        raise RuntimeError(".npy feature file was not found in feature_pos / feature_neg.")
    return items


class NpyDirDataset(Data.Dataset):
    def __init__(self, items):
        assert len(items) > 0, "Empty test items."
        self.items = items

        probe = np.load(self.items[0][0], mmap_mode="r", allow_pickle=False)
        if probe.ndim == 3 and probe.shape[0] == 1:
            L, D = probe.shape[1], probe.shape[2]
        elif probe.ndim == 2:
            L, D = probe.shape
        else:
            raise ValueError(f"Expected feature shape [1, L, D] or [L, D], obtained {probe.shape} @ {self.items[0][0]}")
        self.seq_len = int(L)
        self.embed_dim = int(D)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        arr = np.load(path, allow_pickle=False)  # -> (1,L,D) or (L,D)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]  # -> (L,D)
        x = torch.from_numpy(arr).float()
        return x, torch.tensor(y, dtype=torch.long), path


def make_test_loader(items, batch_size=64, num_workers=0, pin=True):
    ds = NpyDirDataset(items)
    loader = Data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin
    )
    return loader, ds.seq_len, ds.embed_dim


def count_by_label(items):
    return Counter([y for _, y in items])


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred, paths = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y, ps in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        outputs = model(x)           
        loss = criterion(outputs, y)
        total_loss += loss.item()
      
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy())
        paths.extend(ps)

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    return avg_loss, acc, f1, prec, y_true, y_pred, paths


def load_checkpoint_flex(model, ckpt_path, map_location="cpu"):
    state = torch.load(ckpt_path, map_location=map_location)

    try:
        model.load_state_dict(state, strict=True)
        return
    except Exception:
        pass

    if isinstance(state, dict):
        stripped = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        try:
            model.load_state_dict(stripped, strict=True)
            return
        except Exception:
            pass

        added = { (k if k.startswith("module.") else "module."+k): v for k, v in state.items() }
        model.load_state_dict(added, strict=True)  


def main():
    parser = argparse.ArgumentParser("Evaluate on ALL files under pos_dir/neg_dir (no split).")
    parser.add_argument('--pos_dir', default='./structure_features/pos_test', help='positive .npy features directory')
    parser.add_argument('--neg_dir', default='./structure_features/neg_test', help='negative .npy features directory')

    parser.add_argument('--task', default='binary', choices=['binary', 'multi'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--forward_expansion', type=int, default=4)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default=None, help='cpu / cuda / cuda:0 ...')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--ckpt', required=True, help='Model weight .pth path')
    parser.add_argument('--out', default='./results/test_structure', help='Directory for saving test logs and reports')
    parser.add_argument('--save_pred_csv', default=None, help='Optional: Export individual sample predictions CSV')
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == 'cuda')
    pin = True if use_cuda else False
    non_block = True if use_cuda else False
    print(f"Device: {device}")

    test_items = load_all_as_test(args.pos_dir, args.neg_dir)
    print(f"TEST total={len(test_items)}  dist={count_by_label(test_items)}")

    test_loader, seq_len, embed_dim = make_test_loader(test_items, batch_size=args.batch_size, num_workers=args.num_workers, pin=pin)
    num_classes = 2 if args.task == 'binary' else len({y for _, y in test_items})
    print(f"seq_len={seq_len}, embed_dim={embed_dim}, num_classes={num_classes}")
    assert args.embed_size % args.heads == 0, "The embed_size must be divisible by the heads."

    model = Transformer(
        src_vocab_size=embed_dim, 
        src_pad_idx=0,
        device=device,
        max_length=seq_len,
        dropout=args.dropout,
        out_dim=num_classes,
        embed_size=args.embed_size,
        heads=args.heads,
        num_layers=args.num_layers,
        forward_expansion=args.forward_expansion,
    ).to(device)

    load_checkpoint_flex(model, args.ckpt, map_location=device)
    print(f"Loaded checkpoint from: {args.ckpt}")

    with torch.cuda.amp.autocast(enabled=(use_cuda and args.amp)):
        test_loss, test_acc, test_f1, test_prec, y_true, y_pred, paths, y_score = evaluate(
    model, test_loader, device
    )

    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f} prec={test_prec:.4f}")
    print("Test report:")
    report = classification_report(y_true, y_pred, digits=4)
    print(report)

    with open(os.path.join(args.out, "test_report.txt"), "w", encoding="utf-8") as fw:
        fw.write(f"loss={test_loss:.6f} acc={test_acc:.6f} f1={test_f1:.6f} prec={test_prec:.6f}\n\n")
        fw.write(report)

    if args.save_pred_csv is not None:
        try:
            import pandas as pd
            os.makedirs(os.path.dirname(args.save_pred_csv), exist_ok=True)
            pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred}).to_csv(args.save_pred_csv, index=False, encoding="utf-8")
            print(f"Saved predictions to: {args.save_pred_csv}")
        except Exception as e:
            print(f"Failed to save the prediction CSV file：{e}")

    print("Done.")


if __name__ == "__main__":
    main()