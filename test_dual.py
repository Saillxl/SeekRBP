import os
import argparse
import csv
from glob import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model_dual import Trainable1DEncoder, Trainable3DEncoder, DualBranchClassifier


def _map_files(dir_, ext):
    return {Path(p).stem: p for p in glob(str(Path(dir_) / f"*.{ext}"))}


def build_pairs_by_3d_first(pos3d_dir, neg3d_dir, all1d_dir):
    """
    Match 1D .pt files by the longest prefix of the 3D file name
    return (pt_path, npy_path, label, key)
    """
    map_1d = _map_files(all1d_dir, "pt")
    if not map_1d:
        raise RuntimeError(f"Empty 1D dir: {all1d_dir}")
    keys_1d_by_len = sorted(map_1d.keys(), key=len, reverse=True)

    def collect(dir3, label):
        mp3 = _map_files(dir3, "npy")
        if not mp3:
            print(f"[WARN] Empty 3D dir: {dir3}")
            return []
        matched, missed = [], []
        for stem3, p3 in mp3.items():
            hit = next((k for k in keys_1d_by_len if stem3.startswith(k)), None)
            if hit is None:
                missed.append(stem3)
            else:
                matched.append((map_1d[hit], p3, int(label), hit))
        if missed:
            print(f"[WARN] In {dir3}, {len(missed)} 3D samples not matched. Eg {missed[:5]}")
        return matched

    pos = collect(pos3d_dir, 1)
    neg = collect(neg3d_dir, 0)
    items = pos + neg
    print(f"[MATCH] pos={len(pos)} neg={len(neg)} total={len(items)}")
    if not items:
        raise RuntimeError("No matched items")
    return items


class DualDataset(Data.Dataset):
    def __init__(self, items):
        self.items = items
        # Detect dimensions
        p1, p3, _, _ = items[0]
        a = torch.load(p1, map_location="cpu")
        b = np.load(p3, mmap_mode="r")
        if b.ndim == 3 and b.shape[0] == 1:
            b = b[0]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Expect 1D [L,D] and 3D [L,D], got {a.shape} and {b.shape}")
        self.L1, self.D1 = int(a.shape[0]), int(a.shape[1])
        self.L3, self.D3 = int(b.shape[0]), int(b.shape[1])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p1, p3, y, key = self.items[idx]  
        x1 = torch.load(p1, map_location="cpu").float()  # [L1, D1]
        b = np.load(p3, mmap_mode="r")
        if b.ndim == 3 and b.shape[0] == 1:
            b = b[0]
        x3 = torch.tensor(np.asarray(b, dtype=np.float32, order="C", copy=True), dtype=torch.float32)
        return x1, x3, torch.tensor(y, dtype=torch.long), (p1, p3, key)


@torch.no_grad()
def evaluate(model, loader, device, outdir, save_probs=False):
    model.eval()
    ce = nn.CrossEntropyLoss()
    ys, ps, total_loss = [], [], 0.0
    metas = []
    prob_list = []  # optional

    for x1, x3, y, meta in loader:
        x1, x3, y = x1.to(device), x3.to(device), y.to(device)
        if x1.ndim == 2: x1 = x1.unsqueeze(0)
        if x3.ndim == 2: x3 = x3.unsqueeze(0)

        logits = model(x1, x3)
        loss = ce(logits, y)
        total_loss += loss.item()

        preds = logits.argmax(1).cpu().tolist()
        ps.extend(preds)
        ys.extend(y.cpu().tolist())


        if isinstance(meta, (list, tuple)):
            metas.extend(list(meta))
        else:
            metas.append(meta)

        if save_probs:
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            prob_list.extend(probs.tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(ys, ps) if ys else 0.0
    f1 = f1_score(ys, ps, average="macro") if ys else 0.0
    prec = precision_score(ys, ps, average="macro", zero_division=0) if ys else 0.0
    os.makedirs(outdir, exist_ok=True)
    report = classification_report(ys, ps, digits=4)
    with open(os.path.join(outdir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(ys, ps, labels=[0, 1])
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (test)")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "confusion.png"), dpi=200)
    plt.close(fig)

    csv_path = os.path.join(outdir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["pt_path", "npy_path", "key", "y_true", "y_pred"]
        if save_probs:
            header += ["prob_class0", "prob_class1"]
        w.writerow(header)

        for idx, (m, yt, yp) in enumerate(zip(metas, ys, ps)):
            if isinstance(m, (list, tuple)):
                p1 = m[0] if len(m) > 0 else ""
                p3 = m[1] if len(m) > 1 else ""
                key = m[2] if len(m) > 2 else ""
            elif isinstance(m, dict):
                p1, p3, key = m.get("p1", ""), m.get("p3", ""), m.get("key", "")
            else:
                p1, p3, key = str(m), "", ""

            row = [p1, p3, key, yt, yp]
            if save_probs:
                prob = prob_list[idx] if idx < len(prob_list) else ["", ""]
                if isinstance(prob, (list, tuple)) and len(prob) >= 2:
                    row += [prob[0], prob[1]]
                else:
                    row += ["", ""]
            w.writerow(row)

    print(f"[TEST] loss={avg_loss:.4f} acc={acc:.4f} f1={f1:.4f} prec={prec:.4f}")
    print(report)
    print(f"[SAVED] report/confusion/predictions to {outdir}")

    return {"loss": avg_loss, "acc": acc, "f1": f1, "prec": prec}


def main():
    ap = argparse.ArgumentParser()
    # Data path
    ap.add_argument("--pos3d", type=str, default="./structure_features/pos_test")
    ap.add_argument("--neg3d", type=str, default="./structure_features/neg_test")
    ap.add_argument("--all1d", type=str, default="./sequence_features/all_feature")
    ap.add_argument("--outdir", type=str, default="./results/test_dual")

    # Model structure (must be consistent with training)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn_exp", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out_dim", type=int, default=256)
    ap.add_argument("--fusion_hidden", type=int, default=256)
    ap.add_argument("--enc_layers_1d", type=int, default=1)
    ap.add_argument("--enc_layers_3d", type=int, default=1)

    # Running parameters
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ckpt", type=str, required=True, help="Trained best_model.pth or other checkpoint")
    ap.add_argument("--save_probs", action="store_true", help="Append probability for each class (only binary classification) to CSV")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build data
    items = build_pairs_by_3d_first(args.pos3d, args.neg3d, args.all1d)
    ds = DualDataset(items)
    loader = Data.DataLoader(ds, batch_size=args.bs, shuffle=False, drop_last=False, pin_memory=True)
    print(f"[DATA] test={len(ds)} | 1D: L={ds.L1},D={ds.D1} | 3D: L={ds.L3},D={ds.D3}")

    # Build model (structure parameters must be consistent with training)
    enc1d = Trainable1DEncoder(
        d_in_1d=ds.D1,
        embed_size=args.embed,
        heads=args.heads,
        forward_expansion=args.ffn_exp,
        dropout=args.dropout,
        max_length=ds.L1,
        device=device,
        out_dim=args.out_dim,
        num_layers=args.enc_layers_1d,
    ).to(device)

    enc3d = Trainable3DEncoder(
        d_in_3d=ds.D3,
        embed_size=args.embed,
        heads=args.heads,
        forward_expansion=args.ffn_exp,
        dropout=args.dropout,
        max_length=ds.L3,
        device=device,
        out_dim=args.out_dim,
        num_layers=args.enc_layers_3d,
    ).to(device)

    model = DualBranchClassifier(
        enc1d=enc1d,
        enc3d=enc3d,
        num_classes=2,
        fusion_hidden=args.fusion_hidden,
        dropout=0.2,
    ).to(device)

    # Load weights
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[CKPT] Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[CKPT] Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    print(f"[LOAD] checkpoint loaded from {args.ckpt}")

    # Evaluate
    _ = evaluate(model, loader, device, args.outdir, save_probs=args.save_probs)


if __name__ == "__main__":
    main()