import os
import argparse
import random
from glob import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model_dual import Trainable1DEncoder, Trainable3DEncoder, DualBranchClassifier


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _map_files(dir_, ext):
    # Only match files with the extension *.ext in the directory (consistent with the reference script you provided)
    return {Path(p).stem: p for p in glob(str(Path(dir_) / f"*.{ext}"))}


def build_pairs_by_3d_first(pos3d_dir, neg3d_dir, all1d_dir):
    """
    Match the longest prefix of the 3D file name with the 1D key.
    return items: (pt_path, npy_path, label, hit_key)
    """
    map_1d = _map_files(all1d_dir, "pt")
    if not map_1d:
        raise RuntimeError(f"Empty 1D dir: {all1d_dir}")
    keys_1d_by_len = sorted(map_1d.keys(), key=len, reverse=True)

    def collect(dir3, label):
        map_3d = _map_files(dir3, "npy")
        if not map_3d:
            print(f"[WARN] Empty 3D dir: {dir3}")
            return []
        matched, missed = [], []
        for stem3, p3 in map_3d.items():
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
    """
    items: (pt_path, npy_path, label, hit_key)
    Provide pos_indices / neg_indices for dynamic sampling
    """
    def __init__(self, items):
        self.items = items
        self.pos_indices = [i for i, it in enumerate(items) if int(it[2]) == 1]
        self.neg_indices = [i for i, it in enumerate(items) if int(it[2]) == 0]

        p1, p3, _, _ = items[0]
        a = torch.load(p1, map_location="cpu")
        b = np.load(p3, mmap_mode="r")
        if b.ndim == 3 and b.shape[0] == 1:
            b = b[0]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Expect 1D [L,D] and 3D [L,D], got {a.shape} and {b.shape}")

        self.L1, self.D1 = int(a.shape[0]), int(a.shape[1])
        self.L3, self.D3 = int(b.shape[0]), int(b.shape[1])

        print(f"[DATASET] total={len(self.items)} pos={len(self.pos_indices)} neg={len(self.neg_indices)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p1, p3, y, hit = self.items[idx]
        x1 = torch.load(p1, map_location="cpu").float()

        b = np.load(p3, mmap_mode="r")
        if b.ndim == 3 and b.shape[0] == 1:
            b = b[0]
        x3 = torch.tensor(np.asarray(b, dtype=np.float32, order="C", copy=True), dtype=torch.float32)

        # Return idx to LossTracker for dynamic sampling
        meta = (p1, p3, hit)
        return x1, x3, torch.tensor(int(y), dtype=torch.long), torch.tensor(idx, dtype=torch.long), meta


def make_loader(dataset, indices, bs, shuffle, num_workers=0):
    subset = Data.Subset(dataset, indices)
    return Data.DataLoader(
        subset,
        batch_size=bs,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )


class LossTracker:
    def __init__(self, dataset):
        self.value = {}  # idx -> Q_i(l) estimated value
        self.ni = {}     # idx -> Number of selections n_i(l)
        self.seen = set()  # Set of samples that have been seen
        self.dataset = dataset

    def update(self, idxs, losses, labels):
        for i, l, y in zip(idxs, losses, labels):
            if int(y) == 0:  # Only count negative samples (label == 0)
                # Initialize Q_i(l)
                if i not in self.value:
                    self.value[i] = 0.0
                if i not in self.ni:
                    self.ni[i] = 0
                self.value[i] += float(l)
                self.ni[i] += 1
                self.seen.add(i)

    def avg(self, idx):
        return self.value[idx] / max(self.ni[idx], 1)

    def ucb_score(self, idx, total_selections):
        """Calculate UCB score V_i(l)"""
        Q = self.avg(idx)
        exploration_bonus = np.sqrt(2 * np.log(total_selections) / self.ni[idx])  # Exploration factor
        return Q + exploration_bonus

    def topk_seen(self, k):
        if not self.seen:
            return []
        total_selections = sum(self.ni.values())  # Total number of selections for all samples
        scored = [(i, self.ucb_score(i, total_selections)) for i in self.seen]
        scored.sort(key=lambda t: t[1], reverse=True)  # Sort by UCB score
        return [i for i, _ in scored[:k]]

    def reset(self):
        self.value.clear()
        self.ni.clear()
        self.seen.clear()



@torch.no_grad()
def evaluate_on_full_train(model, loader, device):
    """
    Measure on the full training set (full_loader)
    """
    model.eval()
    ce = nn.CrossEntropyLoss()

    ys, ps = [], []
    total_loss = 0.0

    for x1, x3, y, _, _ in loader:
        x1, x3, y = x1.to(device), x3.to(device), y.to(device)
        if x1.ndim == 2: x1 = x1.unsqueeze(0)
        if x3.ndim == 2: x3 = x3.unsqueeze(0)

        logits = model(x1, x3)
        total_loss += ce(logits, y).item()

        ps.extend(logits.argmax(1).cpu().tolist())
        ys.extend(y.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(ys, ps) if ys else 0.0
    f1 = f1_score(ys, ps, average="macro") if ys else 0.0
    prec = precision_score(ys, ps, average="macro", zero_division=0) if ys else 0.0
    return {"loss": avg_loss, "acc": acc, "f1": f1, "prec": prec, "y_true": ys, "y_pred": ps}


def train_epoch_with_mining(model, loader, opt, device, tracker: LossTracker, init_neg: int):
    model.train()
    ce_none = nn.CrossEntropyLoss(reduction="none")

    total_loss = 0.0
    all_y, all_p = [], []

    for x1, x3, y, idx, _ in loader:
        x1, x3, y = x1.to(device), x3.to(device), y.to(device)
        if x1.ndim == 2: x1 = x1.unsqueeze(0)
        if x3.ndim == 2: x3 = x3.unsqueeze(0)

        logits = model(x1, x3)
        losses = ce_none(logits, y)  
        loss = losses.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += float(loss.item())
        all_p.extend(logits.argmax(1).detach().cpu().tolist())
        all_y.extend(y.detach().cpu().tolist())

        tracker.update(
            idxs=idx.detach().cpu().numpy().tolist(),
            losses=losses.detach().cpu().numpy().tolist(),
            labels=y.detach().cpu().numpy().tolist()
        )

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(all_y, all_p) if all_y else 0.0
    f1 = f1_score(all_y, all_p, average="macro") if all_y else 0.0
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()

    # Data path
    parser.add_argument("--pos3d", type=str, default="./structure_features/pos_train")
    parser.add_argument("--neg3d", type=str, default="./structure_features/neg_train")
    parser.add_argument("--all1d", type=str, default="./sequence_features/all_feature")
    parser.add_argument("--outdir", type=str, default="./results/train_dual")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)

    # Model hyperparameters
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn_exp", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fusion_hidden", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--enc_layers_1d", type=int, default=1)
    parser.add_argument("--enc_layers_3d", type=int, default=1)

    # ===== Dynamic negative sampling (UCB) parameters =====
    parser.add_argument("--init_neg", type=int, default=2000, help="initial random negative samples count")
    parser.add_argument("--topk_neg", type=int, default=1000, help="top-k hardest negatives to keep each resample")
    parser.add_argument("--rand_new_neg", type=int, default=1000, help="new random negatives from unseen pool")
    parser.add_argument("--resample_every", type=int, default=2, help="every how many epochs to resample negatives")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Full items (using longest prefix matching)
    items = build_pairs_by_3d_first(args.pos3d, args.neg3d, args.all1d)
    tr_ds = DualDataset(items)

    # full_loader for measurement (full training set)
    full_indices = list(range(len(tr_ds)))
    full_loader = Data.DataLoader(
        tr_ds, batch_size=args.bs, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=args.num_workers
    )

    print(f"[DATA] train={len(tr_ds)} | 1D: L={tr_ds.L1},D={tr_ds.D1} | 3D: L={tr_ds.L3},D={tr_ds.D3}")

    enc1d = Trainable1DEncoder(
        d_in_1d=tr_ds.D1,
        embed_size=args.embed,
        heads=args.heads,
        forward_expansion=args.ffn_exp,
        dropout=args.dropout,
        max_length=tr_ds.L1,
        device=device,
        out_dim=args.out_dim,
        num_layers=args.enc_layers_1d,
    ).to(device)

    enc3d = Trainable3DEncoder(
        d_in_3d=tr_ds.D3,
        embed_size=args.embed,
        heads=args.heads,
        forward_expansion=args.ffn_exp,
        dropout=args.dropout,
        max_length=tr_ds.L3,
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

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    # ===== Dynamic negative sampling: initialize active negative samples =====
    rng = random.Random(args.seed)
    init_k = min(args.init_neg, len(tr_ds.neg_indices))
    active_negs = rng.sample(tr_ds.neg_indices, init_k) if init_k > 0 else []
    train_active_indices = tr_ds.pos_indices + active_negs
    print(f"[MINING] init_neg={len(active_negs)} total_neg={len(tr_ds.neg_indices)} train_size={len(train_active_indices)}")

    train_loader = make_loader(tr_ds, train_active_indices, args.bs, shuffle=True, num_workers=args.num_workers)

    tracker = LossTracker(tr_ds)
    epoch_in_window = 0

    best_train_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        # Training (using active negatives)
        train_loss, train_acc, train_f1 = train_epoch_with_mining(
            model, train_loader, opt, device, tracker, init_neg=args.init_neg
        )

        # Measurement (full training set)
        train_res = evaluate_on_full_train(model, full_loader, device)
        scheduler.step(train_res["loss"])

        print(f"[E{epoch}] tr_loss={train_loss:.4f} tr_acc={train_acc:.4f} tr_f1={train_f1:.4f} | "
              f"full_loss={train_res['loss']:.4f} full_acc={train_res['acc']:.4f} full_f1={train_res['f1']:.4f}")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.outdir, f"ckpt_epoch{epoch}.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()}, ckpt_path)
            print(f"[SAVE] {ckpt_path}")

        if train_res["f1"] > best_train_f1:
            best_train_f1 = train_res["f1"]
            best_path = os.path.join(args.outdir, "best_model.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "train_f1": train_res["f1"]}, best_path)
            print(f"[BEST] saved {best_path} (full_train_f1={train_res['f1']:.4f})")

        # Dynamic negative sampling resampling
        epoch_in_window += 1
        if epoch_in_window >= args.resample_every:
            # Select top-k negative samples by UCB score (only in "seen" negative samples)
            topk = tracker.topk_seen(args.topk_neg)
        
            # Calculate the set of negative samples that have not been trained
            seen_negs = tracker.seen
            unseen_negs = list(set(tr_ds.neg_indices) - set(seen_negs))
            rng.shuffle(unseen_negs)
            rand_pick = unseen_negs[:args.rand_new_neg]
        
            # Build new training indices (all positive samples + new negative samples)
            new_active_negs = list(dict.fromkeys(topk + rand_pick))  # Remove duplicates and maintain order
            if len(new_active_negs) == 0:
                fallback_k = min(args.topk_neg + args.rand_new_neg, len(tr_ds.neg_indices))
                new_active_negs = rng.sample(tr_ds.neg_indices, fallback_k)
        
            train_active_indices = tr_ds.pos_indices + new_active_negs
            train_loader = make_loader(tr_ds, train_active_indices, args.bs, shuffle=True, num_workers=args.num_workers)
        
            print(f"[RESAMPLE @E{epoch}] seen_neg={len(seen_negs)} topk={len(topk)} "
                  f"unseen_pool={len(unseen_negs)} new_active_negs={len(new_active_negs)} "
                  f"train_size={len(train_active_indices)}")
        
            tracker.reset()
            epoch_in_window = 0

    # ===== Training ended: use best_model to output report and confusion matrix on the full training set =====
    best = torch.load(os.path.join(args.outdir, "best_model.pth"), map_location=device)
    model.load_state_dict(best["model"])
    final_res = evaluate_on_full_train(model, full_loader, device)

    report_txt = classification_report(final_res["y_true"], final_res["y_pred"], digits=4)
    with open(os.path.join(args.outdir, "train_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    cm = confusion_matrix(final_res["y_true"], final_res["y_pred"], labels=[0, 1])
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (train)")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, "train_confusion.png"), dpi=200)
    plt.close(fig)

    print("[DONE] Best full_train_f1:", best.get("train_f1", best_train_f1))
    print(report_txt)
    print(f"[SAVED] best_model/report/confusion/hard.txt in {args.outdir}")


if __name__ == "__main__":
    main()