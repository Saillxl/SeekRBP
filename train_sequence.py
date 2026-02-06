import os
import argparse
import random
import numpy as np
import torch
import torch.utils.data as Data
from torch import nn, optim
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score
from model import *  
import time
import psutil
import shutil

def set_seed(seed=30):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PtFeatureDataset(Data.Dataset):
    def __init__(self, pos_txt: str, neg_txt: str, pos_dir: str, neg_dir: str):
        self.items = []           
        self.pos_indices = []     
        self.neg_indices = []

        with open(pos_txt, 'r') as f:
            pos_samples = [line.strip() for line in f.readlines()]
        with open(neg_txt, 'r') as f:
            neg_samples = [line.strip() for line in f.readlines()]

        for sample in pos_samples:
            pos_path = os.path.join(pos_dir, f"{sample}.pt")
            if os.path.exists(pos_path):
                idx = len(self.items)
                self.items.append((pos_path, 1, sample))
                self.pos_indices.append(idx)

        for sample in neg_samples:
            neg_path = os.path.join(neg_dir, f"{sample}.pt")
            if os.path.exists(neg_path):
                idx = len(self.items)
                self.items.append((neg_path, 0, sample))
                self.neg_indices.append(idx)


        probe = torch.load(self.items[0][0], map_location="cpu")

        self.seq_len = probe.shape[0]
        self.embed_dim = probe.shape[1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y, _ = self.items[idx]
        x = torch.load(path, map_location="cpu").float()
        return x, torch.tensor(y, dtype=torch.long), idx

def make_loader(ds, indices, batch_size=256, num_workers=0, shuffle=True, drop_last=True):
    subset = Data.Subset(ds, indices)
    return Data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                           num_workers=num_workers, drop_last=drop_last)

def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x)
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

class LossTracker:
    def __init__(self, dataset):
        self.value = {}
        self.ni = {}
        self.seen = set()
        self.dataset = dataset   

    def update(self, idxs, losses, labels, init_neg):
        for i, l, y in zip(idxs, losses, labels):
            if int(y) == 0:  
                if i not in self.ni:
                    self.ni[i] = 0
                self.ni[i] += 1
                if self.ni[i] == 1:
                    self.value[i] = float(l)
                else:
                    ln_total_samples = np.log(init_neg)
                    self.value[i] += np.sqrt(2 * ln_total_samples / self.ni[i])
                self.seen.add(i)

    def topk_seen(self, k):
        if not self.seen:
            return []
        scored = [(i, self.value[i]) for i in self.seen]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [i for i, _ in scored[:k]]

    def reset(self):
        self.value.clear()
        self.ni.clear()
        self.seen.clear()

    def save_hard_negatives(self, k, filepath='hard.txt'):

        topk = self.topk_seen(k)
        with open(filepath, 'w') as f:
            for idx in topk:
                _, _, sample_name = self.dataset.items[idx]
                f.write(f"{sample_name}\n")
        print(f"Saved {len(topk)} hard negative sample NAMES to {filepath}")
        

def main():
    parser = argparse.ArgumentParser(description="Train Transformer classifier (binary/multi) on ESM features with dynamic negative mining.")
    parser.add_argument('--pos_txt',
                        default='./pos_trainval_sets/train_set_fold_1.txt',
                        help='positive samples txt')
    parser.add_argument('--neg_txt',
                        default='./neg_trainval_sets/train_set_fold_1.txt',
                        help='negative samples txt')
    parser.add_argument('--pos_dir', default='./pos_esm2_fea',
                        help='positive .pt features directory')
    parser.add_argument('--neg_dir', default='./neg_esm2_fea',
                        help='negative .pt features directory')
    parser.add_argument('--val_pos_txt', default='./pos_trainval_sets/val_set_fold_1.txt',
                        help='positive samples validation txt')
    parser.add_argument('--val_neg_txt', default='./neg_trainval_set/val_set_fold_1.txt',
                        help='negative samples validation txt')
    parser.add_argument('--task', default='binary', choices=['binary', 'multi'])
    parser.add_argument('--nepoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--save_name', default='transformer_binary.pth')
    parser.add_argument('--device', default=None, help='cpu / cuda / cuda:0 ...')
    parser.add_argument('--out', default='./results/train_sequence/', help='directory to save outputs')

    parser.add_argument('--init_neg', type=int, default=20000, help='initial random negative samples count')
    parser.add_argument('--topk_neg', type=int, default=10000, help='number of highest loss negative samples to keep each resampling')
    parser.add_argument('--rand_new_neg', type=int, default=10000, help='number of new random negative samples to draw each resampling')
    parser.add_argument('--resample_every', type=int, default=2, help='every how many epochs to resample negatives')

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = PtFeatureDataset(args.pos_txt, args.neg_txt, args.pos_dir, args.neg_dir)
    val_ds = PtFeatureDataset(args.val_pos_txt, args.val_neg_txt, args.pos_dir, args.neg_dir)
    val_loader = make_loader(val_ds, range(len(val_ds)), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=True)

    num_classes = 2 if args.task == 'binary' else len(set([ds.items[i][1] for i in range(len(ds))]))
    print(f"seq_len={ds.seq_len}, embed_dim={ds.embed_dim}, num_classes={num_classes}")
    print(f"Train size={len(ds.items)}, Validation size={len(val_ds.items)}")

    rng = random.Random(args.seed)
    init_k = min(args.init_neg, len(ds.neg_indices))
    active_negs = rng.sample(ds.neg_indices, init_k) if init_k > 0 else []
    train_active_indices = ds.pos_indices + active_negs
    print(f"Initial active negatives: {len(active_negs)}")

    model = Transformer(
        src_vocab_size=ds.embed_dim,
        src_pad_idx=0,
        device=device,
        max_length=ds.seq_len,
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

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp))

    best_metric = 0.0
    save_path = os.path.join(args.out, args.save_name)

    train_loader = make_loader(ds, train_active_indices, batch_size=args.batch_size,
                               num_workers=args.num_workers, shuffle=True, drop_last=True)

    tracker = LossTracker(dataset=ds)
    epoch_in_window = 0

    start = time.time()
    print("Training started...")
    print_resource_usage()
    for epoch in range(1, args.nepoch + 1):
        model.train()
        running_loss = 0.0

        for xb, yb, idxb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and args.amp)):
                logits = model(xb)
                losses = criterion(logits, yb)
                loss = losses.mean()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += float(loss.item())

            tracker.update(
                idxs=idxb.numpy().tolist(),
                losses=losses.detach().cpu().numpy().tolist(),
                labels=yb.detach().cpu().numpy().tolist(),
                init_neg=args.init_neg
            )

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_acc, val_f1, val_prec, y_true, y_pred = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_prec={val_prec:.4f}")

        if epoch % 10 == 0:
            path = os.path.join(args.out, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), path)

        if val_f1 > best_metric:
            best_metric = val_f1
            torch.save(model.state_dict(), save_path)
            print(f" -> Saved best to {save_path} (F1={best_metric:.4f})")
            print("Validation report:")
            print(classification_report(y_true, y_pred, digits=4))

        epoch_in_window += 1
        if epoch_in_window >= args.resample_every:
            topk = tracker.topk_seen(args.topk_neg)

            seen_negs = tracker.seen
            unseen_negs = list(set(ds.neg_indices) - set(seen_negs))
            rng.shuffle(unseen_negs)
            rand_pick = unseen_negs[:args.rand_new_neg]

            new_active_negs = list(dict.fromkeys(topk + rand_pick))
            if len(new_active_negs) == 0:
                fallback_k = min(args.topk_neg + args.rand_new_neg, len(ds.neg_indices))
                new_active_negs = rng.sample(ds.neg_indices, fallback_k)

            train_active_indices = ds.pos_indices + new_active_negs
            train_loader = make_loader(ds, train_active_indices, batch_size=args.batch_size,
                                       num_workers=args.num_workers, shuffle=True, drop_last=True)

            print(f"[Resample @ epoch {epoch}] seen_neg={len(seen_negs)}, "
                  f"topk={len(topk)}, unseen_pool={len(unseen_negs)}, "
                  f"new_active_negs={len(new_active_negs)}, train_size={len(train_active_indices)}")

            # Save the ID of the most difficult negative samples to a file
            tracker.save_hard_negatives(k=10000, filepath=os.path.join(args.out, 'hard.txt'))

            tracker.reset()
            epoch_in_window = 0
    end = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = end - start

    print("Done.")

if __name__ == "__main__":
    main()
