# Examples: run tests on provided sample features

- Uses the **sample feature files already included** in this repo:
  - `sequence_features/` (ESM2 `.pt` features)
  - `structure_features/` (SaProt `.npy` features)
- Uses the **pretrained checkpoints included** in Google drive:
  - `checkpoint/sequence/best.pth`
  - `checkpoint/structure/best.pth`
  - `checkpoint/dual/best.pth`

## 1) Install dependencies

Recommended Python: 3.10

```bash
pip install -r requirements.txt
```

---

## 2) One-command runs

### A) Sequence-only evaluation (`test_sequence.py`)

```bash
python test_sequence.py \
  --pos_dir ./sequence_features/pos_esm2_fea \
  --neg_dir ./sequence_features/neg_esm2_fea \
  --load_model ./checkpoint/sequence/best.pth \
  --out ./results/test_sequence
```

Outputs:
- `results/test_sequence/test_results.txt`

### B) Structure-only evaluation (`test_structure.py`)

```bash
python test_structure.py \
  --pos_dir ./structure_features/pos_test \
  --neg_dir ./structure_features/neg_test \
  --ckpt ./checkpoint/structure/best.pth \
  --out ./results/test_structure
```

Outputs:
- `results/test_structure/test_report.txt`


### C) Dual (sequence + structure) evaluation (`test_dual.py`)

```bash
python test_dual.py \
  --pos3d ./structure_features/pos_test \
  --neg3d ./structure_features/neg_test \
  --all1d ./sequence_features/all_feature \
  --ckpt ./checkpoint/dual/best.pth \
  --outdir ./results/test_dual
```

Outputs (under `results/test_dual/`):
- `classification_report.txt`
- `confusion.png`
- `predictions.csv`
