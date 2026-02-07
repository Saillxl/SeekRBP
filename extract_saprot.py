"""
Generate token-level fixed-length [1, 512, H] representation for ColabFold (AlphaFold2) prediction structures in batches
- Only process PDB files for rank_001: *_unrelaxed_rank_001_alphafold2_ptm_model_.pdb
- Use SaProt + ESM tokenizer
- Enable pLDDT mask (for AF2 structures), use combined_seq
- Pad/truncate the model output to a fixed length of 512 manually to obtain [1, 512, H]
- Save each sample as .npy (with the same name)

"""
import argparse
import os
from glob import glob
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.foldseek_util import get_struc_seq
from model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer

# Initialization
def build_model_and_tokenizer(config_path: str, load_pretrained: bool = True):
    cfg = {"task": "base", "config_path": config_path, "load_pretrained": load_pretrained}
    model = SaprotBaseModel(**cfg)
    tokenizer = EsmTokenizer.from_pretrained(config_path)
    return model, tokenizer


def choose_device(device_arg: str = "auto") -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg in ("cuda", "cpu", "mps"):
        if device_arg == "cuda" and not torch.cuda.is_available():
            print("[Warn] CUDA is not available, fallback to CPU")
            return "cpu"
        return device_arg
    print("[Warn] Unknown device parameter, fallback to auto")
    return "cuda" if torch.cuda.is_available() else "cpu"


# Tokenize & forward
def tokenize_with_truncation(tokenizer, seq: str, max_length: int, device: str):
    """
    On the tokenizer side, perform max_length truncation and padding. Note: The model will remove the padding.
    """
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def get_hidden_states_no_reduction(model, inputs) -> torch.Tensor:
    """
    Return the residue-level representation of [B, L, H].
    Compatible with Tensor or List/Tuple[Tensor] (taking the last layer).
    """
    def _pick_last_layer(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)) and len(x) > 0:
            for t in reversed(x):
                if isinstance(t, torch.Tensor):
                    return t
        raise TypeError(f"Unsupported hidden_states type: {type(x)}")

    with torch.no_grad():
        # 1) reduction=None
        try:
            out = model.get_hidden_states(inputs, reduction=None)
            last = _pick_last_layer(out)
            return last if last.dim() == 3 else last.unsqueeze(0)  #  [B,L,H]
        except TypeError:
            pass

        # 2) reduction="none"
        try:
            out = model.get_hidden_states(inputs, reduction="none")
            last = _pick_last_layer(out)
            return last if last.dim() == 3 else last.unsqueeze(0)
        except TypeError:
            pass

        # 3) no reduction
        out = model.get_hidden_states(inputs)
        last = _pick_last_layer(out)
        return last if last.dim() == 3 else last.unsqueeze(0)


# Sequence extraction (AF2)
def extract_seq_from_pdb(foldseek_bin: str, pdb_path: str, prefer_chain: str = "A") -> Tuple[str, str, str]:
    """
    Extract (seq, foldseek_seq, combined_seq) using get_struc_seq.
    - Preferentially select chain A; if there is no A, automatically select the first available chain.
    - Use plddt_mask=True for the AF2 structure.
    """
    parsed: Dict[str, Tuple[str, str, str]] = get_struc_seq(
        foldseek_bin, pdb_path, [prefer_chain], plddt_mask=True
    )
    if prefer_chain in parsed:
        return parsed[prefer_chain]

    # If there is no A chain, then take all the chains and select the first one.
    for cid in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        try:
            parsed2 = get_struc_seq(foldseek_bin, pdb_path, [cid], plddt_mask=True)
            if cid in parsed2:
                return parsed2[cid]
        except Exception:
            continue
    raise RuntimeError(f"No sequences of any chains were extracted from {pdb_path}")


# Main process (batch)
def main():
    parser = argparse.ArgumentParser(description="Generate fixed-length token representations [1, 512, H] for the ColabFold rank_001 structure in batches.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory (including the ColabFold PDB file named rank_001)")
    parser.add_argument("--output_dir", type=str, default="./feature_neg_test",
                        help="Output directory")
    parser.add_argument("--config_path", type=str, default="./weights", help="SaProt weights and tokenizer directory")
    parser.add_argument("--foldseek_bin", type=str, default="bin/foldseek", help="The executable file path of Foldseek")
    parser.add_argument("--pad_to", type=int, default=512, help="Fixed token length")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu/mps")
    parser.add_argument("--save_float16", action="store_true", help="Store in float16 format (to save space)")
    parser.add_argument("--overwrite", action="store_true", help="If a file with the same name exists as .npy, it will be overwritten.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"[Info] Using device: {device}")

    print("[Info] Initializing model and tokenizer...")
    model, tokenizer = build_model_and_tokenizer(args.config_path, load_pretrained=True)
    model.to(device).eval()

    # Only select PDB files for rank_001
    pattern = str(input_dir / "*_unrelaxed_rank_001_alphafold2_ptm_model_*.pdb")
    pdb_files = sorted(glob(pattern))
    if not pdb_files:
        print(f"[Warn] No PDB files for rank_001 were found in {input_dir}, using pattern: {pattern}")
        return

    print(f"[Info] Number of files to process: {len(pdb_files)}")
    error_log = []

    for pdb_path in tqdm(pdb_files, desc="Processing", ncols=100):
        pdb_path = Path(pdb_path)
        out_path = output_dir / (pdb_path.stem + ".npy")

        if out_path.exists() and not args.overwrite:
            continue

        try:
            # 1) Extract sequence (AF2: plddt_mask=True, using combined_seq)
            seq, foldseek_seq, combined_seq = extract_seq_from_pdb(args.foldseek_bin, str(pdb_path))
            seq_in = combined_seq  

            # 2) Tokenizer truncation (upper limit control), note that the model will remove padding
            inputs = tokenize_with_truncation(tokenizer, seq_in, args.pad_to, device)

            # 3) Forward to get the residue-level representation; normalize to [1,L,H]
            last_hidden = get_hidden_states_no_reduction(model, inputs)
            if not isinstance(last_hidden, torch.Tensor) or last_hidden.dim() != 3:
                raise RuntimeError(f"Expected [B,L,H] Tensor, got: {type(last_hidden)} with shape {getattr(last_hidden, 'shape', None)}")

            B, L, H = last_hidden.shape

            # 4) Construct the attention_mask for the real tokens (length L, to be extended later during padding)
            attn_mask_aligned = torch.ones((B, L), device=last_hidden.device, dtype=torch.long)

            # 5) Truncate to pad_to
            target_len = args.pad_to
            if L > target_len:
                last_hidden = last_hidden[:, :target_len, :]
                attn_mask_aligned = attn_mask_aligned[:, :target_len]
                L = target_len

            # 6) Pad zeros to the right to pad_to → [1, pad_to, H]
            if L < target_len:
                pad_len = target_len - L
                last_hidden = F.pad(last_hidden, (0, 0, 0, pad_len, 0, 0), value=0.0)
                attn_mask_aligned = F.pad(attn_mask_aligned, (0, pad_len, 0, 0), value=0)

            assert last_hidden.shape[1] == target_len
            # Here we directly save the token-level matrix (you can do any pooling/Transformer stacking during training)
            out_tensor = last_hidden  # [1, 512, H]

            # 7) Save to disk
            arr = out_tensor.detach().cpu().numpy()
            if args.save_float16:
                arr = arr.astype(np.float16)
            np.save(out_path, arr)

        except Exception as e:
            error_log.append((str(pdb_path), repr(e)))
            continue

    print(f"[Done] Successfully generated: {len(pdb_files) - len(error_log)} files, saved to: {output_dir}")
    if error_log:
        print("[Errors]")
        for p, msg in error_log[:20]:
            print(f"- {p}: {msg}")
        if len(error_log) > 20:
            print(f"... There are {len(error_log)} errors, truncated display.")


if __name__ == "__main__":
    main()