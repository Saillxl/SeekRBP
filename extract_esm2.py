import os
import gc
import torch
import esm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()

def clean_sequence(seq: str) -> str:
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join([c for c in seq if c in valid_chars])

def read_fasta(path: str):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    sequences = []
    name, seq = None, []
    for line in lines:
        if line.startswith(">"):
            if name:
                sequences.append((name, clean_sequence("".join(seq))))
            name = line[1:].split()[0]
            seq = []
        else:
            seq.append(line)
    if name:
        sequences.append((name, clean_sequence("".join(seq))))
    return sequences

fasta_path = "./all.faa"
save_dir = "./esm2_fea"
os.makedirs(save_dir, exist_ok=True)
sequences = read_fasta(fasta_path)

MAX_SEQ_LEN = 5000
TARGET_LEN = 1024  # fixed output length
processed_count = 0

for seq_id, seq in sequences:
    if len(seq) == 0:
        print(f"skip empty: {seq_id}")
        continue
    if len(seq) > MAX_SEQ_LEN:
        print(f"skip {seq_id}, length {len(seq)} exceeds limit")
        continue

    safe_seq_id = seq_id.translate(str.maketrans("", "", "|/\\:*?\"<>"))
    data = [(seq_id, seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    try:
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            reps = results["representations"][33]  # [1, L+2, d]
            per_residue = reps[0, 1:len(seq)+1, :]  # remove BOS/EOS
            emb_dim = per_residue.size(1)

            # Pad or truncate to TARGET_LEN
            if per_residue.size(0) < TARGET_LEN:
                pad_size = TARGET_LEN - per_residue.size(0)
                padding = torch.zeros(pad_size, emb_dim, device=per_residue.device)
                fixed_emb = torch.cat([per_residue, padding], dim=0)
            else:
                fixed_emb = per_residue[:TARGET_LEN, :]

        torch.save(fixed_emb.cpu(), os.path.join(save_dir, f"{safe_seq_id}.pt"))
        print(f"saved: {seq_id}, shape {tuple(fixed_emb.shape)}")
        processed_count += 1

    except torch.cuda.OutOfMemoryError:
        print(f"OOM skip: {seq_id}")

    del batch_tokens, results, reps, per_residue, fixed_emb
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

print(f"done. processed {processed_count} sequences.")
