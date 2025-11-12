import os, math, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import TinyGPT
from tqdm import tqdm
import random
import string

# ===============initialisations==============
DATA_PATH = 'train.txt'
SEQ_LEN = 128
BATCH = 4
GRAD_ACCUM = 8
STEPS = 2000
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = 'opfiles'
FP16 = True
SEED = 67
PAD = 0
BOS = 1
EOS = 2
BYTE_OFFSET = 3
VOCAB_SIZE = 256 + BYTE_OFFSET
# ============================================

torch.manual_seed(SEED)
random.seed(SEED)

def encode_bytes(s):
    b = s.encode("utf-8", errors="replace")
    return [BYTE_OFFSET + byte for byte in b]

def decode_bytes(ids):
    bs = bytearray()
    for i in ids:
        if i >= BYTE_OFFSET:
            bs.append(i - BYTE_OFFSET)
    return bs.decode("utf-8", errors="replace")

class ChunkDataset(Dataset):
    def __init__(self, path, seq_len):
        txt = open(path, "r", encoding="utf-8", errors="replace").read()
        tokens = []
        for line in txt.splitlines():
            if not line.strip():
                continue
            enc = encode_bytes(line + "\n")
            tokens.extend(enc)
            
        self.seq_len = seq_len
        
        if len(tokens) < seq_len+1:
            tokens = tokens + [PAD] * (seq_len+1 - len(tokens))
        
        self.chunks = []
        i=0
        
        while i + seq_len + 1 <= len(tokens):
            chunk = tokens[i:i+seq_len+1]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))
            i += seq_len + 1
            
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        seq = self.chunks[idx]
        inp = seq[:-1]
        tgt = seq[1:]
        return torch.as_tensor(inp, dtype=torch.long), torch.as_tensor(tgt, dtype=torch.long)
    
ds = ChunkDataset(DATA_PATH, SEQ_LEN)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)

model = TinyGPT(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, n_layer=4, d_model=128, n_head=4, d_ff=512).to(DEVICE)
optim = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

scaler = torch.amp.GradScaler(enabled=FP16)

step = 0
os.makedirs(SAVE_DIR, exist_ok=True)
pbar = tqdm(total=STEPS)
model.train()
while step < STEPS:
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=FP16):
            logits = model(xb)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), yb.view(-1))
            loss = loss / GRAD_ACCUM
            
        scaler.scale(loss).backward()
        
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
        
        if step % 10 == 0:
            tqdm.write(f"Step {step}, Loss: {loss.item() * GRAD_ACCUM:.4f}")
            
        if step % 200 == 0 and step > 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_step{step}.pt"))
            
        step += 1
        pbar.update(1)
        
        if step >= STEPS:
            break
        
torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_final.pt"))
pbar.close()