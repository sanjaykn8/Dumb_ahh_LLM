import torch
from model import TinyGPT
from train import decode_bytes, encode_bytes, VOCAB_SIZE, SEQ_LEN, DEVICE
import argparse
import random

def decode_bytes_local(ids):
    bs = bytearray()
    for i in ids:
        if i >= 3:
            bs.append(i - 3)
    return bs.decode("utf-8", errors="replace")

def encode_bytes_local(s):
    b = s.encode("utf-8", errors="replace")
    return [3 + byte for byte in b]

# replace sample(...) in infer.py with this function
def sample(model, prompt, max_new_tokens=120, top_k=50, temp=1.0):
    model.eval()
    enc = encode_bytes_local(prompt)
    cur_tokens = enc[-SEQ_LEN:]
    # left-pad to model context length
    cur = [0] * (SEQ_LEN - len(cur_tokens)) + cur_tokens
    cur = torch.tensor([cur], dtype=torch.long).to(DEVICE)
    out = enc.copy()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits_all = model(cur)  # (1, T, V)
            # choose the last position's logits
            t_index = len(out) - 1
            if t_index < 0:
                t_index = 0
            if t_index >= logits_all.shape[1]:
                t_index = logits_all.shape[1] - 1
            logits = logits_all[0, t_index, :]  # (V,)

            logits = logits / max(1e-8, float(temp))
            k = int(min(int(top_k), int(logits.size(-1))))
            topk_vals, topk_idx = torch.topk(logits, k)
            probs = torch.softmax(topk_vals, dim=-1)
            chosen_idx_in_topk = torch.multinomial(probs, num_samples=1).item()
            chosen = topk_idx[chosen_idx_in_topk].item()

            out.append(chosen)
            # prepare next context window (right-most SEQ_LEN tokens)
            tail = out[-SEQ_LEN:]
            padded = [0] * (SEQ_LEN - len(tail)) + tail
            cur = torch.tensor([padded], dtype=torch.long).to(DEVICE)

    return decode_bytes_local(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="opfiles/model_final.pt")
    parser.add_argument("--prompt", default="love")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temp", type=float, default=1.0)
    args = parser.parse_args()

    model = TinyGPT(vocab_size=259, seq_len=128, n_layer=4, d_model=128, n_head=4, d_ff=512).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    
    print(sample(model, args.prompt, args.max_new_tokens, args.top_k, args.temp))