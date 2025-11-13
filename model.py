import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
    
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x)
        
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        
        if mask is not None:
            att += mask
            
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out(out)
        
        return out
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x
    
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, seq_len=128, n_layer=4, d_model=128, n_head=4, d_ff=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, d_ff) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.seq_len = seq_len
        
    def forward(self, x):
        B, T = x.size()
        assert T <= self.seq_len, "Input sequence length exceeds model's maximum sequence length"
            
        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb[:, :T, :]
        x = token_embeddings + position_embeddings
            
        mask = torch.triu(torch.ones(T, T, device = x.device) * float('-inf'), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
                
        x = self.ln_f(x)
        logits = self.head(x)
        return logits 