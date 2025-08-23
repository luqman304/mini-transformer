import math, argparse, torch, torch.nn as nn
from tqdm import tqdm

class ByteTokenizer:
    def __init__(self): self.vocab_size = 256
    def encode(self, s): return list(s.encode('utf-8', errors='ignore'))
    def decode(self, ids): return bytes([max(0, min(255, i)) for i in ids]).decode('utf-8', errors='ignore')

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__(); assert d_model % n_heads == 0
        self.n_heads = n_heads; self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False); self.o = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x, attn_mask=None):
        B,T,C = x.shape
        qkv = self.qkv(x).view(B,T,3,self.n_heads,self.d_head).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
        if attn_mask is not None: att = att.masked_fill(attn_mask==0, float('-inf'))
        w = att.softmax(dim=-1)
        out = (w @ v).transpose(2,1).contiguous().view(B,T,C)
        return self.o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=4):
        super().__init__(); hidden = d_model*d_ff
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=4):
        super().__init__(); self.ln1=nn.LayerNorm(d_model); self.attn=MultiHeadSelfAttention(d_model, n_heads)
        self.ln2=nn.LayerNorm(d_model); self.ff=FeedForward(d_model, d_ff)
    def forward(self, x, mask=None): x = x + self.attn(self.ln1(x), attn_mask=mask); x = x + self.ff(self.ln2(x)); return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=256, n_heads=4, n_layers=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model); self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model); self.head = nn.Linear(d_model, vocab_size, bias=False); self.seq_len=seq_len
    def forward(self, idx):
        B,T = idx.shape; pos = torch.arange(0, T, device=idx.device).unsqueeze(0).expand(B,T)
        x = self.token_emb(idx) + self.pos_emb(pos)
        mask = torch.tril(torch.ones(T,T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks: x = blk(x, mask)
        x = self.ln_f(x); return self.head(x)

def make_dataset(text, tok, seq_len):
    ids = tok.encode(text); n = len(ids) - seq_len - 1
    X,Y=[],[]
    for i in range(max(0, n)):
        X.append(ids[i:i+seq_len]); Y.append(ids[i+1:i+seq_len+1])
    if not X: ids = ids*(seq_len*4//max(1,len(ids))+1); return make_dataset(tok.decode(ids), tok, seq_len)
    return torch.tensor(X), torch.tensor(Y)

def iter_batches(X,Y,batch,device):
    import random; idx=list(range(len(X))); random.shuffle(idx)
    for i in range(0,len(idx),batch):
        j=idx[i:i+batch]; yield X[j].to(device), Y[j].to(device)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = ByteTokenizer()
    with open("data/toy.txt","r",encoding="utf-8") as f: text=f.read()
    X,Y = make_dataset(text, tok, args.seq_len)
    model = DecoderOnlyTransformer(tok.vocab_size, args.seq_len, args.d_model, args.n_heads, args.n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, args.epochs+1):
        tot=cnt=0.0
        for xb,yb in tqdm(iter_batches(X,Y,args.batch_size,device), total=len(X)//args.batch_size+1):
            logits = model(xb); loss = loss_fn(logits.reshape(-1, tok.vocab_size), yb.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); cnt += 1
        import math
        print(f"epoch {ep}: loss={tot/max(1,cnt):.3f}, ppl={math.exp(tot/max(1,cnt)):.2f}")

if __name__ == '__main__':
    from tqdm import tqdm
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seq-len', type=int, default=128)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    args = ap.parse_args(); main(args)
