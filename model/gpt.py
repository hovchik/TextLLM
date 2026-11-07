"""GPT-style decoder-only transformer (pre-norm, RoPE, SwiGLU, tied embeddings)."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


# Presets: "cpu" ~14M (8k vocab) trains on a laptop CPU; "small" ~110M fits one 24 GB GPU; "1b" wants multi-GPU.
PRESETS = {
    "tiny": dict(n_layer=4, n_head=4, n_embd=256, block_size=256),
    "cpu": dict(n_layer=6, n_head=6, n_embd=384, block_size=256),
    "small": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024),
    "medium": dict(n_layer=24, n_head=16, n_embd=1024, block_size=1024),
    "1b": dict(n_layer=24, n_head=16, n_embd=2048, block_size=2048),
}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x)


def rope_cache(seq_len, head_dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):  # x: (B, H, T, D)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head, self.head_dim = c.n_head, c.n_embd // c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.dropout = c.dropout

    def forward(self, x, cos, sin, kv=None):
        """kv: (cache, layer) during generation. The new keys/values are written into the cache's
        preallocated buffers, so the next token only attends to what is stored (no copying)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask, causal = None, True
        if kv is not None:
            cache, i = kv
            pos, end = cache.pos, cache.pos + T
            if cache.k[i] is None:  # first use: room for the whole window
                shape = (B, self.n_head, cache.size, self.head_dim)
                cache.k[i], cache.v[i] = k.new_empty(shape), v.new_empty(shape)
            cache.k[i][:, :, pos:end], cache.v[i][:, :, pos:end] = k, v
            k, v = cache.k[i][:, :, :end], cache.v[i][:, :, :end]
            if pos:
                causal = False
                if T > 1:  # new tokens see all cached ones plus the new ones up to themselves
                    mask = torch.ones(T, end, dtype=torch.bool, device=x.device).tril(pos)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        hidden = int(8 * c.n_embd / 3 / 64 + 1) * 64
        self.gate = nn.Linear(c.n_embd, hidden, bias=False)
        self.up = nn.Linear(c.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, c.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn, self.n2, self.mlp = RMSNorm(c.n_embd), Attention(c), RMSNorm(c.n_embd), MLP(c)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin, kv=None):
        x = x + self.drop(self.attn(self.n1(x), cos, sin, kv))
        return x + self.drop(self.mlp(self.n2(x)))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tie
        cos, sin = rope_cache(c.block_size, c.n_embd // c.n_head, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / (2 * c.n_layer) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, cache=None):
        """cache: from new_cache(), for generation. The new tokens are placed after the ones
        already in the cache, and only the last position's logits are computed."""
        T = idx.shape[1]
        pos = cache.pos if cache is not None else 0
        assert pos + T <= self.config.block_size
        x = self.wte(idx)
        cos, sin = self.cos[pos:pos + T], self.sin[pos:pos + T]
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, (cache, i) if cache is not None else None)
        if cache is not None:
            cache.pos += T
        if cache is not None and targets is None:
            x = x[:, -1:]  # the vocabulary projection is the costliest layer; skip it for the prompt
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def new_cache(self):
        return KVCache(len(self.blocks), self.config.block_size)

    @torch.no_grad()
    def stream(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95, stop_id=None, keep=0.75):
        """Yield one sampled token per step, shape (B, 1), using a key/value cache so each step
        only runs the new token through the model (instead of the whole context again).

        When the context fills the model's window, the cache is rebuilt from the last
        `keep` share of the window, so long texts keep going at the same speed."""
        bs = self.config.block_size
        cache, ctx = self.new_cache(), idx[:, -bs:]
        logits, _ = self(ctx, cache=cache)
        for _ in range(max_new_tokens):
            nxt = sample(logits[:, -1, :], temperature, top_k, top_p)
            yield nxt
            if stop_id is not None and (nxt == stop_id).all():
                return
            ctx = torch.cat([ctx, nxt], 1)
            if cache.pos >= bs:
                ctx = ctx[:, -max(1, int(bs * keep)):]
                cache.pos = 0  # reuse the buffers
                logits, _ = self(ctx, cache=cache)
            else:
                logits, _ = self(nxt, cache=cache)

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95, stop_id=None):
        out = [idx] + list(self.stream(idx, max_new_tokens, temperature, top_k, top_p, stop_id))
        return torch.cat(out, 1)


class KVCache:
    """Keys and values of the tokens seen so far, per layer, for generation. Buffers hold the
    whole window and are allocated on first use (batch size, device and dtype of the model)."""

    def __init__(self, n_layer, size):
        self.size, self.pos = size, 0
        self.k, self.v = [None] * n_layer, [None] * n_layer


def sample(logits, temperature=0.8, top_k=50, top_p=0.95):
    """Pick the next token from (B, vocab) logits with temperature, top-k and nucleus sampling."""
    logits = logits.float() / max(temperature, 1e-5)
    V = logits.size(-1)
    k = min(top_k, V) if top_k else V
    if k >= V and top_p >= 1.0:  # nothing to prune
        return torch.multinomial(torch.softmax(logits, -1), 1)
    if k >= V:  # top-p only: needs the full order
        sl, si = torch.sort(logits, descending=True)
    else:  # topk is already sorted, so nucleus sampling only touches k values instead of the whole
        sl, si = torch.topk(logits, k)  # vocabulary (sorting 50 numbers instead of 32,000 per token)
    if top_p < 1.0:
        p = torch.softmax(sl, -1)
        sl = sl.masked_fill(p.cumsum(-1) - p > top_p, -float("inf"))  # keep the smallest set over top_p
    pick = torch.multinomial(torch.softmax(sl, -1), 1)
    return si.gather(1, pick)
