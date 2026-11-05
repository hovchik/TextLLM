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

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
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

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.n1(x), cos, sin))
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

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        assert T <= self.config.block_size
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x, self.cos[:T], self.sin[:T])
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95, stop_id=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cum = torch.softmax(sl, -1).cumsum(-1)
                sl[cum - torch.softmax(sl, -1) > top_p] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(1, si, sl)
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if stop_id is not None and (nxt == stop_id).all():
                break
        return idx
