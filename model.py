"""
Neural's language model: a small decoder-only transformer, built the same
way as modern open models (Llama-style), just smaller.

  - RMSNorm            simpler, faster layer normalization
  - Rotary positions   (RoPE) the model knows word order without a size limit table
  - SwiGLU             a better feed-forward layer
  - KV cache           fast replies: past words aren't recomputed for each new word
  - Weight tying       input and output embeddings share weights

Pick a size with a preset (see PRESETS) or set the numbers yourself.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Approximate size with an 8k-32k vocabulary. Rough guide to what each needs:
PRESETS = {
    #           width  layers heads context  vocab     ~params   good for
    "tiny":   dict(dim=256,  layers=4,  heads=4,  context=256,  vocab=8000),   # ~5M     any laptop, minutes
    "small":  dict(dim=512,  layers=8,  heads=8,  context=512,  vocab=16000),  # ~33M    laptop CPU: hours, GPU: minutes
    "medium": dict(dim=768,  layers=12, heads=12, context=1024, vocab=32000),  # ~110M   gaming GPU, a few hours
    "large":  dict(dim=1024, layers=24, heads=16, context=1024, vocab=32000),  # ~335M   strong GPU (12GB+), a day+
    "xl":     dict(dim=2048, layers=24, heads=16, context=2048, vocab=32000),  # ~1.3B   24GB GPU, days to weeks
}


def estimate_params(dim, layers, vocab, **_):
    return vocab * dim + layers * 12 * dim * dim


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def rope_tables(head_dim, length, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(length, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (batch, heads, length, head_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin, cache=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:  # generation: remember keys/values of earlier words
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=(t > 1 and k.size(2) == t),
            attn_mask=None if (t == 1 or k.size(2) == t) else self._mask(t, k.size(2), x.device),
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(b, t, d))

    @staticmethod
    def _mask(t, total, device):
        # new tokens can see all cached tokens plus earlier new tokens
        return torch.ones(t, total, dtype=torch.bool, device=device).tril(total - t)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = 64 * ((int(8 * dim / 3) + 63) // 64)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1, self.attn = RMSNorm(dim), Attention(dim, heads, dropout)
        self.norm2, self.ff = RMSNorm(dim), FeedForward(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, cache))
        return x + self.drop(self.ff(self.norm2(x)))


class NeuralLM(nn.Module):
    def __init__(self, vocab, dim=256, layers=4, heads=4, context=256, dropout=0.0):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, dropout) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, vocab, bias=False)
        self.out.weight = self.embed.weight
        cos, sin = rope_tables(dim // heads, context, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init)
        for name, p in self.named_parameters():  # smaller init for residual outputs
            if name.endswith(("proj.weight", "w2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, targets=None, caches=None, start=0):
        t = x.size(1)
        cos, sin = self.cos[start:start + t], self.sin[start:start + t]
        h = self.embed(x)
        for i, block in enumerate(self.blocks):
            h = block(h, cos, sin, caches[i] if caches is not None else None)
        logits = self.out(self.norm(h))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_tokens=200, temperature=0.7, top_k=40,
                 top_p=0.9, repetition_penalty=1.15, stop_ids=()):
        """Generate a reply token by token. ids: list of prompt token ids."""
        self.eval()
        device = self.embed.weight.device
        ids = ids[-(self.context - max_tokens):] if len(ids) > self.context - max_tokens else ids
        caches = [{} for _ in self.blocks]
        x = torch.tensor([ids], device=device)
        logits = self(x, caches=caches)[:, -1, :]
        pos, out = len(ids), []

        for _ in range(max_tokens):
            logits = logits.float().squeeze(0)
            if repetition_penalty != 1.0 and out:
                seen = torch.tensor(sorted(set(out[-64:])), device=device)
                vals = logits[seen]
                logits[seen] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            logits = logits / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_p, idx = probs.sort(descending=True)
                drop = sorted_p.cumsum(0) - sorted_p > top_p
                sorted_p[drop] = 0
                probs = torch.zeros_like(probs).scatter(0, idx, sorted_p)
            nxt = int(torch.multinomial(probs / probs.sum(), 1))
            if nxt in stop_ids or pos >= self.context:
                break
            out.append(nxt)
            logits = self(torch.tensor([[nxt]], device=device), caches=caches, start=pos)[:, -1, :]
            pos += 1
        return out
