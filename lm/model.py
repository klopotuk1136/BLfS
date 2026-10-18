"""Part 2: Transformer LM built from nn.Module + nn.Parameter only."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """y = x W^T with W of shape (out_features, in_features); no bias."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(w)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...i,oi->...o", x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(w, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(w)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        xf = x.float()
        rms_inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms_inv).to(in_dtype) * self.weight.to(in_dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def default_d_ff(d_model: int) -> int:
    """8/3 * d_model rounded to the nearest multiple of 64 (1344 for d_model = 512)."""
    return int(round(8 * d_model / 3 / 64)) * 64


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        d_ff = d_ff or default_d_ff(d_model)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    """Rotates pairs (x[2i], x[2i+1]) at position m by m * theta^(-2i/d_k)."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        inv_freq = theta ** (-torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)   # (d_k/2,)
        ang = torch.arange(max_seq_len, dtype=torch.float32, device=device)[:, None] * inv_freq      # (L, d_k/2)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos, sin = self.cos[token_positions], self.sin[token_positions]       # (..., seq, d_k/2)
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
        y = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
        return y.to(x.dtype)


def softmax(x: Tensor, dim: int) -> Tensor:
    z = x - x.amax(dim=dim, keepdim=True)
    e = z.exp()
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """softmax(q k^T / sqrt(d_k) + M) v; mask is boolean with True = may attend. Softmax in fp32."""
    d_k = q.shape[-1]
    scores = torch.einsum("...qd,...kd->...qk", q.float(), k.float()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = softmax(scores, dim=-1)
    return torch.einsum("...qk,...kd->...qd", probs, v.float()).to(q.dtype)


class MultiheadSelfAttention(nn.Module):
    """Causal self-attention with grouped-query attention and optional RoPE on q and k."""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int | None = None,
                 rope: RotaryPositionalEmbedding | None = None, causal: bool = True, device=None, dtype=None):
        super().__init__()
        num_kv_heads = num_kv_heads or num_heads
        assert d_model % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.d_head = d_model // num_heads
        self.causal = causal
        self.rope = rope
        self.q_proj = Linear(d_model, num_heads * self.d_head, device, dtype)
        self.k_proj = Linear(d_model, num_kv_heads * self.d_head, device, dtype)
        self.v_proj = Linear(d_model, num_kv_heads * self.d_head, device, dtype)
        self.output_proj = Linear(num_heads * self.d_head, d_model, device, dtype)

    def _split(self, x: Tensor, n: int) -> Tensor:
        *lead, seq, _ = x.shape
        return x.reshape(*lead, seq, n, self.d_head).transpose(-3, -2)        # (..., n, seq, d_head)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        *lead, seq, _ = x.shape
        q = self._split(self.q_proj(x), self.num_heads)
        k = self._split(self.k_proj(x), self.num_kv_heads)
        v = self._split(self.v_proj(x), self.num_kv_heads)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device)
            pos = token_positions.unsqueeze(-2)                               # broadcast over heads
            q, k = self.rope(q, pos), self.rope(k, pos)
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=-3)
            v = v.repeat_interleave(rep, dim=-3)
        mask = None
        if self.causal:
            mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        o = scaled_dot_product_attention(q, k, v, mask)                        # (..., h, seq, d_head)
        o = o.transpose(-3, -2).reshape(*lead, seq, self.num_heads * self.d_head)
        return self.output_proj(o)


class TransformerBlock(nn.Module):
    """Pre-norm: h = x + Attn(RMSNorm(x)); y = h + SwiGLU(RMSNorm(h))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int | None = None,
                 theta: float | None = 10000.0, num_kv_heads: int | None = None, eps: float = 1e-5,
                 device=None, dtype=None):
        super().__init__()
        rope = None
        if theta is not None and max_seq_len is not None:
            rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device)
        self.ln1 = RMSNorm(d_model, eps, device, dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, num_kv_heads, rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        h = x + self.attn(self.ln1(x), token_positions)
        return h + self.ffn(self.ln2(h))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int,
                 d_ff: int, rope_theta: float = 10000.0, num_kv_heads: int | None = None, device=None, dtype=None):
        super().__init__()
        self.vocab_size, self.context_length = vocab_size, context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, num_kv_heads,
                             device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, ids: Tensor) -> Tensor:
        x = self.token_embeddings(ids)
        pos = torch.arange(ids.shape[-1], device=ids.device)
        for layer in self.layers:
            x = layer(x, pos)
        return self.lm_head(self.ln_final(x)).float()


# --------------------------------------------------------------------------- accounting (Lecture 2)

def _matmul_params(d_model: int, num_layers: int, num_heads: int, d_ff: int, num_kv_heads: int | None) -> int:
    """Parameters of all matmuls inside the blocks."""
    d_head = d_model // num_heads
    nkv = num_kv_heads or num_heads
    attn = 2 * d_model * d_model + 2 * nkv * d_head * d_model                   # W_Q, W_O, W_K, W_V
    ffn = 3 * d_model * d_ff
    return num_layers * (attn + ffn)


def count_params(vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,
                 num_kv_heads: int | None = None, tied: bool = False) -> int:
    """Embedding + blocks (incl. two RMSNorm gains each) + final norm + (untied) head. RoPE has no params."""
    blocks = _matmul_params(d_model, num_layers, num_heads, d_ff, num_kv_heads) + num_layers * 2 * d_model
    head = 0 if tied else vocab_size * d_model
    return vocab_size * d_model + blocks + d_model + head


def count_flops(vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,
                seq_len: int, num_kv_heads: int | None = None) -> tuple[int, int]:
    """(forward, training) FLOPs per token: 2 x matmul params (blocks + head) + attention scores
    (QK^T and AV: 2 * 2 * seq_len * d_model per layer); training = 3 x forward."""
    matmul = _matmul_params(d_model, num_layers, num_heads, d_ff, num_kv_heads) + vocab_size * d_model
    fwd = 2 * matmul + 4 * num_layers * seq_len * d_model
    return fwd, 3 * fwd
