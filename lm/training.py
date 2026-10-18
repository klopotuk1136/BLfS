"""Part 2: loss, optimizer, schedule, clipping, data loading, checkpointing and sampling."""
from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from typing import IO, BinaryIO

import numpy as np
import torch
from torch import Tensor

from lm.model import softmax


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean NLL over all positions: logsumexp(z) - z_target, with the max logit subtracted first."""
    logits = logits.float()
    z = logits - logits.amax(dim=-1, keepdim=True)
    log_sum_exp = z.exp().sum(dim=-1).log()
    target_logit = z.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logit).mean()


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay:
    m <- b1 m + (1-b1) g;  v <- b2 v + (1-b2) g^2;  theta <- theta - lr (m_hat / (sqrt(v_hat) + eps) + wd theta)."""

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1):
        if lr < 0:
            raise ValueError(f"invalid learning rate {lr}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)
                p.mul_(1 - lr * wd)                                   # decoupled weight decay
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss


def lr_cosine(step: int, lr_max: float, lr_min: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup 0 -> lr_max, cosine decay to lr_min at total_steps, constant afterwards."""
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    if step <= total_steps:
        frac = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * frac))
    return lr_min


def clip_grad_norm_(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> Tensor:
    """Scale all gradients so their global L2 norm is at most max_norm; return the norm before clipping."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total = torch.sqrt(sum(g.detach().float().pow(2).sum() for g in grads))
    if total > max_norm:
        coef = max_norm / (total + eps)
        for g in grads:
            g.mul_(coef.to(g.dtype))
    return total


def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: str | torch.device) -> tuple[Tensor, Tensor]:
    """Random windows: inputs ids[i:i+L], targets ids[i+1:i+L+1], both int64 on device."""
    starts = np.random.randint(0, len(data) - context_length, size=batch_size)   # last start = len - L - 1
    x = np.stack([data[i: i + context_length] for i in starts]).astype(np.int64)
    y = np.stack([data[i + 1: i + 1 + context_length] for i in starts]).astype(np.int64)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if str(device).startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int,
                    path: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(path: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


# --------------------------------------------------------------------------- sampling

def sample_next(logits: Tensor, temperature: float, top_p: float, generator: torch.Generator | None = None) -> int:
    """Sample one id from 1-D logits with temperature and nucleus (top-p) sampling."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    probs = softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        keep = mass_before < top_p                                    # smallest prefix with mass >= top_p
        sorted_probs = sorted_probs * keep
        choice = torch.multinomial(sorted_probs / sorted_probs.sum(), 1, generator=generator)
        return int(sorted_ids[choice])
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate_ids(model: torch.nn.Module, prompt_ids: list[int], max_new_tokens: int, temperature: float = 1.0,
                 top_p: float = 1.0, eos_id: int | None = None, generator: torch.Generator | None = None) -> list[int]:
    """Autoregressively sample up to max_new_tokens ids; stop after eos_id. Returns only the new ids."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    context_length = getattr(model, "context_length", None)
    ids, out = list(prompt_ids), []
    for _ in range(max_new_tokens):
        window = ids[-context_length:] if context_length else ids
        logits = model(torch.tensor([window], dtype=torch.long, device=device))[0, -1].cpu()
        nxt = sample_next(logits, temperature, top_p, generator)
        ids.append(nxt)
        out.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
    model.train(was_training)
    return out


def generate(model: torch.nn.Module, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0,
             top_p: float = 0.9, seed: int | None = None, eos_token: str = "<|endoftext|>") -> str:
    """Encode `prompt`, sample a continuation and decode it (stops at <|endoftext|>)."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    eos_bytes = eos_token.encode("utf-8")
    eos_id = next((i for i, b in tokenizer.vocab.items() if b == eos_bytes), None)
    new_ids = generate_ids(model, tokenizer.encode(prompt), max_new_tokens, temperature, top_p, eos_id, generator)
    if eos_id is not None and new_ids and new_ids[-1] == eos_id:
        new_ids = new_ids[:-1]
    return tokenizer.decode(new_ids)
