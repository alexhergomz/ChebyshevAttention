from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cheb_T_seq(x: torch.Tensor, n: int) -> torch.Tensor:
    x = x.unsqueeze(-1)
    out = torch.empty(x.shape[:-1] + (n + 1,), device=x.device, dtype=x.dtype)
    out[..., 0] = 1.0
    if n >= 1:
        out[..., 1] = x.squeeze(-1)
        for k in range(2, n + 1):
            out[..., k] = 2.0 * x.squeeze(-1) * out[..., k - 1] - out[..., k - 2]
    return out


def _cheb_U_seq(x: torch.Tensor, m: int) -> torch.Tensor:
    x = x.unsqueeze(-1)
    out = torch.empty(x.shape[:-1] + (m + 1,), device=x.device, dtype=x.dtype)
    out[..., 0] = 1.0
    if m >= 1:
        out[..., 1] = 2.0 * x.squeeze(-1)
        for k in range(2, m + 1):
            out[..., k] = 2.0 * x.squeeze(-1) * out[..., k - 1] - out[..., k - 2]
    return out


def _phi_n(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return torch.ones(x.shape + (1,), device=x.device, dtype=x.dtype)
    x = torch.clamp(x, -1.0, 1.0)
    Tn = _cheb_T_seq(x, n)[..., n]
    Unm1 = _cheb_U_seq(x, n - 1)[..., n - 1]
    s = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))
    return torch.stack([Tn, s * Unm1], dim=-1)


def _psi_total(x_vec: torch.Tensor, max_n: int, weights: torch.Tensor) -> torch.Tensor:
    """Sum of feature maps across head-dim: ψ_total(x) = sum_d sqrt(w_n) φ_n(x_d).

    x_vec: (..., Dh)
    weights: (N+1,) softmax
    Returns: (..., F) with F = 1 + 2*max_n
    """
    parts = []
    # n=0 term
    p0 = torch.sqrt(weights[0]).reshape(1) * _phi_n(x_vec, 0)[..., 0:1]  # (...,1)
    parts.append(p0)
    for n in range(1, max_n + 1):
        pn = torch.sqrt(weights[n]).reshape(1) * _phi_n(x_vec, n)  # (...,2)
        parts.append(pn)
    # sum over Dh
    return torch.cat(parts, dim=-1).sum(dim=-2)


class PBFAFeatureMixAttention(nn.Module):
    """Chebyshev feature-map attention with convex mixture over degrees (no denominator).

    - L2-normalizes Q,K per head-dim
    - Learns per-head logits over degrees 0..N; weights = softmax(logits)
    - ψ_total(x) = sum_d sqrt(w_n) φ_n(x_d); out = ψ_total(Q) @ (sum_j ψ_total(K_j) ⊗ V_j)
    """

    def __init__(self, d_model: int, n_heads: int, max_n: int = 6, bias: bool = True, with_out_proj: bool = True) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_n = max_n
        self.with_out_proj = with_out_proj
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        # Per-head logits
        init = torch.tensor([-(n * 1.0) for n in range(max_n + 1)], dtype=torch.float32).unsqueeze(0).repeat(n_heads, 1)
        self.degree_logits = nn.Parameter(init)  # (H, N+1)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        # Project and reshape
        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2).contiguous()  # (B,H,S,Dh)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2).contiguous()
        # L2-normalize per head-dim
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)
        # Masking
        if attention_mask is not None:
            attn = attention_mask
            if attn.dim() == 4:
                attn = attn.squeeze(1).squeeze(1)
            # True for keep (HF uses 1 as keep); convert to mask
            key_pad = (attn == 0).to(torch.bool)  # (B,S)
        else:
            key_pad = None
        # Precompute per-head weights
        w = torch.softmax(self.degree_logits, dim=-1)  # (H, N+1)
        Fdim = 1 + 2 * self.max_n
        KV = torch.zeros(B, H, Fdim, Dh, device=x.device, dtype=x.dtype)
        # Accumulate over tokens (non-causal)
        for s in range(S):
            if key_pad is not None and key_pad[:, s].any():
                ks = k[:, :, s, :]  # (B,H,Dh)
                vs = v[:, :, s, :]
                # zero only for masked positions
                mask_row = (~key_pad[:, s]).float().view(B, 1, 1)
                psiKs = []
                for h in range(H):
                    psiKs.append(_psi_total(ks[:, h, :], self.max_n, w[h]))
                psiK = torch.stack(psiKs, dim=1)  # (B,H,F)
                psiK = psiK * mask_row
            else:
                ks = k[:, :, s, :]
                vs = v[:, :, s, :]
                psiKs = []
                for h in range(H):
                    psiKs.append(_psi_total(ks[:, h, :], self.max_n, w[h]))
                psiK = torch.stack(psiKs, dim=1)
            KV = KV + torch.einsum('bhf,bhd->bhfd', psiK, vs)
        # Now compute outputs for each position
        out = torch.empty(B, H, S, Dh, device=x.device, dtype=x.dtype)
        for s in range(S):
            psiQs = []
            for h in range(H):
                psiQs.append(_psi_total(q[:, h, s, :], self.max_n, w[h]))
            psiQ = torch.stack(psiQs, dim=1)  # (B,H,F)
            ctx = torch.einsum('bhf,bhfd->bhd', psiQ, KV)
            out[:, :, s, :] = ctx
        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, H * Dh)
        return self.out_proj(out) if self.with_out_proj else out


