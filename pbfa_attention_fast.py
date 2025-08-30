"""
pbfa_attention_fast.py – Standalone, optimized Collapsed PBFA attention

Highlights
- Fused QKV projection to reduce kernel launches
- Chebyshev recurrence (no full stack materialization)
- Optional Triton fast path (forward) when available on CUDA
- Compatible with torch.autocast; optional torch.compile acceleration
- Causal-mask support via prefix-sum path
- Incremental decode API for autoregressive variant
- Drop-in forward(x[, attn_mask, key_padding_mask]) → (B,S,D)

This module is self-contained and can be imported from anywhere in the repo.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_clamp_unit_interval(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Inputs to Chebyshev are ideally in [-1, 1]. We softly clamp to stabilise.
    return torch.clamp(x, -1.0 + eps, 1.0 - eps)


def _alpha_to_beta(alpha: torch.Tensor) -> torch.Tensor:
    """Convert α_0..α_{P-1} to collapsed β of length (2P-1).

    β = [0, tail(α)[1:], zeros(P-1)]  then normalised
    where tail(α)[i] = Σ_{j≥i} α_j
    """
    P = alpha.numel()
    tail = torch.flip(torch.cumsum(torch.flip(alpha, (0,)), dim=0), (0,))
    beta = torch.cat([alpha.new_zeros(1), tail[1:], alpha.new_zeros(P - 1)])
    beta = beta / beta.sum()
    return beta


class CollapsedPBFAOptimized(nn.Module):
    """Optimized Collapsed PBFA attention (parallel forward).

    Parameters
    - d_model: model dimension
    - n_heads: number of heads
    - order: Chebyshev order P (β has length 2P-1)
    - bias: include bias in projections
    - learnable_power: if True, learn per-head power exponent to form β
    - fused_qkv: single Linear projecting to Q,K,V in one matmul
    - clamp_inputs: clamp scaled q/k to [-1,1] for Chebyshev stability
    - triton_fast: attempt Triton fast path when available and masks are None
    - compile_fwd: wrap forward with torch.compile if supported (SM>=7)
    - causal: if True, run causal (autoregressive) semantics for the parallel API
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        order: int = 6,
        *,
        bias: bool = False,
        learnable_power: bool = False,
        fused_qkv: bool = True,
        clamp_inputs: bool = True,
        triton_fast: bool = True,
        compile_fwd: bool = False,
        causal: bool = False,
        den_normalization: str = "none",  # 'none' | 'l1' | 'l2' | 'l2prod'
        norm_type: Optional[str] = None,  # 'layernorm' | 'rmsnorm' | 'scalenorm' | None
        norm_eps: float = 1e-5,
        block_dh: Optional[int] = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.P = order
        self.d_head = d_model // n_heads
        self.fused_qkv = fused_qkv
        self.clamp_inputs = clamp_inputs
        self.triton_fast = triton_fast
        self.causal = causal
        self._compiled = False
        self.norm: Optional[nn.Module] = None
        self.block_dh = block_dh
        dn = den_normalization.lower()
        if dn not in ("none", "l1", "l2", "l2prod"):
            raise ValueError("den_normalization must be one of: 'none', 'l1', 'l2', 'l2prod'")
        self.den_normalization = dn
        if self.block_dh is None:
            # Simple heuristic for head-dim blocking
            if self.d_head >= 128:
                self.block_dh = 64
            elif self.d_head >= 64:
                self.block_dh = 64
            elif self.d_head >= 32:
                self.block_dh = 32
            else:
                self.block_dh = self.d_head

        # Optional normalizer selection
        if norm_type is not None:
            nt = norm_type.lower()
            if nt == "layernorm":
                self.norm = nn.LayerNorm(d_model, eps=norm_eps)
            elif nt == "rmsnorm":
                self.norm = _RMSNorm(d_model, eps=norm_eps)
            elif nt == "scalenorm":
                self.norm = _ScaleNorm(d_model, eps=norm_eps)
            else:
                raise ValueError("norm_type must be one of: 'layernorm', 'rmsnorm', 'scalenorm', or None")

        if fused_qkv:
            self.in_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=bias)
            self.k_proj = nn.Linear(d_model, d_model, bias=bias)
            self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # β initialisation: either fixed power-law or learnable exponent per head
        if learnable_power:
            self.power_exponent = nn.Parameter(torch.full((n_heads,), -1.5, dtype=torch.float32))
            # Buffer beta will be materialised on first forward (device-aware)
            self.register_buffer("beta", torch.empty(0), persistent=False)
        else:
            j = torch.arange(order, dtype=torch.float32)
            alpha = (j + 1.0) ** -1.5
            beta = _alpha_to_beta(alpha)
            self.register_buffer("beta", beta, persistent=True)  # (2P-1,)
            self.power_exponent = None

        # Optionally compile the forward for additional fusion on supported GPUs
        if compile_fwd and hasattr(torch, "compile"):
            try:
                major_cc = None
                if torch.cuda.is_available():
                    major_cc, _ = torch.cuda.get_device_capability()
                if (major_cc is None) or (major_cc >= 7):
                    # Defer wrapping until first call to preserve scripted modules
                    self._want_compile = True
                else:
                    self._want_compile = False
            except Exception:
                self._want_compile = False
        else:
            self._want_compile = False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _get_beta_heads(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return β broadcast for heads: (H, 2P-1)."""
        if self.power_exponent is not None:
            P2 = 2 * self.P - 1
            # Build directly a per-head β via power-law over 2P-1 terms
            j = torch.arange(P2, dtype=torch.float32, device=device).unsqueeze(0)  # (1,P2)
            alpha_direct = (j + 1.0) ** self.power_exponent.unsqueeze(1)          # (H,P2)
            beta = alpha_direct / alpha_direct.sum(dim=-1, keepdim=True)
            return beta.to(dtype)
        else:
            # Fixed β shared across heads
            return self.beta.to(device=device, dtype=dtype).expand(self.n_heads, -1)

    def _fused_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        if self.fused_qkv:
            if self.norm is not None:
                x = self.norm(x)
            qkv = self.in_proj(x)  # (B,S,3D)
            q, k, v = qkv.split(self.d_model, dim=-1)
        else:
            x_in = self.norm(x) if self.norm is not None else x
            q = self.q_proj(x_in)
            k = self.k_proj(x_in)
            v = self.v_proj(x_in)
        # (B,H,S,Dh)
        q = q.view(B, S, H, Dh).transpose(1, 2).contiguous()
        k = k.view(B, S, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B, S, H, Dh).transpose(1, 2).contiguous()
        return q, k, v

    # (For brevity, forward and helpers omitted in this local copy)
    # In your repository, copy the full implementation from the root
    # pbfa_attention_fast.py to ensure identical behavior.

    def _forward_from_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        apply_out_proj: bool = False,
        return_denominator: bool = False,
    ) -> torch.Tensor:
        """Run core attention given pre-projected Q,K,V shaped (B,H,S,Dh).

        This bypasses QKV projection to isolate attention behaviour for sensitivity tests.
        """
        B, H, S, Dh = q.shape
        assert k.shape == v.shape == q.shape
        P = self.P
        dtype = q.dtype

        if key_padding_mask is not None:
            mask = key_padding_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(-1)  # (B,1,S,1)
            k = k.masked_fill(mask, 0)
            v = v.masked_fill(mask, 0)

        beta_h = self._get_beta_heads(device=q.device, dtype=torch.float32)  # (H,2P-1)

        if self.causal or (attn_mask is not None and attn_mask.dim() == 2 and attn_mask.size(0) == attn_mask.size(1) == S):
            inv_sqrt_d = 1.0 / math.sqrt(Dh)
            xk = k * inv_sqrt_d
            xq = q * inv_sqrt_d
            if self.clamp_inputs:
                xk = _maybe_clamp_unit_interval(xk)
                xq = _maybe_clamp_unit_interval(xq)

            P2 = 2 * P - 1

            xk32 = xk.to(torch.float32)
            xq32 = xq.to(torch.float32)
            v32 = v.to(torch.float32)

            Tk = torch.empty(B, H, S, P2, Dh, device=q.device, dtype=torch.float32)
            Tq = torch.empty(B, H, S, P2, Dh, device=q.device, dtype=torch.float32)
            Tk[..., 0, :] = 1.0
            Tq[..., 0, :] = 1.0
            if P2 > 1:
                Tk[..., 1, :] = xk32
                Tq[..., 1, :] = xq32
                for p_idx in range(2, P2):
                    Tk[..., p_idx, :] = 2.0 * xk32 * Tk[..., p_idx - 1, :] - Tk[..., p_idx - 2, :]
                    Tq[..., p_idx, :] = 2.0 * xq32 * Tq[..., p_idx - 1, :] - Tq[..., p_idx - 2, :]

            Tv = Tk * v32.unsqueeze(3)                     # (B,H,S,P2,Dh)
            kv_prefix = torch.cumsum(Tv, dim=2)            # (B,H,S,P2,Dh)
            ks_prefix = torch.cumsum(Tk.sum(dim=-1), dim=2)  # (B,H,S,P2)

            kv_beta = kv_prefix * beta_h.view(1, H, 1, -1, 1)
            k_beta = ks_prefix * beta_h.view(1, H, 1, -1)
            num = (Tq * kv_beta).sum(dim=3)                # (B,H,S,Dh)
            if self.den_normalization == "l2prod":
                tq_sum = Tq.sum(dim=-1)                    # (B,H,S,P2)
                row = (tq_sum.pow(2).sum(dim=3)).sqrt()    # (B,H,S)
                col = (k_beta.pow(2).sum(dim=3)).sqrt()    # (B,H,S)
                den = row * col                            # (B,H,S)
            else:
                dvec = (Tq.sum(dim=-1) * k_beta)           # (B,H,S,P2)
                if self.den_normalization == "none":
                    den = dvec.sum(dim=3)
                elif self.den_normalization == "l1":
                    den = dvec.abs().sum(dim=3)
                else:
                    den = (dvec.pow(2).sum(dim=3)).sqrt()
            out_h = num / (den.unsqueeze(-1) + 1e-7)
            out = out_h.to(dtype).permute(0, 2, 1, 3).contiguous().view(B, S, H * Dh)
            if apply_out_proj:
                out = self.out_proj(out)
            if return_denominator:
                return out, den
            return out

        inv_sqrt_d = 1.0 / math.sqrt(Dh)
        xk = k * inv_sqrt_d
        xq = q * inv_sqrt_d
        if self.clamp_inputs:
            xk = _maybe_clamp_unit_interval(xk)
            xq = _maybe_clamp_unit_interval(xq)

        P2 = 2 * P - 1

        xk32 = xk.to(torch.float32)
        xq32 = xq.to(torch.float32)
        v32 = v.to(torch.float32)

        num = torch.zeros(B, H, S, Dh, device=q.device, dtype=torch.float32)
        den = torch.zeros(B, H, S,     device=q.device, dtype=torch.float32)
        if self.den_normalization == "l2prod":
            tq_sum_p = torch.zeros(B, H, S, P2, device=q.device, dtype=torch.float32)
            kb_p = torch.zeros(B, H, P2, device=q.device, dtype=torch.float32)

        bd = self.block_dh or Dh
        for d0 in range(0, Dh, bd):
            d1 = min(d0 + bd, Dh)
            xk_blk = xk32[..., d0:d1]
            xq_blk = xq32[..., d0:d1]
            v_blk  = v32[..., d0:d1]

            Tk_prev = torch.ones_like(xk_blk)  # T0(k)
            Tq_prev = torch.ones_like(xq_blk)  # T0(q)
            kv0 = (Tk_prev * v_blk).sum(dim=2)           # (B,H,bd)
            k0  = Tk_prev.sum(dim=-1).sum(dim=2)         # (B,H)
            num[..., d0:d1] += beta_h[:, 0].view(1, H, 1, 1) * (Tq_prev * kv0.unsqueeze(2))
            if self.den_normalization == "l2prod":
                tq_sum_p[..., 0] += Tq_prev.sum(dim=-1)
                kb_p[..., 0] += k0
            else:
                contrib0 = beta_h[:, 0].view(1, H, 1) * (Tq_prev.sum(dim=-1) * k0.unsqueeze(2))
                if self.den_normalization == "none":
                    den += contrib0
                elif self.den_normalization == "l1":
                    den += contrib0.abs()
                else:
                    den += contrib0.pow(2)

            if P2 > 1:
                Tk_cur = xk_blk
                Tq_cur = xq_blk
                kv1 = (Tk_cur * v_blk).sum(dim=2)
                k1  = Tk_cur.sum(dim=-1).sum(dim=2)
                num[..., d0:d1] += beta_h[:, 1].view(1, H, 1, 1) * (Tq_cur * kv1.unsqueeze(2))
                if self.den_normalization == "l2prod":
                    tq_sum_p[..., 1] += Tq_cur.sum(dim=-1)
                    kb_p[..., 1] += k1
                else:
                    contrib1 = beta_h[:, 1].view(1, H, 1) * (Tq_cur.sum(dim=-1) * k1.unsqueeze(2))
                    if self.den_normalization == "none":
                        den += contrib1
                    elif self.den_normalization == "l1":
                        den += contrib1.abs()
                    else:
                        den += contrib1.pow(2)

                for p_idx in range(2, P2):
                    Tq_next = 2.0 * xq_blk * Tq_cur - Tq_prev
                    Tk_next = 2.0 * xk_blk * Tk_cur - Tk_prev
                    kvp = (Tk_next * v_blk).sum(dim=2)
                    kp  = Tk_next.sum(dim=-1).sum(dim=2)
                    num[..., d0:d1] += beta_h[:, p_idx].view(1, H, 1, 1) * (Tq_next * kvp.unsqueeze(2))
                    if self.den_normalization == "l2prod":
                        tq_sum_p[..., p_idx] += Tq_next.sum(dim=-1)
                        kb_p[..., p_idx] += kp
                    else:
                        contribp = beta_h[:, p_idx].view(1, H, 1) * (Tq_next.sum(dim=-1) * kp.unsqueeze(2))
                        if self.den_normalization == "none":
                            den += contribp
                        elif self.den_normalization == "l1":
                            den += contribp.abs()
                        else:
                            den += contribp.pow(2)
                    Tq_prev, Tq_cur = Tq_cur, Tq_next
                    Tk_prev, Tk_cur = Tk_cur, Tk_next

        if self.den_normalization == "l2prod":
            row = (tq_sum_p.pow(2).sum(dim=3)).sqrt()  # (B,H,S)
            col = ((kb_p * beta_h.view(1, H, -1)).pow(2).sum(dim=2)).sqrt()  # (B,H)
            den_final = row * col.unsqueeze(2)
            out_h = num / (den_final.unsqueeze(-1) + 1e-7)
        else:
            if self.den_normalization == "l2":
                den = den.sqrt()
            out_h = num / (den.unsqueeze(-1) + 1e-7)
        out = out_h.to(dtype).permute(0, 2, 1, 3).contiguous().view(B, S, H * Dh)
        if apply_out_proj:
            out = self.out_proj(out)
        if return_denominator:
            return out, (den_final if self.den_normalization == "l2prod" else den)
        return out

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Optional one-time compile wrapping
        if hasattr(self, "_want_compile") and self._want_compile and not self._compiled and hasattr(torch, "compile"):
            try:
                self.forward = torch.compile(self.forward)  # type: ignore[assignment]
                self._compiled = True
            except Exception:
                self._compiled = False

        B, S, _ = x.shape
        H, Dh, P = self.n_heads, self.d_head, self.P
        dtype = x.dtype

        q, k, v = self._fused_qkv(x)

        # Key padding mask (B,S) → zero-out K,V rows for masked positions
        if key_padding_mask is not None:
            mask = key_padding_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(-1)  # (B,1,S,1)
            k = k.masked_fill(mask, 0)
            v = v.masked_fill(mask, 0)

        # β per head (H, 2P-1)
        beta_h = self._get_beta_heads(device=x.device, dtype=torch.float32)  # accumulate in fp32

        # Attempt Triton fast path when allowed
        if (
            self.triton_fast
            and attn_mask is None
            and key_padding_mask is None
            and x.is_cuda
        ):
            try:
                from pbf_triton_kernels import collapsed_pbfa_forward_triton as _triton_fwd  # noqa: WPS433
                out_fast = _triton_fwd(
                    q.to(torch.float16),
                    k.to(torch.float16),
                    v.to(torch.float16),
                    beta_h.to(torch.float16),
                )
                if out_fast is not None:
                    out = out_fast.to(dtype).permute(0, 2, 1, 3).contiguous().view(B, S, H * Dh)
                    return self.out_proj(out)
            except Exception:
                pass  # fall back to reference path

        # Causal path: prefix-sum over sequence to respect triangular mask
        # This supports either explicit causal mode or an explicit square mask.
        if self.causal or (attn_mask is not None and attn_mask.dim() == 2 and attn_mask.size(0) == attn_mask.size(1) == S):
            inv_sqrt_d = 1.0 / math.sqrt(Dh)
            xk = k * inv_sqrt_d
            xq = q * inv_sqrt_d
            if self.clamp_inputs:
                xk = _maybe_clamp_unit_interval(xk)
                xq = _maybe_clamp_unit_interval(xq)

            P2 = 2 * P - 1

            # Materialise Chebyshev stacks once; two cumsums along sequence
            xk32 = xk.to(torch.float32)
            xq32 = xq.to(torch.float32)
            v32 = v.to(torch.float32)

            Tk = torch.empty(B, H, S, P2, Dh, device=x.device, dtype=torch.float32)
            Tq = torch.empty(B, H, S, P2, Dh, device=x.device, dtype=torch.float32)
            Tk[..., 0, :] = 1.0
            Tq[..., 0, :] = 1.0
            if P2 > 1:
                Tk[..., 1, :] = xk32
                Tq[..., 1, :] = xq32
                for p_idx in range(2, P2):
                    Tk[..., p_idx, :] = 2.0 * xk32 * Tk[..., p_idx - 1, :] - Tk[..., p_idx - 2, :]
                    Tq[..., p_idx, :] = 2.0 * xq32 * Tq[..., p_idx - 1, :] - Tq[..., p_idx - 2, :]

            Tv = Tk * v32.unsqueeze(3)                     # (B,H,S,P2,Dh)
            kv_prefix = torch.cumsum(Tv, dim=2)            # (B,H,S,P2,Dh)
            ks_prefix = torch.cumsum(Tk.sum(dim=-1), dim=2)  # (B,H,S,P2)

            kv_beta = kv_prefix * beta_h.view(1, H, 1, -1, 1)
            k_beta = ks_prefix * beta_h.view(1, H, 1, -1)
            num = (Tq * kv_beta).sum(dim=3)                # (B,H,S,Dh)
            dvec = (Tq.sum(dim=-1) * k_beta)               # (B,H,S,P2)
            if self.den_normalization == "none":
                den = dvec.sum(dim=3)
            elif self.den_normalization == "l1":
                den = dvec.abs().sum(dim=3)
            else:  # "l2"
                den = (dvec.pow(2).sum(dim=3)).sqrt()
            out_h = num / (den.unsqueeze(-1) + 1e-7)
            out = out_h.to(dtype).permute(0, 2, 1, 3).contiguous().view(B, S, H * Dh)
            return self.out_proj(out)

        # Reference non-causal fast path (Chebyshev recurrence, small P)
        inv_sqrt_d = 1.0 / math.sqrt(Dh)
        xk = k * inv_sqrt_d
        xq = q * inv_sqrt_d
        if self.clamp_inputs:
            xk = _maybe_clamp_unit_interval(xk)
            xq = _maybe_clamp_unit_interval(xq)

        P2 = 2 * P - 1

        # Streaming recurrence over degrees – no P2-sized buffers
        xk32 = xk.to(torch.float32)
        xq32 = xq.to(torch.float32)
        v32 = v.to(torch.float32)

        # Accumulators for final result
        num = torch.zeros(B, H, S, Dh, device=x.device, dtype=torch.float32)
        den = torch.zeros(B, H, S,     device=x.device, dtype=torch.float32)

        # Optional head-dim blocking for better cache reuse
        bd = self.block_dh or Dh
        for d0 in range(0, Dh, bd):
            d1 = min(d0 + bd, Dh)
            xk_blk = xk32[..., d0:d1]
            xq_blk = xq32[..., d0:d1]
            v_blk  = v32[..., d0:d1]

            # Degree 0
            Tk_prev = torch.ones_like(xk_blk)  # T0(k)
            Tq_prev = torch.ones_like(xq_blk)  # T0(q)
            kv0 = (Tk_prev * v_blk).sum(dim=2)           # (B,H,bd)
            k0  = Tk_prev.sum(dim=-1).sum(dim=2)         # (B,H)
            num[..., d0:d1] += beta_h[:, 0].view(1, H, 1, 1) * (Tq_prev * kv0.unsqueeze(2))
            contrib0 = beta_h[:, 0].view(1, H, 1) * (Tq_prev.sum(dim=-1) * k0.unsqueeze(2))
            if self.den_normalization == "none":
                den += contrib0
            elif self.den_normalization == "l1":
                den += contrib0.abs()
            else:  # "l2"
                den += contrib0.pow(2)

            if P2 > 1:
                # Degree 1
                Tk_cur = xk_blk
                Tq_cur = xq_blk
                kv1 = (Tk_cur * v_blk).sum(dim=2)
                k1  = Tk_cur.sum(dim=-1).sum(dim=2)
                num[..., d0:d1] += beta_h[:, 1].view(1, H, 1, 1) * (Tq_cur * kv1.unsqueeze(2))
                contrib1 = beta_h[:, 1].view(1, H, 1) * (Tq_cur.sum(dim=-1) * k1.unsqueeze(2))
                if self.den_normalization == "none":
                    den += contrib1
                elif self.den_normalization == "l1":
                    den += contrib1.abs()
                else:  # "l2"
                    den += contrib1.pow(2)

                # Degrees 2..P2-1
                for p_idx in range(2, P2):
                    Tq_next = 2.0 * xq_blk * Tq_cur - Tq_prev
                    Tk_next = 2.0 * xk_blk * Tk_cur - Tk_prev
                    kvp = (Tk_next * v_blk).sum(dim=2)
                    kp  = Tk_next.sum(dim=-1).sum(dim=2)
                    num[..., d0:d1] += beta_h[:, p_idx].view(1, H, 1, 1) * (Tq_next * kvp.unsqueeze(2))
                    contribp = beta_h[:, p_idx].view(1, H, 1) * (Tq_next.sum(dim=-1) * kp.unsqueeze(2))
                    if self.den_normalization == "none":
                        den += contribp
                    elif self.den_normalization == "l1":
                        den += contribp.abs()
                    else:  # "l2"
                        den += contribp.pow(2)
                    Tq_prev, Tq_cur = Tq_cur, Tq_next
                    Tk_prev, Tk_cur = Tk_cur, Tk_next

        if self.den_normalization == "l2":
            den = den.sqrt()
        out_h = num / (den.unsqueeze(-1) + 1e-7)
        out = out_h.to(dtype).permute(0, 2, 1, 3).contiguous().view(B, S, H * Dh)
        return self.out_proj(out)


class _RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,S,D)
        # rms over the last dim
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x_norm = x * rms
        return x_norm * self.weight


class _ScaleNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        # single scale parameter, initialised to sqrt(D)
        self.scale = nn.Parameter(torch.tensor(d_model ** 0.5))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 norm along last dim, avoid div by 0 with eps
        denom = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return self.scale * (x / denom)
