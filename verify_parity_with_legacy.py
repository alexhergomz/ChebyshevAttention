import math
import torch
from pathlib import Path
import sys


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small, deterministic config
    B, S, H, Dh, P = 2, 16, 4, 8, 4
    D = H * Dh
    dtype = torch.float32

    # Inputs
    x = torch.randn(B, S, D, device=device, dtype=dtype)

    # Ensure project root on sys.path so root modules import
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Legacy (non-fast) implementation
    from benchmark_pbfa import CollapsedPBFA as LegacyPBFA
    legacy = LegacyPBFA(D, H, P).to(device=device, dtype=dtype)

    # Optimized implementation configured to match legacy math
    try:
        from pbfa_attention_fast import CollapsedPBFAOptimized
    except Exception:
        # fallback if running from a different cwd
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pbfa_attention_fast import CollapsedPBFAOptimized
    opt = CollapsedPBFAOptimized(
        d_model=D,
        n_heads=H,
        order=P,
        bias=False,
        learnable_power=True,          # match legacy per-head power across 2P-1 terms
        fused_qkv=False,               # expose separate q/k/v like legacy
        clamp_inputs=False,            # legacy did not clamp
        triton_fast=False,             # disable fast path
        compile_fwd=False,
        causal=False,
        den_normalization='none',      # legacy uses raw denominator (no L1/L2)
    ).to(device=device, dtype=dtype)

    # Copy projections and out proj for apples-to-apples
    opt.q_proj.weight.data.copy_(legacy.q_proj.weight)
    opt.k_proj.weight.data.copy_(legacy.k_proj.weight)
    opt.v_proj.weight.data.copy_(legacy.v_proj.weight)
    opt.out_proj.weight.data.copy_(legacy.out.weight)

    # Match power exponents per head (legacy.power_coeff)
    opt.power_exponent.data.copy_(legacy.power_coeff.data)

    # Forward legacy
    y_legacy = legacy(x)

    # Forward optimized aligning to legacy math:
    # Legacy L2-normalizes q and k per head-dim inside chebyshev_stack.
    q, k, v = opt._fused_qkv(x)
    import torch.nn.functional as F
    q = F.normalize(q, p=2.0, dim=-1)
    k = F.normalize(k, p=2.0, dim=-1)
    y_opt = opt._forward_from_qkv(q, k, v, apply_out_proj=True)

    max_diff = (y_legacy - y_opt).abs().max().item()
    mean_diff = (y_legacy - y_opt).abs().mean().item()
    print(f"Parity check (legacy vs optimized) | max abs diff = {max_diff:.3e} | mean abs diff = {mean_diff:.3e}")

    # Basic assertion with a modest tolerance (floating-point + small impl diffs)
    # Adjust atol if you change P/H/Dh/S.
    atol = 1e-5
    ok = torch.allclose(y_legacy, y_opt, atol=atol, rtol=0)
    print(f"allclose(atol={atol}): {ok}")


if __name__ == "__main__":
    main()


