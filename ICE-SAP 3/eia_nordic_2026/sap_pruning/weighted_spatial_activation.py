"""
Weighted Spatial Activation (WSA) — core metric for SAP channel selection.

WSA_i = [Σ_x F_i(x) · W(x)] / [Σ_x F_i(x) + ε]

Measures whether channel i's activations concentrate within the high-weight
boundary regions defined by W(x) (the Boundary Decay Field).

Replaces standard Taylor gradient criterion, which discards filters encoding
the narrow SWIR ice–water transition zone governing >95% of IoU variance
(≤3% of pixels).

Proposition 2 (ICE-SAP):
  Under HSIC formulation, WSA_i is a consistent lower-bound estimator of
  true boundary mutual information I_true(F_i):
    I_true(F_i) ≥ WSA_i − O(√(log(1/δ)/n))
  Greedy selection achieves (1 − 1/e)-approximation to optimal channel set.

References:
  ICE-SAP §3.3 (EIA Nordic 2026).
  TGlacierEdge §III.B (DCOSS-IoT 2026).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.inference.boundary_decay_field import compute_boundary_decay_field


def compute_wsa_scores(
    feature_maps: torch.Tensor,
    W_field: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute WSA score for each channel.

        WSA_i = [Σ_x F_i(x) · W(x)] / [Σ_x F_i(x) + ε]

    Args:
        feature_maps: (B, C, H, W) — activations from a layer.
        W_field:      (B, 1, H, W) or (1, 1, H, W) — Boundary Decay Field.
        eps:          Numerical floor.

    Returns:
        wsa_scores: (C,) tensor — one score per channel.
    """
    # Mean over batch dimension
    F = feature_maps.mean(dim=0)          # (C, H, W)
    W = W_field.mean(dim=0).squeeze(0)    # (H, W)

    # Absolute activations (importance = magnitude of response)
    F_abs = F.abs()                        # (C, H, W)

    # Numerator: Σ_x |F_i(x)| · W(x)
    numerator = (F_abs * W.unsqueeze(0)).sum(dim=(-2, -1))   # (C,)

    # Denominator: Σ_x |F_i(x)| + ε
    denominator = F_abs.sum(dim=(-2, -1)) + eps               # (C,)

    return numerator / denominator    # (C,)


def compute_wsa_importance_score(
    gradient: torch.Tensor,
    feature_map: torch.Tensor,
    W_boundary: torch.Tensor,
    sigma: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the SAP importance score θ^A(f^i) from ICE-SAP §III.B:

        θ^A(f^i) = |Q[∇φ f^i · W_{∂G}]|

    where W_{∂G}(x) = exp(−d(x,∂G)/σ) is the boundary distance weight.

    This combines the Taylor criterion with spatial boundary weighting,
    preserving SWIR-sensitive filters near ice–water transitions.

    Args:
        gradient:    (B, C, H, W) gradients ∂L/∂f^i.
        feature_map: (B, C, H, W) activations.
        W_boundary:  (B, 1, H, W) Boundary Decay Field.
        sigma:       BDF bandwidth (px).
        eps:         Numerical floor.

    Returns:
        importance: (C,) importance scores.
    """
    # Taylor criterion: gradient × activation
    taylor = (gradient * feature_map).mean(dim=0)   # (C, H, W)

    W = W_boundary.mean(dim=0).squeeze(0)           # (H, W)

    # Weight by boundary proximity
    weighted = taylor * W.unsqueeze(0)              # (C, H, W)

    # Aggregate: mean over spatial dims, absolute value
    importance = weighted.abs().mean(dim=(-2, -1))  # (C,)
    return importance


class WSAHook:
    """
    Forward hook to capture feature maps and compute WSA scores.

    Usage:
        hook = WSAHook(layer, W_field)
        output = model(x)
        scores = hook.get_wsa_scores()
        hook.remove()
    """

    def __init__(self, module: nn.Module, W_field: torch.Tensor):
        self.W_field = W_field
        self._activations: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self._activations = output.detach()

    def get_wsa_scores(self) -> Optional[torch.Tensor]:
        if self._activations is None:
            return None
        return compute_wsa_scores(self._activations, self.W_field)

    def remove(self):
        self._handle.remove()


def rank_channels_by_wsa(
    wsa_scores: torch.Tensor,
    retention_rate: float = 0.166,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rank channels by WSA score and select top fraction.

    Pareto optimal operating point: 16.6% retention (ICE-SAP §3.3):
      - From 100% to 16.6%: Boundary IoU drops only 0.5 pp
      - Below 15%: cliff-edge drop >5 pp

    Args:
        wsa_scores:     (C,) WSA scores.
        retention_rate: Fraction of channels to keep (default 16.6%).

    Returns:
        keep_indices:  Indices of channels to retain.
        drop_indices:  Indices of channels to prune.
    """
    C = len(wsa_scores)
    n_keep = max(1, int(C * retention_rate))

    ranked = torch.argsort(wsa_scores, descending=True)
    keep_indices = ranked[:n_keep]
    drop_indices = ranked[n_keep:]

    return keep_indices, drop_indices


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 4, 64, 32, 32

    # Random feature maps
    F = torch.rand(B, C, H, W)

    # BDF: high weight near centre (boundary zone)
    W = torch.zeros(B, 1, H, W)
    W[:, :, H//2-4:H//2+4, :] = 1.0

    scores = compute_wsa_scores(F, W)
    print(f"WSA scores: min={scores.min():.4f}  max={scores.max():.4f}  "
          f"mean={scores.mean():.4f}")

    keep, drop = rank_channels_by_wsa(scores, retention_rate=0.166)
    print(f"Keep {len(keep)}/{C} channels ({len(keep)/C*100:.1f}%)  "
          f"[paper: 16.6%]")
