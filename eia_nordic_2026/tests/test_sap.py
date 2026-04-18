"""Tests for SAP pruning."""
import numpy as np
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from eia_nordic_2026.sap_pruning.weighted_spatial_activation import (
    compute_wsa_scores, rank_channels_by_wsa
)


def test_wsa_boundary_concentration():
    """Channels concentrated near boundary should score higher."""
    B, C, H, W = 2, 16, 32, 32
    F = torch.zeros(B, C, H, W)
    # Channel 0: high activation at boundary (centre)
    F[:, 0, H//2-2:H//2+2, :] = 5.0
    # Channel 1: high activation away from boundary
    F[:, 1, 0:4, :] = 5.0
    # BDF: high weight at centre
    Wf = torch.zeros(B, 1, H, W)
    Wf[:, :, H//2-4:H//2+4, :] = 1.0
    scores = compute_wsa_scores(F, Wf)
    assert scores[0] > scores[1], "Boundary channel should score higher"


def test_retention_rate():
    """rank_channels_by_wsa keeps correct fraction."""
    scores = torch.rand(100)
    keep, drop = rank_channels_by_wsa(scores, retention_rate=0.166)
    assert len(keep) == 16 or len(keep) == 17  # 16.6% of 100
    assert len(keep) + len(drop) == 100


def test_wsa_nonnegative():
    """WSA scores must be non-negative."""
    F = torch.randn(4, 32, 16, 16).abs()
    W = torch.rand(4, 1, 16, 16)
    scores = compute_wsa_scores(F, W)
    assert (scores >= 0).all()
