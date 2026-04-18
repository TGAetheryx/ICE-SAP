"""Tests for consensus threshold."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dcoss_iot_2026.deployment.consensus_threshold import (
    consensus_vote, tau_passing_criterion
)


def test_consensus_shape():
    probs = [np.random.rand(64, 64).astype(np.float32) for _ in range(3)]
    c = consensus_vote(probs)
    assert c.shape == (64, 64)


def test_consensus_binary():
    probs = [np.full((32, 32), 0.8, dtype=np.float32) for _ in range(3)]
    c = consensus_vote(probs, tau=0.5)
    assert np.all(c == 1.0)


def test_tau_passing():
    assert tau_passing_criterion(0.6) is True
    assert tau_passing_criterion(0.3) is False
