"""Tests for TGlacierEdge quantized inference."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dcoss_iot_2026.deployment.quantized_inference import TGlacierEdgeNode


def test_node_runs_cycle():
    """TGlacierEdgeNode should complete a sensing cycle."""
    from shared.model.unet import UNet
    import torch
    node = TGlacierEdgeNode(node_id=0)
    model = UNet(in_channels=6)
    model.eval()
    node._inference._model = model
    node._inference.backend = "torch"
    patch = np.random.randn(6, 128, 128).astype(np.float32)
    result = node.run_sensing_cycle(patch)
    assert "sigma_meta" in result
    assert "delta_spec" in result
    assert result["delta_spec"] >= 0


def test_delta_spec_history():
    """History should grow with each cycle."""
    from shared.model.unet import UNet
    node = TGlacierEdgeNode(node_id=1)
    node._inference._model = UNet(in_channels=6)
    node._inference._model.eval()
    node._inference.backend = "torch"
    for _ in range(5):
        patch = np.random.randn(6, 128, 128).astype(np.float32)
        node.run_sensing_cycle(patch)
    assert len(node.delta_spec_series) == 5
