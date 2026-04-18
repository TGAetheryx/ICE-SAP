"""
Unified inference pipeline for ICE-SAP / TGlacierEdge.

Supports three backends:
  - PyTorch (training / GPU)
  - ONNX Runtime INT8 (RPi 4 deployment)
  - NumPy (lightweight testing)

Inference flow:
  1. Load Sentinel-2 patch (6 bands: RGB + B11 + B12 + B10)
  2. Compute Meta-Net features → σ_meta
  3. Forward pass through SAP-compressed + SAAQ-quantised U-Net
  4. Compute Boundary Decay Field W(x)
  5. Select transmission tiles (W > τ) for SBT
  6. Compute Δ̂_spec for early-warning monitoring

References:
  ICE-SAP §4 (EIA Nordic 2026).
  TGlacierEdge §III (DCOSS-IoT 2026).
  ASPT §4 (Computers & Geosciences 2026).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from pathlib import Path


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Output of a single patch inference."""

    # Segmentation
    prob_map: np.ndarray            # (H, W) float32 ∈ [0,1]
    pred_mask: np.ndarray           # (H, W) uint8 binary
    boundary_mask: np.ndarray       # (H, W) uint8 — extracted boundary ∂G

    # BDF and meta
    W_field: np.ndarray             # (H, W) Boundary Decay Field
    sigma_meta: float               # predicted bandwidth (px)

    # Early-warning
    delta_spec: float               # Δ̂_spec = Var[H_b]

    # SBT tile selection
    selected_tiles: Optional[List[Tuple[int, int]]] = None   # (row, col) indices
    uplink_fraction: float = 1.0    # fraction of tiles selected

    # Performance
    inference_ms: float = 0.0       # wall-clock inference time (ms)
    power_mw: float = 320.0         # peak inference power (mW)


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class GlacierInference:
    """
    Single-node ICE-SAP inference engine.

    Args:
        model_path:   Path to ONNX model file (INT8 quantised) or PyTorch
                      checkpoint (.pt / .pth).
        meta_net_path: Path to Meta-Net weights (.pt).
        backend:      "onnx" (RPi 4) or "torch" (GPU/CPU training).
        tau:          SBT tile selection threshold (default 0.5).
        tile_size:    Tile size in pixels for SBT (default 16).
        seg_threshold: Segmentation binarisation threshold (default 0.5).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        meta_net_path: Optional[Union[str, Path]] = None,
        backend: str = "torch",
        tau: float = 0.5,
        tile_size: int = 16,
        seg_threshold: float = 0.5,
    ):
        self.backend = backend
        self.tau = tau
        self.tile_size = tile_size
        self.seg_threshold = seg_threshold

        self._model = None
        self._meta_net = None

        if model_path is not None:
            self.load_model(model_path)
        if meta_net_path is not None:
            self.load_meta_net(meta_net_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if path.suffix == ".onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._model = ort.InferenceSession(
                str(path), sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self.backend = "onnx"
        else:
            import torch
            from shared.model.unet import UNet
            checkpoint = torch.load(path, map_location="cpu")
            self._model = UNet(in_channels=6)
            if "model_state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self._model.load_state_dict(checkpoint)
            self._model.eval()
            self.backend = "torch"

    def load_meta_net(self, path: Union[str, Path]) -> None:
        import torch
        from shared.model.meta_net import MetaNetMLP
        checkpoint = torch.load(path, map_location="cpu")
        self._meta_net = MetaNetMLP()
        self._meta_net.load_state_dict(checkpoint)
        self._meta_net.eval()

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def __call__(self, patch: np.ndarray,
                 ndsi_prev: Optional[np.ndarray] = None) -> InferenceResult:
        """
        Run full inference pipeline on a single 6-band patch.

        Args:
            patch:     (6, H, W) float32 Sentinel-2 patch.
                       Bands: [B2, B3, B4, B11, B12, B10]
                              [Blue, Green, Red, SWIR1, SWIR2, TIR]
            ndsi_prev: Optional previous NDSI map for ΔNDSI computation.

        Returns:
            InferenceResult
        """
        import time

        assert patch.ndim == 3 and patch.shape[0] == 6, \
            f"Expected (6, H, W), got {patch.shape}"

        # Step 1: Extract Meta-Net features
        sigma_meta = self._predict_sigma(patch, ndsi_prev)

        # Step 2: Neural network forward pass
        t0 = time.perf_counter()
        prob_map = self._forward(patch)
        inference_ms = (time.perf_counter() - t0) * 1000

        # Step 3: Binarise segmentation
        pred_mask = (prob_map >= self.seg_threshold).astype(np.uint8)

        # Step 4: Extract boundary ∂G
        from shared.inference.boundary_decay_field import (
            extract_boundary_from_mask, compute_boundary_decay_field
        )
        boundary_mask = extract_boundary_from_mask(pred_mask)

        # Step 5: Compute BDF W(x)
        ndsi_map = _compute_ndsi_from_patch(patch)
        W_field = compute_boundary_decay_field(
            boundary_mask, sigma_meta, image=ndsi_map,
            entropy_alpha=0.3, normalise=True,
        )

        # Step 6: Δ̂_spec
        from shared.inference.entropy import delta_spec
        d_spec = delta_spec(prob_map)

        # Step 7: SBT tile selection
        tiles, uplink_frac = self._select_tiles(W_field)

        return InferenceResult(
            prob_map=prob_map,
            pred_mask=pred_mask,
            boundary_mask=boundary_mask,
            W_field=W_field,
            sigma_meta=sigma_meta,
            delta_spec=d_spec,
            selected_tiles=tiles,
            uplink_fraction=uplink_frac,
            inference_ms=inference_ms,
            power_mw=320.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_sigma(self, patch: np.ndarray,
                       ndsi_prev: Optional[np.ndarray] = None) -> float:
        """Predict σ_meta using Meta-Net (or return default 5.0)."""
        if self._meta_net is None:
            return 5.0   # default: Vatnajökull calibrated value

        import torch
        from shared.inference.entropy import extract_meta_net_features

        ndsi_map = _compute_ndsi_from_patch(patch)
        thermal = patch[5]           # Band B10 (TIR), index 5
        features = extract_meta_net_features(ndsi_map, thermal, ndsi_prev)
        feat_t = torch.tensor(features[np.newaxis], dtype=torch.float32)

        with torch.no_grad():
            sigma = self._meta_net(feat_t).item()
        return float(sigma)

    def _forward(self, patch: np.ndarray) -> np.ndarray:
        """Run segmentation model forward pass → probability map."""
        if self.backend == "onnx":
            return self._forward_onnx(patch)
        else:
            return self._forward_torch(patch)

    def _forward_onnx(self, patch: np.ndarray) -> np.ndarray:
        inp = patch[np.newaxis].astype(np.float32)   # (1, 6, H, W)
        input_name = self._model.get_inputs()[0].name
        out = self._model.run(None, {input_name: inp})[0]  # (1, 1, H, W)
        logits = out[0, 0]
        return _sigmoid(logits)

    def _forward_torch(self, patch: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            inp = torch.from_numpy(patch[np.newaxis]).float()
            logits = self._model(inp)[0, 0]
            return torch.sigmoid(logits).numpy()

    def _select_tiles(
        self, W_field: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], float]:
        """
        SBT tile selection: select tiles where mean W(x) > τ.

        Returns:
            tiles: list of (row, col) tile indices
            uplink_fraction: fraction of selected tiles
        """
        H, W = W_field.shape
        ts = self.tile_size
        rows = H // ts
        cols = W // ts
        total = rows * cols
        selected = []

        for r in range(rows):
            for c in range(cols):
                tile_w = W_field[r*ts:(r+1)*ts, c*ts:(c+1)*ts]
                if tile_w.mean() > self.tau:
                    selected.append((r, c))

        frac = len(selected) / max(total, 1)
        return selected, frac


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _compute_ndsi_from_patch(patch: np.ndarray) -> np.ndarray:
    """Extract NDSI from 6-band patch (bands: B2 B3 B4 B11 B12 B10)."""
    b3   = patch[1]    # Green
    b11  = patch[3]    # SWIR1
    ndsi = (b3 - b11) / (b3 + b11 + 1e-8)
    return np.clip(ndsi, -1.0, 1.0).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


if __name__ == "__main__":
    # Sanity check with random patch (no model weights)
    engine = GlacierInference(backend="torch")

    # Manually create a tiny model for testing
    from shared.model.unet import UNet
    engine._model = UNet(in_channels=6)
    engine._model.eval()

    patch = np.random.randn(6, 128, 128).astype(np.float32)
    result = engine(patch)
    print(f"prob_map shape:   {result.prob_map.shape}")
    print(f"sigma_meta:       {result.sigma_meta:.2f} px")
    print(f"delta_spec:       {result.delta_spec:.4f}")
    print(f"uplink_fraction:  {result.uplink_fraction:.2%}")
    print(f"inference_ms:     {result.inference_ms:.1f} ms")
