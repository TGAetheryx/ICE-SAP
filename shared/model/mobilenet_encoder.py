"""
MobileNetV2 encoder (standalone module for clarity).

Re-exports MobileNetV2Encoder from unet.py and adds a 6-channel pretrained
weight initialisation helper (initialise the extra channels from the mean of
the RGB channels — a common strategy for multi-spectral adaptation).

Reference:
  Howard et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks",
  CVPR 2018.
  Adaptation strategy: ICE-SAP §4.3 (Baseline Fairness and Adaptation).
"""

import torch
import torch.nn as nn
from shared.model.unet import MobileNetV2Encoder


def init_6ch_from_3ch(encoder: MobileNetV2Encoder,
                      pretrained_3ch_weights: dict = None) -> None:
    """
    Initialise 6-channel first convolution from 3-channel ImageNet weights.

    The 3 extra channels (SWIR1, SWIR2, TIR) are initialised to the channel
    mean of the original 3-channel weights — this preserves the spectral
    structure while providing a reasonable starting point for fine-tuning.

    Strategy from: ICE-SAP §4.3:
      "We replace the first convolutional layer with a 6-channel equivalent,
       randomly initialize its weights, and retrain the entire network from
       scratch."
    Here we offer the mean-initialisation as an alternative to random init.

    Args:
        encoder:                 MobileNetV2Encoder with in_channels=6.
        pretrained_3ch_weights:  Optional state dict from a 3-channel model.
    """
    if pretrained_3ch_weights is None:
        # Random Xavier initialisation
        nn.init.xavier_uniform_(encoder.stem.block[0].weight)
        return

    # Original 3-channel weight: (out_ch, 3, kH, kW)
    w3 = pretrained_3ch_weights.get("stem.block.0.weight")
    if w3 is None:
        return

    out_ch, _, kH, kW = w3.shape
    # New 6-channel weight: (out_ch, 6, kH, kW)
    w6 = torch.zeros(out_ch, 6, kH, kW, dtype=w3.dtype)
    w6[:, :3, :, :] = w3
    # Extra channels initialised to mean of RGB weights
    rgb_mean = w3.mean(dim=1, keepdim=True)   # (out_ch, 1, kH, kW)
    w6[:, 3:, :, :] = rgb_mean.expand(-1, 3, -1, -1)

    with torch.no_grad():
        encoder.stem.block[0].weight.copy_(w6)


if __name__ == "__main__":
    enc = MobileNetV2Encoder(in_channels=6)
    x = torch.randn(2, 6, 128, 128)
    features = enc(x)
    for i, f in enumerate(features):
        print(f"skip{i}: {f.shape}")
