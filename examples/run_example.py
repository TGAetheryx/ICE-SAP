"""Quick-start example: run TGlacierEdge inference on a sample tile."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.model.unet import UNet
from shared.inference.infer import GlacierInference
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    print("=== TGlacierEdge Quick-Start Example ===")
    engine = GlacierInference(model_path=args.model, backend="torch")
    model = UNet(in_channels=6)
    model.eval()
    engine._model = model
    np.random.seed(42)
    patch = np.random.randn(6, 128, 128).astype(np.float32)
    result = engine(patch)
    print(f"  sigma_meta: {result.sigma_meta:.2f} px")
    print(f"  delta_spec: {result.delta_spec:.4f}")
    print(f"  uplink reduction: {(1-result.uplink_fraction)*100:.1f}%")
    print(f"  inference: {result.inference_ms:.0f} ms")

if __name__ == "__main__":
    main()
