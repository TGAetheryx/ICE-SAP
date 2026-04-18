"""Re-export delta_spec from entropy module for convenience."""
from shared.inference.entropy import (
    boundary_entropy_field,
    delta_spec,
    delta_spec_series,
)
__all__ = ["boundary_entropy_field", "delta_spec", "delta_spec_series"]
