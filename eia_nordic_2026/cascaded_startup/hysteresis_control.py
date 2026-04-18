"""
Hysteresis control for cascaded startup (ICE-SAP §3.7, Proposition 3).

Prevents power-state chattering via dead-band [θ_down, θ_up].
Minimax optimal under uncertainty ‖p − p₀‖₁ ≤ θ.
"""


# Default hysteresis thresholds
L1_L2_THETA_UP   = 0.35
L1_L2_THETA_DOWN = 0.20
L2_L3_THETA_UP   = 0.50
L2_L3_THETA_DOWN = 0.30

# Polar-night mode
POLAR_NIGHT_L1_L2_UP   = 0.25
POLAR_NIGHT_L1_L2_DOWN = 0.15

# Blizzard / high-wind mode
BLIZZARD_L2_L3_UP   = 0.60
BLIZZARD_L2_L3_DOWN = 0.40

MIN_DWELL_S = 30.0   # minimum dwell time at each level


def get_thresholds(mode: str = "normal") -> dict:
    """
    Return hysteresis thresholds for a given operating mode.

    Args:
        mode: 'normal' | 'polar_night' | 'blizzard'

    Returns:
        dict with L1_L2 and L2_L3 up/down thresholds.
    """
    if mode == "polar_night":
        return {
            "L1_L2": {"up": POLAR_NIGHT_L1_L2_UP,
                      "down": POLAR_NIGHT_L1_L2_DOWN},
            "L2_L3": {"up": L2_L3_THETA_UP, "down": L2_L3_THETA_DOWN},
        }
    elif mode == "blizzard":
        return {
            "L1_L2": {"up": L1_L2_THETA_UP, "down": L1_L2_THETA_DOWN},
            "L2_L3": {"up": BLIZZARD_L2_L3_UP, "down": BLIZZARD_L2_L3_DOWN},
        }
    else:
        return {
            "L1_L2": {"up": L1_L2_THETA_UP, "down": L1_L2_THETA_DOWN},
            "L2_L3": {"up": L2_L3_THETA_UP, "down": L2_L3_THETA_DOWN},
        }


def minimax_cost_bound(
    nominal_cost: float,
    uncertainty_theta: float,
    horizon: int = 100,
) -> float:
    """
    Compute minimax cost bound from Proposition 3.

    max_{p} E_p[cost] ≤ E_{p₀}[cost] + O(θ·T)

    Args:
        nominal_cost:      E_{p₀}[cost] under nominal transition matrix p₀.
        uncertainty_theta: ‖p − p₀‖₁ ≤ θ.
        horizon:           Time horizon T.

    Returns:
        worst_case_cost: Upper bound on worst-case expected cost.
    """
    return nominal_cost + uncertainty_theta * horizon


if __name__ == "__main__":
    for mode in ("normal", "polar_night", "blizzard"):
        t = get_thresholds(mode)
        print(f"\n{mode} mode:")
        print(f"  L1→L2: θ_up={t['L1_L2']['up']}, θ_down={t['L1_L2']['down']}")
        print(f"  L2→L3: θ_up={t['L2_L3']['up']}, θ_down={t['L2_L3']['down']}")
