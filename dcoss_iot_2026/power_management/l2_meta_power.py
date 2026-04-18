"""l2_meta_power — see cascaded_power.py and eia_nordic_2026/cascaded_startup/ for full implementation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eia_nordic_2026.cascaded_startup.l1_ghost import L1_POWER_MW, L1_DUTY
from eia_nordic_2026.cascaded_startup.l2_meta  import L2_POWER_MW, L2_DUTY
from eia_nordic_2026.cascaded_startup.l3_core  import L3_POWER_MW, L3_DUTY, average_system_power

L1_CONTRIBUTION_MW = L1_POWER_MW * L1_DUTY   # 13.5 mW
L2_CONTRIBUTION_MW = L2_POWER_MW * L2_DUTY   #  6.4 mW
L3_CONTRIBUTION_MW = L3_POWER_MW * L3_DUTY   #  6.4 mW
TOTAL_AVG_MW       = L1_CONTRIBUTION_MW + L2_CONTRIBUTION_MW + L3_CONTRIBUTION_MW

if __name__ == "__main__":
    print(f"L1: {L1_POWER_MW} mW × {L1_DUTY:.0%} = {L1_CONTRIBUTION_MW:.1f} mW")
    print(f"L2: {L2_POWER_MW} mW × {L2_DUTY:.0%} = {L2_CONTRIBUTION_MW:.1f} mW")
    print(f"L3: {L3_POWER_MW} mW × {L3_DUTY:.0%} = {L3_CONTRIBUTION_MW:.1f} mW")
    print(f"Total P_avg = {TOTAL_AVG_MW:.1f} mW  (paper: 23.9 mW)")
