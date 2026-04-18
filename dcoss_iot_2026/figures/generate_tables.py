"""Generate all TGlacierEdge paper tables."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dcoss_iot_2026.experiments.run_rpi4_benchmark import (
    print_table_i, print_table_ii
)
from dcoss_iot_2026.fault_tolerance.single_node_fault import print_table_iii
from eia_nordic_2026.hardware.lora.rfm95w_config import print_lora_vs_wifi_table
from eia_nordic_2026.experiments.cross_site_generalization import (
    print_cross_site_table
)

if __name__ == "__main__":
    print_table_i()
    print_table_ii()
    print_table_iii()
    print_lora_vs_wifi_table()
    print_cross_site_table()
