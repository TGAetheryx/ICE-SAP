"""
Generate all tables from ICE-SAP paper (EIA Nordic 2026).

Table 1: Gaps in prior work
Table 2: 56-site segmentation metrics
Table 3: System performance on RPi 4
Table 4: Energy per inference cycle
Table 5: Attribution analysis (mIoU)
Table 6: Component contributions (Boundary IoU)
"""


def print_table1():
    print("\n=== Table 1. Gaps in Prior Work ===")
    rows = [
        ("Disciplinary isolation",      "[10,15]"),
        ("Physics-free AI",             "[2,15]"),
        ("Insufficient validation",     "[5,11]"),
        ("Unresolved trilemma",         "[1,16]"),
        ("Missing engineering found.",  "[10,15]"),
    ]
    print(f"  {'Gap':<32} {'References':>12}")
    print("-"*48)
    for g, r in rows:
        print(f"  {g:<32} {r:>12}")


def print_table2():
    print("\n=== Table 2. 56-Site Segmentation Performance ===")
    metrics = [
        ("Overall IoU (%)",        "91.0 ± 1.2", "91.3"),
        ("Boundary IoU (%)",       "87.8 ± 2.1", "88.1"),
        ("Pixel Accuracy (%)",     "94.2 ± 1.0", "94.5"),
        ("Ice-edge error (m)",     "10.7 ± 1.5", "10.3"),
        ("Jökulhlaup recall (%)",  "96.5 ± 1.4", "96.8"),
    ]
    print(f"  {'Metric':<26} {'Mean±Std':>14} {'Median':>8}")
    print("-"*52)
    for m, ms, med in metrics:
        print(f"  {m:<26} {ms:>14} {med:>8}")


def print_table3():
    print("\n=== Table 3. System Performance on RPi 4 ===")
    rows = [
        ("Peak inference power",      "356±16 mW",  "Single forward pass"),
        ("Average system power",      "27.0 mW",    "Cascaded startup (L1/L2/L3)"),
        ("Battery life",              "22.8±1.5 d", "4000 mAh cell"),
        ("Comm power (TX)",           "3.5±0.3 mW", "LoRa SF10 +14 dBm"),
        ("Comm power (RX)",           "9.2±0.4 mW", "Listen mode"),
        ("Uplink reduction",          "93.8±0.6%",  "SBT τ=0.5"),
        ("Inference latency",         "210±6 ms",   "ONNX Runtime INT8"),
        ("Model size",                "1.8 MB",     "After pruning+quant."),
    ]
    print(f"  {'Metric':<28} {'Value':>14} {'Notes'}")
    print("-"*70)
    for m, v, n in rows:
        print(f"  {m:<28} {v:>14}  {n}")


def print_table4():
    print("\n=== Table 4. Energy per Inference Cycle ===")
    comps = [
        ("L1 Ghost Sensing",   0.015, 12),
        ("L2 Meta Inference",  0.008,  6),
        ("L3 Core Execution",  0.064, 51),
        ("Communication",      0.039, 31),
    ]
    print(f"  {'Component':<28} {'Energy (J)':>12} {'%':>6}")
    print("-"*50)
    for c, e, pct in comps:
        print(f"  {c:<28} {e:>12.3f} {pct:>5}%")
    print(f"  {'Total':<28} {'0.126':>12}  100%")


def print_table5():
    print("\n=== Table 5. Attribution Analysis (mIoU) ===")
    rows = [
        ("MobileNetV3",    "Standard CE",  "78.2%"),
        ("MobileNetV3",    "L_geo",        "81.5%"),
        ("KD",             "KD loss",      "80.1%"),
        ("TGlacierEdge",   "L_geo + WSA",  "83.7%"),
    ]
    print(f"  {'Config':<16} {'Training Loss':<14} {'mIoU':>8}")
    print("-"*42)
    for c, l, m in rows:
        print(f"  {c:<16} {l:<14} {m:>8}")
    print("\n  Note: Table 5 uses equal class weights (ablation benchmark).")
    print("  Table 2 uses natural class distribution (56-site benchmark).")


def print_table6():
    print("\n=== Table 6. Component Contributions (Boundary IoU) ===")
    rows = [
        ("Standard Taylor pruning",  "N/A",     "85.2±2.1", "—"),
        ("Standard PTQ",             "N/A",     "86.0±1.8", "—"),
        ("Full TGlacierEdge",        "356±12",  "87.8±2.1", "93.8±0.6"),
        ("w/o boundary pruning",     "415±18",  "86.2±2.4", "93.5±0.7"),
        ("w/o SAAQ",                 "505±22",  "88.1±2.0", "94.0±0.5"),
    ]
    print(f"  {'Config':<28} {'Power(mW)':>10} {'B-IoU(%)':>12} {'DataRed(%)':>12}")
    print("-"*66)
    for r in rows:
        print(f"  {r[0]:<28} {r[1]:>10} {r[2]:>12} {r[3]:>12}")


if __name__ == "__main__":
    print_table1()
    print_table2()
    print_table3()
    print_table4()
    print_table5()
    print_table6()
