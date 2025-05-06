# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : metrics_test.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/05/06 08:30
"""
from src.core.utils.metrics import RadarPerformanceMetrics
import numpy as np

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # Example Data (replace with actual simulation results)
    example_targets = [
        {'id': 1, 'priority': 3.0},
        {'id': 2, 'priority': 1.0},
        {'id': 3, 'priority': 2.0}
    ]
    # Rows: Targets (0, 1, 2 corresponding to IDs 1, 2, 3)
    # Cols: Time steps (0, 1, 2, 3, 4)
    # Values: Radar ID (e.g., 101, 102) or 0 if not tracked
    example_assignment = np.array([
        [101, 101, 102,   0, 102],  # Target 1
        [  0, 101, 101, 101,   0],  # Target 2
        [102,   0,   0, 101, 101]   # Target 3
    ])
    # Example: Target 1 always in range, Target 2 in range steps 1-3, Target 3 in range steps 0, 3-4
    example_in_range = np.array([
        [True, True, True, True, True],
        [False, True, True, True, False],
        [True, False, False, True, True]
    ])
    example_time_step = 0.5 # seconds

    # --- Instantiate and Calculate ---
    metrics_calculator = RadarPerformanceMetrics(
        targets=example_targets,
        assignment_history=example_assignment,
        in_range_history=example_in_range,
        time_step=example_time_step
    )

    # --- Get Individual Metrics ---
    wtt = metrics_calculator.calculate_weighted_tracking_time()
    tcr = metrics_calculator.calculate_tracking_coverage_ratio()
    th = metrics_calculator.calculate_tracking_handoffs()

    print(f"Weighted Tracking Time: {wtt:.2f}")
    print(f"Tracking Coverage Ratio: {tcr:.2f}" if tcr is not None else "Tracking Coverage Ratio: N/A")
    print(f"Tracking Handoffs: {th}")

    # --- Generate Full Report ---
    full_report = metrics_calculator.generate_report()
    print("\n--- Full Report ---")
    import json
    print(json.dumps(full_report, indent=2))

    # --- Example without in_range_history ---
    print("\n--- Testing without In-Range History ---")
    metrics_calculator_no_range = RadarPerformanceMetrics(
        targets=example_targets,
        assignment_history=example_assignment,
        time_step=example_time_step
    )
    report_no_range = metrics_calculator_no_range.generate_report()
    print(json.dumps(report_no_range, indent=2))