#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：radar
@File    ：plot_test.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/5/7 10:10
"""

import os
import json
import matplotlib.pyplot as plt

# 数据目录
data_directory = r"c:\Users\Reznov Lee\PycharmProjects\radar\output\scenario-2025-04-28"
bfsa_file = os.path.join(data_directory, "bfsa_rho_assignment_history.json")
rule_file = os.path.join(data_directory, "rule_based_assignment_history.json")

def load_assignment_ratio(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    timestamps = []
    ratios = []
    for step_data in data:
        timestamp = step_data['timestamp']
        assignments = step_data['assignments']
        total_targets = len(assignments)
        if total_targets > 0:
            assigned_count = sum(1 for radar_id in assignments.values() if radar_id is not None)
            ratio = assigned_count / total_targets
        else:
            ratio = 0
        timestamps.append(timestamp)
        ratios.append(ratio)
    return timestamps, ratios

# 加载两种算法的分配比例
bfsa_timestamps, bfsa_ratios = load_assignment_ratio(bfsa_file)
rule_timestamps, rule_ratios = load_assignment_ratio(rule_file)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(bfsa_timestamps, bfsa_ratios, marker='o', linestyle='-', label='BFSA-RHO')
plt.plot(rule_timestamps, rule_ratios, marker='s', linestyle='--', label='Rule-based')
plt.title('Assigned Ratio Comparison: BFSA-RHO vs Rule-based')
plt.xlabel('Timestamp')
plt.ylabel('Assigned Ratio (Assigned / Total)')
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存合成图
plot_save_path = os.path.join(data_directory, "assigned_ratio_comparison.png")
try:
    plt.savefig(plot_save_path)
    print(f"合成分配比例图已保存到: {plot_save_path}")
except Exception as e:
    print(f"保存图形时出错: {e}")

plt.show()

