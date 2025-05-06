#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：radar
@File    ：BFSA_rho_test.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/5/6 17:28
"""
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import logging

from src.core.models.radar_model import Radar, RadarNetwork
from bfsa_rho import BFSARHO

# 配置日志记录
logging.basicConfig(level=logging.INFO)

# 1. 创建 RadarNetwork 实例
radars_list = [
    Radar(radar_id=1, radar_position=np.array([0, 0, 0]), radar_radius=100000, num_channels=5),
    Radar(radar_id=2, radar_position=np.array([50000, 0, 0]), radar_radius=120000, num_channels=8)
    # ... 可添加更多雷达
]
radar_network = RadarNetwork({r.radar_id: r for r in radars_list})

# 2. 指定数据目录和文件
data_directory = r"c:\Users\Reznov Lee\PycharmProjects\radar\output\scenario-2025-04-28"
data_filename = "10-targets.csv"
data_path = os.path.join(data_directory, data_filename)

# 3. 读取目标观测数据
df = pd.read_csv(data_path)
required_cols = ['id', 'timestamp', 'position_x', 'position_y', 'position_z', 'velocity_x', 'velocity_y', 'velocity_z', 'priority', 'type']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV文件缺少必要列: {col}")

timestamps = sorted(df['timestamp'].unique())
assignment_history = []

# 4. 创建调度器实例
scheduler = BFSARHO(radar_network, window_size=3, prediction_steps=2)

for t_idx, timestamp in enumerate(timestamps):
    # 获取当前时刻所有目标的观测信息
    targets_now = df[df['timestamp'] == timestamp]
    observed_targets = []
    targets = []
    for _, row in targets_now.iterrows():
        obs = {
            'id': row['id'],
            'position': np.array([row['position_x'], row['position_y'], row['position_z']]),
            'velocity': np.array([row['velocity_x'], row['velocity_y'], row['velocity_z']]),
            'priority': row['priority'],
            'type': row['type']
        }
        observed_targets.append(obs)
        targets.append({
            'id': row['id'],
            'priority': row['priority'],
            'type': row['type']
        })
    # 调用BFSA-RHO算法分配
    assignment = scheduler.solve(targets, observed_targets, t_idx)
    # 记录分配结果
    assignments_this_step = {}
    for i, obs in enumerate(observed_targets):
        radar_idx = assignment.getrow(i).nonzero()[1]
        assigned_radar_id = int(radar_network.radar_ids[radar_idx[0]]) if radar_idx.size > 0 else None
        assignments_this_step[obs['id']] = assigned_radar_id
    assignment_history.append({
        'timestamp': timestamp,
        'assignments': assignments_this_step
    })

# 5. 保存分配结果
results_save_path = os.path.join(data_directory, "bfsa_rho_assignment_history.json")
try:
    with open(results_save_path, 'w') as f:
        json.dump(assignment_history, f, indent=4)
    print(f"分配结果已保存到: {results_save_path}")
except Exception as e:
    print(f"保存结果时出错: {e}")

# 6. 计算并绘制分配比例
timestamps_plot = []
assigned_ratios = []
for step_data in assignment_history:
    timestamp = step_data['timestamp']
    assignments = step_data['assignments']
    total_targets = len(assignments)
    if total_targets > 0:
        assigned_count = sum(1 for radar_id in assignments.values() if radar_id is not None)
        ratio = assigned_count / total_targets
    else:
        ratio = 0
    timestamps_plot.append(timestamp)
    assigned_ratios.append(ratio)

plt.figure(figsize=(12, 6))
plt.plot(timestamps_plot, assigned_ratios, marker='.', linestyle='-')
plt.title('Ratio of Assigned Targets Over Time (BFSA-RHO)')
plt.xlabel('Timestamp')
plt.ylabel('Assigned Ratio (Assigned / Total)')
plt.ylim(0, 1.1)
plt.grid(True)
plt.tight_layout()

plot_save_path = os.path.join(data_directory, "bfsa_rho_assigned_ratio.png")
try:
    plt.savefig(plot_save_path)
    print(f"分配比例图已保存到: {plot_save_path}")
except Exception as e:
    print(f"保存图形时出错: {e}")

plt.show()

