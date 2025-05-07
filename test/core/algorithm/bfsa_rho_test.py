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
from src.core.algorithm.bfsa_rho import BFSARHO

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
# id,timestep,position_x,position_y,position_z,velocity_x,velocity_y,velocity_z,target_type,priority
required_cols = ['id', 'timestep', 'position_x', 'position_y', 'position_z', 'velocity_x', 'velocity_y', 'velocity_z', 'target_type', 'priority']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV文件缺少必要列: {col}")

timestamps = sorted(df['timestep'].unique())
assignment_history = []
assignment_matrix = []  # 新增：用于评估的分配矩阵

# 4. 创建调度器实例
scheduler = BFSARHO(radar_network, window_size=3, prediction_steps=2)

for t_idx, timestep in enumerate(timestamps):
    # 获取当前时刻所有目标的观测信息
    targets_now = df[df['timestep'] == timestep]
    observed_targets = []
    targets = []
    for _, row in targets_now.iterrows():
        obs = {
            'id': row['id'],
            'position': np.array([row['position_x'], row['position_y'], row['position_z']]),
            'velocity': np.array([row['velocity_x'], row['velocity_y'], row['velocity_z']]),
            'priority': row['priority'],
            'type': row['target_type']
        }
        observed_targets.append(obs)
        targets.append({
            'id': row['id'],
            'priority': row['priority'],
            'target_type': row['target_type']
        })
    # 调用BFSA-RHO算法分配
    assignment = scheduler.solve(targets, observed_targets, t_idx)
    # 记录分配结果
    assignments_this_step = {}
    row_assignment = []
    for i, obs in enumerate(observed_targets):
        radar_idx = assignment.getrow(i).nonzero()[1]
        assigned_radar_id = int(radar_network.radar_ids[radar_idx[0]]) if radar_idx.size > 0 else 0  # 用0表示未分配
        assignments_this_step[obs['id']] = assigned_radar_id if assigned_radar_id != 0 else None
        row_assignment.append(assigned_radar_id)
    assignment_history.append({
        'timestamp': timestep,
        'assignments': assignments_this_step
    })
    assignment_matrix.append(row_assignment)  # 新增

# 先收集所有目标ID
all_target_ids = sorted({str(row['id']) for _, timestep in enumerate(timestamps) for row in df[df['timestep'] == timestep].to_dict('records')})

# 构建 assignment_matrix_np，行：目标，列：时间
assignment_matrix_np = np.zeros((len(all_target_ids), len(timestamps)), dtype=int)
target_id_to_row = {tid: idx for idx, tid in enumerate(all_target_ids)}

for t_idx, step in enumerate(assignment_history):
    assignments = step['assignments']
    for tid, radar_id in assignments.items():
        row_idx = target_id_to_row[str(tid)]
        assignment_matrix_np[row_idx, t_idx] = radar_id if radar_id is not None else 0

# 可选：in_range_history，如果有可用的目标在雷达范围内的历史，可以在此处生成
in_range_history = None  # 若有可用数据可替换

# 评估性能指标
from src.core.utils.metrics import RadarPerformanceMetrics

# 构建与 assignment_matrix_np 行顺序一致的 targets 列表
targets_for_metrics = []
# 取每个目标ID的属性（如优先级、类型），可从 df 中查找
for tid in all_target_ids:
    # 取该目标的第一条记录
    row = df[df['id'] == int(tid)].iloc[0]
    targets_for_metrics.append({
        'id': int(tid),
        'priority': row['priority'],
        'target_type': row['target_type']
    })

metrics = RadarPerformanceMetrics(
    targets=targets_for_metrics,
    assignment_history=assignment_matrix_np,
    in_range_history=in_range_history,
    time_step=1.0
)
report = metrics.generate_report()
print("BFSA-RHO调度性能评估：")
for k, v in report.items():
    print(f"{k}: {v}")

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

