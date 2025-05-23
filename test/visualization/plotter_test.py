# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : plotter_test.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 15:56
"""

import json
import os
import numpy as np
from src.visualization.plotter import RadarPlotter

# 初始化绘图器
plotter = RadarPlotter(figsize=(14, 8))

# 定义文件路径
scenario_dir = "scenario-2025-05-15"
bfsa_rho_file = os.path.join(scenario_dir, "bfsa_rho_assignment_history.json")
rule_based_file = os.path.join(scenario_dir, "rule_based_assignment_history.json")
output_dir = os.path.join(scenario_dir, "visualization")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)


# 读取分配历史数据
def load_assignment_history(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


bfsa_rho_history = load_assignment_history(bfsa_rho_file)
rule_based_history = load_assignment_history(rule_based_file)


# 提取雷达和目标信息
def extract_info(history):
    radars = set()
    targets = set()
    for record in history:
        assignments = record["assignments"]
        for target_id, assignment in assignments.items():
            if assignment is not None and assignment["radar_id"] is not None:
                radars.add(assignment["radar_id"])
                targets.add(target_id)
    return radars, targets


bfsa_radars, bfsa_targets = extract_info(bfsa_rho_history)
rule_radars, rule_targets = extract_info(rule_based_history)

# 合并雷达和目标信息
all_radars = bfsa_radars.union(rule_radars)
all_targets = bfsa_targets.union(rule_targets)

# 构建雷达信息字典 (假设每个雷达有2个通道)
# 动态统计每个雷达实际用到的最大通道号，确保不会KeyError
# 构建雷达信息字典
radar_info = {}
for radar_id in all_radars:  # 使用 all_radars 而不是 bfsa_radars
    max_channel_id = -1
    # 检查 BFSA-RHO 的历史数据
    for record in bfsa_rho_history:
        for _, assignment in record["assignments"].items():
            if assignment is not None and assignment["radar_id"] == radar_id and assignment["channel_id"] is not None:
                if assignment["channel_id"] > max_channel_id:
                    max_channel_id = assignment["channel_id"]
    
    # 检查 Rule-Based 的历史数据
    for record in rule_based_history:
        for _, assignment in record["assignments"].items():
            if assignment is not None and assignment["radar_id"] == radar_id and assignment["channel_id"] is not None:
                if assignment["channel_id"] > max_channel_id:
                    max_channel_id = assignment["channel_id"]
    
    # 设置通道数为最大通道号+1（如果没有分配则为1）
    radar_info[radar_id] = max_channel_id + 1 if max_channel_id >= 0 else 1

target_info = {target_id: {} for target_id in bfsa_targets}
time_range = (0, len(bfsa_rho_history))


# 将分配历史转换为甘特图数据格式
def convert_to_gantt_data(history):
    gantt_data = []
    target_segments = {target_id: [] for target_id in bfsa_targets}

    for i, record in enumerate(history):
        timestamp = record["timestamp"]
        assignments = record["assignments"]

        for target_id, assignment in assignments.items():
            if target_id not in target_segments:
                continue

            segments = target_segments[target_id]
            
            if assignment is not None and assignment["radar_id"] is not None:
                radar_id = assignment["radar_id"]
                channel_id = assignment["channel_id"]
                
                if not segments or \
                   segments[-1]["radar_id"] != radar_id or \
                   segments[-1]["channel_id"] != channel_id or \
                   segments[-1]["end_time"] != timestamp:
                    segments.append({
                        "target_id": int(target_id),
                        "radar_id": radar_id,
                        "channel_id": channel_id,
                        "start_time": timestamp,
                        "end_time": timestamp + 1.0
                    })
                else:
                    segments[-1]["end_time"] = timestamp + 1.0

    for target_id, segments in target_segments.items():
        gantt_data.extend(segments)

    return gantt_data


bfsa_gantt_data = convert_to_gantt_data(bfsa_rho_history)
rule_gantt_data = convert_to_gantt_data(rule_based_history)


# 计算目标切换次数
def calculate_switches(gantt_data):
    # 按目标ID分组
    target_segments = {}
    for segment in gantt_data:
        target_id = segment["target_id"]
        if target_id not in target_segments:
            target_segments[target_id] = []
        target_segments[target_id].append(segment)

    # 计算每个目标的切换次数
    switches = {}
    for target_id, segments in target_segments.items():
        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x["start_time"])

        # 计算雷达切换次数
        switch_count = 0
        for i in range(1, len(sorted_segments)):
            if sorted_segments[i]["radar_id"] != sorted_segments[i - 1]["radar_id"]:
                switch_count += 1

        switches[target_id] = switch_count

    return switches


bfsa_switches = calculate_switches(bfsa_gantt_data)
rule_switches = calculate_switches(rule_gantt_data)


# 生成收敛曲线数据 (模拟数据)
def generate_convergence_data():
    # 这里使用模拟数据，实际应用中应该从算法运行过程中收集
    iterations = 20
    bfsa_values = [0.3]
    rule_values = [0.2]

    for i in range(1, iterations):
        bfsa_values.append(min(0.85, bfsa_values[-1] + 0.5 / (i + 2)))
        rule_values.append(min(0.75, rule_values[-1] + 0.4 / (i + 2)))

    return {
        "BFSA-Rho": bfsa_values,
        "Rule-Based": rule_values
    }


convergence_data = generate_convergence_data()


# 生成性能数据 (模拟多次运行的结果)
def generate_performance_data():
    # 模拟10次运行的结果
    runs = 10
    np.random.seed(42)  # 设置随机种子以确保可重复性

    bfsa_perf = np.random.normal(0.82, 0.02, runs)
    rule_perf = np.random.normal(0.74, 0.03, runs)

    return {
        "BFSA-Rho": bfsa_perf.tolist(),
        "Rule-Based": rule_perf.tolist()
    }


performance_data = generate_performance_data()

# 绘制并保存图表
# 1. BFSA-Rho 雷达甘特图
plotter.plot_radar_gantt(
    bfsa_gantt_data,
    time_range,
    radar_info,
    target_info,
    save_path=os.path.join(output_dir, "bfsa_rho_radar_gantt.png")
)


# 2. Rule-Based 雷达甘特图
plotter.plot_radar_gantt(
    rule_gantt_data,
    time_range,
    radar_info,
    target_info,
    save_path=os.path.join(output_dir, "rule_based_radar_gantt.png")
)


# 3. BFSA-Rho 目标甘特图
plotter.plot_target_gantt(
    bfsa_gantt_data,
    time_range,
    target_info,
    radar_info,
    save_path=os.path.join(output_dir, "bfsa_rho_target_gantt.png")
)


# 4. Rule-Based 目标甘特图
plotter.plot_target_gantt(
    rule_gantt_data,
    time_range,
    target_info,
    radar_info,
    save_path=os.path.join(output_dir, "rule_based_target_gantt.png")
)


# 5. 目标切换频次对比图
# 将两个算法的切换数据合并为一个格式
switching_data = {
    "BFSA-Rho": bfsa_gantt_data,
    # "Rule-Based": rule_gantt_data
}

for algo_name, data in switching_data.items():
    plotter.plot_target_switching(
        data,
        target_info,
        algorithm_name=algo_name,
        save_path=os.path.join(output_dir, f"{algo_name.lower().replace('-', '_')}_target_switching.png")
    )

print(f"所有图表已保存到 {output_dir}")
