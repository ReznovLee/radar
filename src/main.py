# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : main.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:23
"""

import os
import yaml
import json
import numpy as np
import pandas as pd
import platform
from matplotlib import pyplot as plt

from core.models.radar_model import Radar, RadarNetwork
from core.algorithm.bfsa_rho import BFSARHO
from core.algorithm.rule_based import RuleBasedScheduler
from core.algorithm.LNS import LNS
from src.visualization.plotter import RadarPlotter


def load_yaml_config(config_path):
    """ Load the YAML config file

    Load parameters from the YAML config file

    :param config_path: path to the YAML config file
    :return config: loaded YAML config file
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_radar_csv(radar_csv_path):
    """ Load radar csv file

    Load radar csv file

    :param radar_csv_path: path to radar csv file
    :return radar: loaded radar csv file
    """
    df = pd.read_csv(radar_csv_path)
    radar_dict = {}
    for _, row in df.iterrows():
        radar_id = int(row['id'])
        radar_dict[radar_id] = {
            'x': float(row['x']),
            'y': float(row['y']),
            'z': float(row['z']),
            'radius': float(row['radius']),
            'number_channel': int(row['number_channel'])
        }
    return radar_dict


def load_targets_csv(target_csv_path):
    """ Load targe csv file

    Load the target csv file with all targets info by timestep

    :param target_csv_path: path to the target csv file
    :return targets: loaded target csv file
    """
    df = pd.read_csv(target_csv_path)
    targets_by_timestep = {}
    for _, row in df.iterrows():
        timestep = int(float(row['timestep']))
        if timestep not in targets_by_timestep:
            targets_by_timestep[timestep] = []
        targets_by_timestep[timestep].append({
            'id': int(row['id']),
            'position': np.array([float(row['position_x']), float(row['position_y']), float(row['position_z'])]),
            'velocity': np.array([float(row['velocity_x']), float(row['velocity_y']), float(row['velocity_z'])]),
            'target_type': str(row['target_type']),
            'priority': int(row['priority'])
        })
    return targets_by_timestep


def build_radar_network(radar_dict):
    """ Build radar network

    Using Radar class builds radar network

    :param radar_dict: radar dict from radar_dict
    :return radar_network: radar network
    """
    radars = {}
    for radar_id, info in radar_dict.items():
        radar = Radar(
            radar_id=radar_id,
            radar_position=np.array([info['x'], info['y'], info['z']]),
            radar_radius=info['radius'],
            num_channels=info['number_channel']
        )
        radars[radar_id] = radar
    radar_network = RadarNetwork(radars)
    return radar_network


def run_simulation(algorithm_class,
                   radar_network,
                   targets_by_timestep,
                   total_time,
                   output_json_path):
    assignment_history = []
    radar_channel_state = {rid: [None] * radar_network.radars[rid].num_channels for rid in radar_network.radars}
    algorithm = algorithm_class(radar_network)
    for t in range(total_time + 1):
        targets = targets_by_timestep.get(t, [])
        observed_targets = []
        for target in targets:
            observed_targets.append({
                'id': target['id'],
                'position': target['position'],
                'velocity': target['velocity'],
                'target_type': target['target_type'],
                'priority': target['priority']
            })
        assignment_matrix = algorithm.solve(targets, observed_targets, t)
        assignments = {}
        for rid in radar_channel_state:
            radar_channel_state[rid] = [None] * radar_network.radars[rid].num_channels
        if assignment_matrix is not None:
            for i, target in enumerate(targets):
                row = assignment_matrix.getrow(i).toarray().ravel()
                assigned_radar = None
                assigned_channel = None
                if np.any(row > 0):
                    assigned_radar = radar_network.radar_ids[np.argmax(row)]
                    channels = radar_channel_state[assigned_radar]
                    for ch_idx in range(len(channels)):
                        if channels[ch_idx] is None:
                            assigned_channel = ch_idx
                            channels[ch_idx] = target['id']
                            break
                    if assigned_channel is None:
                        assigned_radar = None
                assignments[str(target['id'])] = {
                    "radar_id": assigned_radar,
                    "channel_id": assigned_channel
                }
        assignment_history.append({
            'timestamp': float(t),
            'assignments': assignments
        })
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(assignment_history, f, indent=2)


def main():
    if platform.system() == 'Windows':
        config_path = os.path.join("data\\config\\param_config.yaml")  # windows
        data_dir = "data"
    else:
        config_path = os.path.join('data', 'config', 'param_config.yaml')  # linux
        data_dir = "data"
    radar_csv_path = os.path.join('..', 'output', 'scenario-2025-05-13', '10-radar.csv')
    target_csv_path = os.path.join('..', 'output', 'scenario-2025-05-13', '100-targets.csv')
    output_dir = os.path.join('..', 'output', 'scenario-2025-05-15')
    os.makedirs(output_dir, exist_ok=True)

    config = load_yaml_config(config_path)
    total_time = int(config['simulation']['total_time'])
    radar_dict = load_radar_csv(radar_csv_path)
    targets_by_timestep = load_targets_csv(target_csv_path)
    radar_network = build_radar_network(radar_dict)

    # BFSA-RHO algorithm
    bfsa_rho_output = os.path.join(output_dir, 'bfsa_rho_assignment_history.json')
    run_simulation(
        algorithm_class=BFSARHO,
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        total_time=total_time,
        output_json_path=bfsa_rho_output
    )

    # Rule-Based algorithm
    rule_based_output = os.path.join(output_dir, 'rule_based_assignment_history.json')
    run_simulation(
        algorithm_class=RuleBasedScheduler,
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        total_time=total_time,
        output_json_path=rule_based_output
    )
    
    # LNS algorithm
    lns_output = os.path.join(output_dir, 'lns_assignment_history.json')
    run_simulation(
        algorithm_class=LNS,
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        total_time=total_time,
        output_json_path=lns_output
    )

    # 可视化部分 - 使用真实分配历史数据
    visualize_results(output_dir, radar_dict, targets_by_timestep, total_time)
    
    # 返回所需的三个值
    return output_dir, radar_dict, total_time


def visualize_results(output_dir, radar_dict, targets_by_timestep, total_time):
    """使用真实分配历史数据进行可视化"""
    # 初始化绘图器
    plotter = RadarPlotter(figsize=(14, 8))
    
    # 读取分配历史数据
    bfsa_rho_file = os.path.join(output_dir, 'bfsa_rho_assignment_history.json')
    rule_based_file = os.path.join(output_dir, 'rule_based_assignment_history.json')
    lns_file = os.path.join(output_dir, 'lns_assignment_history.json')
    
    # 创建可视化输出目录
    vis_output_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # 读取分配历史数据
    with open(bfsa_rho_file, 'r') as f:
        bfsa_rho_history = json.load(f)
    with open(rule_based_file, 'r') as f:
        rule_based_history = json.load(f)
    with open(lns_file, 'r') as f:
        lns_history = json.load(f)
    
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
    lns_radars, lns_targets = extract_info(lns_history)
    
    # 合并雷达和目标信息
    all_radars = bfsa_radars.union(rule_radars).union(lns_radars)
    all_targets = bfsa_targets.union(rule_targets).union(lns_targets)
    
    # 构建雷达信息字典
    radar_info = {}
    for radar_id in all_radars:
        radar_info[radar_id] = radar_dict[radar_id]['number_channel']
    
    # 构建目标信息字典
    target_info = {target_id: {} for target_id in all_targets}
    
    # 设置时间范围
    time_range = (0, total_time)
    
    # 将分配历史转换为甘特图数据格式
    def convert_to_gantt_data(history):
        gantt_data = []
        target_segments = {target_id: [] for target_id in all_targets}
        
        for record in history:
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
    lns_gantt_data = convert_to_gantt_data(lns_history)
    
    # 绘制雷达甘特图 - 修复：直接使用原始分配历史数据，而不是转换后的甘特图数据
    plotter.plot_radar_gantt(
        bfsa_rho_history,  # 使用原始分配历史数据
        time_range,
        radar_info,
        target_info,
        save_path=os.path.join(vis_output_dir, 'bfsa_rho_radar_gantt.png')
    )
    
    plotter.plot_radar_gantt(
        rule_based_history,  # 使用原始分配历史数据
        time_range,
        radar_info,
        target_info,
        save_path=os.path.join(vis_output_dir, 'rule_based_radar_gantt.png')
    )
    
    plotter.plot_radar_gantt(
        lns_history,  # 使用原始分配历史数据
        time_range,
        radar_info,
        target_info,
        save_path=os.path.join(vis_output_dir, 'lns_radar_gantt.png')
    )
    
    # 绘制目标甘特图 - 这里需要使用转换后的甘特图数据
    plotter.plot_target_gantt(
        bfsa_gantt_data,
        time_range,
        target_info,
        radar_info,
        save_path=os.path.join(vis_output_dir, 'bfsa_rho_target_gantt.png')
    )
    
    plotter.plot_target_gantt(
        rule_gantt_data,
        time_range,
        target_info,
        radar_info,
        save_path=os.path.join(vis_output_dir, 'rule_based_target_gantt.png')
    )
    
    plotter.plot_target_gantt(
        lns_gantt_data,
        time_range,
        target_info,
        radar_info,
        save_path=os.path.join(vis_output_dir, 'lns_target_gantt.png')
    )
    
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
    lns_switches = calculate_switches(lns_gantt_data)
    
    # 绘制切换次数比较图
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    target_ids = sorted(list(all_targets), key=int)
    bfsa_switch_values = [bfsa_switches.get(tid, 0) for tid in target_ids]
    rule_switch_values = [rule_switches.get(tid, 0) for tid in target_ids]
    lns_switch_values = [lns_switches.get(tid, 0) for tid in target_ids]
    
    x = np.arange(len(target_ids))
    width = 0.25
    
    # 绘制柱状图
    plt.bar(x - width, bfsa_switch_values, width, label='BFSA-Rho')
    plt.bar(x, rule_switch_values, width, label='Rule-Based')
    plt.bar(x + width, lns_switch_values, width, label='LNS')
    
    plt.xlabel('目标ID')
    plt.ylabel('雷达切换次数')
    plt.title('不同算法的目标雷达切换次数比较')
    plt.xticks(x, [f'T{tid}' for tid in target_ids], rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(vis_output_dir, 'radar_switches_comparison.png'), dpi=300)
    
    # 绘制分配率随时间变化图
    plt.figure(figsize=(12, 6))
    
    # 计算每个时间步的分配率
    def calculate_assignment_rates(history):
        rates = []
        for record in history:
            assignments = record["assignments"]
            total = len(assignments)
            assigned = sum(1 for a in assignments.values() if a is not None and a["radar_id"] is not None)
            rates.append(assigned / total if total > 0 else 0)
        return rates
    
    bfsa_rates = calculate_assignment_rates(bfsa_rho_history)
    rule_rates = calculate_assignment_rates(rule_based_history)
    lns_rates = calculate_assignment_rates(lns_history)
    
    time_steps = list(range(len(bfsa_rates)))
    
    plt.plot(time_steps, bfsa_rates, 'b-', label='BFSA-Rho')
    plt.plot(time_steps, rule_rates, 'r-', label='Rule-Based')
    plt.plot(time_steps, lns_rates, 'g-', label='LNS')
    
    plt.xlabel('时间步')
    plt.ylabel('分配率')
    plt.title('不同算法的目标分配率随时间变化')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(vis_output_dir, 'assignment_rates_comparison.png'), dpi=300)
    
    print(f"可视化结果已保存到: {vis_output_dir}")

# 在main函数中调用可视化函数
if __name__ == "__main__":
    output_dir, radar_dict, total_time = main()
    
    # 运行模拟
    visualize_results(output_dir)
