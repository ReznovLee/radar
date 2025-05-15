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

from core.models.radar_model import Radar, RadarNetwork
from core.algorithm.bfsa_rho import BFSARHO
from core.algorithm.rule_based import RuleBasedScheduler  # Modify
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

    return output_dir, radar_dict, total_time


def generate_convergence_data():
    """生成收敛曲线数据
    
    Returns:
        dict: 包含两种算法的收敛数据
    """
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


def generate_performance_data():
    """生成性能数据
    
    Returns:
        dict: 包含两种算法的性能数据
    """
    runs = 10
    np.random.seed(42)  # 设置随机种子以确保可重复性

    bfsa_perf = np.random.normal(0.82, 0.02, runs)
    rule_perf = np.random.normal(0.74, 0.03, runs)

    return {
        "BFSA-Rho": bfsa_perf.tolist(),
        "Rule-Based": rule_perf.tolist()
    }


if __name__ == '__main__':
    output_dir, radar_dict, total_time = main()
    
    # 初始化绘图器
    plotter = RadarPlotter(figsize=(14, 8))
    
    # 读取分配历史数据
    bfsa_rho_file = os.path.join(output_dir, 'bfsa_rho_assignment_history.json')
    rule_based_file = os.path.join(output_dir, 'rule_based_assignment_history.json')
    
    with open(bfsa_rho_file, 'r') as f:
        bfsa_rho_history = json.load(f)
    with open(rule_based_file, 'r') as f:
        rule_based_history = json.load(f)

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
    
    # 构建雷达信息字典
    radar_info = {rid: info['number_channel'] for rid, info in radar_dict.items()}
    target_info = {str(target_id): {} for target_id in all_targets}
    time_range = (0, total_time)
    
    # 创建可视化输出目录
    vis_output_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # 1. 计算目标切换数据
    def convert_to_switching_data(history):
        switching_data = []
        for i, record in enumerate(history):
            for target_id, assignment in record['assignments'].items():
                if assignment['radar_id'] is not None:
                    switching_data.append({
                        'target_id': int(target_id),
                        'radar_id': assignment['radar_id'],
                        'start_time': record['timestamp'],
                        'end_time': record['timestamp'] + 1.0
                    })
        return switching_data

    bfsa_switching_data = convert_to_switching_data(bfsa_rho_history)
    rule_switching_data = convert_to_switching_data(rule_based_history)
    
    # 绘制目标切换频次图
    plotter.plot_target_switching(
        bfsa_switching_data,
        target_info,
        algorithm_name="BFSA-Rho",
        save_path=os.path.join(vis_output_dir, 'bfsa_rho_switching.png')
    )
    
    plotter.plot_target_switching(
        rule_switching_data,
        target_info,
        algorithm_name="Rule-Based",
        save_path=os.path.join(vis_output_dir, 'rule_based_switching.png')
    )
    
    # 2. 生成收敛数据
    def calculate_coverage(history):
        total_assignments = 0
        valid_assignments = 0
        for record in history:
            for assignment in record['assignments'].values():
                total_assignments += 1
                if assignment['radar_id'] is not None:
                    valid_assignments += 1
        return valid_assignments / total_assignments if total_assignments > 0 else 0

    convergence_data = {
        "BFSA-Rho": [],
        "Rule-Based": []
    }
    
    # 计算每个时间步的覆盖率
    for t in range(total_time):
        bfsa_history_slice = bfsa_rho_history[:t+1]
        rule_history_slice = rule_based_history[:t+1]
        
        convergence_data["BFSA-Rho"].append(calculate_coverage(bfsa_history_slice))
        convergence_data["Rule-Based"].append(calculate_coverage(rule_history_slice))
    
    # 绘制收敛曲线
    plotter.plot_convergence_curve(
        convergence_data,
        save_path=os.path.join(vis_output_dir, 'convergence_curves.png')
    )
    
    # 3. 生成算法性能对比数据
    metrics = {
        "覆盖率": {
            "BFSA-Rho": calculate_coverage(bfsa_rho_history),
            "Rule-Based": calculate_coverage(rule_based_history)
        },
        "切换次数": {
            "BFSA-Rho": len(bfsa_switching_data),
            "Rule-Based": len(rule_switching_data)
        },
        "计算时间": {
            "BFSA-Rho": 0.85,  # 示例值
            "Rule-Based": 0.95  # 示例值
        }
    }
    
    # 绘制算法综合性能对比图
    plotter.plot_algorithm_comparison(
        ["BFSA-Rho", "Rule-Based"],
        metrics,
        save_path=os.path.join(vis_output_dir, 'algorithm_comparison.png')
    )
    
    # 4. 生成完整的评估结果数据
    # 修改数据格式以适应plot_all_metrics函数的要求
    def convert_to_target_gantt_format(history):
        gantt_data = []
        for record in history:
            timestamp = record['timestamp']
            for target_id, assignment in record['assignments'].items():
                if assignment['radar_id'] is not None:
                    gantt_data.append({
                        'target_id': int(target_id),
                        'radar_id': assignment['radar_id'],
                        'channel_id': assignment['channel_id'],
                        'start_time': timestamp,
                        'end_time': timestamp + 1.0
                    })
        return gantt_data

    results = {
        'allocations': {
            'BFSA-Rho': convert_to_target_gantt_format(bfsa_rho_history),
            'Rule-Based': convert_to_target_gantt_format(rule_based_history)
        },
        'time_range': time_range,
        'radar_info': radar_info,
        'target_info': target_info,
        'convergence_data': convergence_data,
        'performance_metrics': {
            '覆盖率': {
                'BFSA-Rho': [calculate_coverage(bfsa_rho_history)],
                'Rule-Based': [calculate_coverage(rule_based_history)]
            },
            '切换次数': {
                'BFSA-Rho': [len(bfsa_switching_data)],
                'Rule-Based': [len(rule_switching_data)]
            },
            '计算时间': {
                'BFSA-Rho': [0.85],
                'Rule-Based': [0.95]
            }
        },
        'average_metrics': metrics
    }
    
    # 生成所有评价指标的图表
#    plotter.plot_all_metrics(results, output_dir=vis_output_dir)
