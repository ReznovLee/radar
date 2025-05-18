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

"""import matplotlib
matplotlib.rc("font",family='TimeNewRoman')"""

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
    radars_dict = {}
    for _, row in df.iterrows():
        radar_id = int(row['id'])
        radars_dict[radar_id] = {
            'x': float(row['x']),
            'y': float(row['y']),
            'z': float(row['z']),
            'radius': float(row['radius']),
            'number_channel': int(row['number_channel'])
        }
    return radars_dict


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


def build_radar_network(radars_dict):
    """ Build radar network

    Using Radar class builds radar network

    :param radars_dict: radar dict from radar_dict
    :return radar_network: radar network
    """
    radars = {}
    for radar_id, info in radars_dict.items():
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
                   sim_total_time,
                   output_json_path):
    assignment_history = []
    radar_channel_state = {rid: [None] * radar_network.radars[rid].num_channels for rid in radar_network.radars}
    algorithm = algorithm_class(radar_network)
    for t in range(sim_total_time + 1):
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


def load_assignment_history(file_path):
    """加载分配历史数据"""
    with open(file_path, 'r') as f:
        return json.load(f)


def main():
    if platform.system() == 'Windows':
        config_path = os.path.join("data\\config\\param_config.yaml")  # windows
    else:
        config_path = os.path.join('data', 'config', 'param_config.yaml')  # linux
    radar_csv_path = os.path.join('..', 'output', 'scenario-2025-05-18', '5-radar.csv')
    target_csv_path = os.path.join('..', 'output', 'scenario-2025-05-18', '50-targets.csv')
    outputs_dir = os.path.join('..', 'output', 'scenario-2025-05-18')
    os.makedirs(outputs_dir, exist_ok=True)

    config = load_yaml_config(config_path)
    sim_total_time = int(config['simulation']['total_time'])
    radars_dict = load_radar_csv(radar_csv_path)
    targets_by_timestep = load_targets_csv(target_csv_path) # targets_by_timestep 在这里加载
    radar_network = build_radar_network(radars_dict)

    # BFSA-RHO algorithm
    bfsa_rho_output = os.path.join(outputs_dir, 'bfsa_rho_assignment_history.json')
    run_simulation(
        algorithm_class=BFSARHO,
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        sim_total_time=sim_total_time,
        output_json_path=bfsa_rho_output
    )

    # Rule-Based algorithm
    rule_based_output = os.path.join(outputs_dir, 'rule_based_assignment_history.json')
    run_simulation(
        algorithm_class=RuleBasedScheduler,
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        sim_total_time=sim_total_time,
        output_json_path=rule_based_output
    )

    # LNS algorithm
    lns_output = os.path.join(outputs_dir, 'lns_assignment_history.json')
    run_simulation(
        algorithm_class=LNS, # 假设LNS类也已定义和导入
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        sim_total_time=sim_total_time,
        output_json_path=lns_output
    )
    
    # 根据 traceback，visualize_results 调用在 main 函数的第 211 行附近
    # 将 targets_by_timestep 传递给 visualize_results
    visualize_results(outputs_dir, radars_dict, targets_by_timestep, sim_total_time)

    # 根据 traceback, main() 函数的返回
    return outputs_dir, radars_dict, sim_total_time


def visualize_results(outputs_dir, radars_dict, targets_by_timestep, sim_total_time):
    """可视化仿真结果"""
    plotter = RadarPlotter()
    
    # 加载分配历史
    bfsa_history = load_assignment_history(os.path.join(outputs_dir, 'bfsa_rho_assignment_history.json'))
    rule_history = load_assignment_history(os.path.join(outputs_dir, 'rule_based_assignment_history.json'))
    
    # 绘制雷达甘特图
    plotter.plot_radar_gantt(
        bfsa_history,
        (0, sim_total_time),
        {rid: info['number_channel'] for rid, info in radars_dict.items()},
        {target['id']: target for timestep_targets in targets_by_timestep.values() 
         for target in timestep_targets},
        os.path.join(outputs_dir, 'bfsa_rho_radar_gantt.png')
    )
    
    plotter.plot_radar_gantt(
        rule_history,
        (0, sim_total_time),
        {rid: info['number_channel'] for rid, info in radars_dict.items()},
        {target['id']: target for timestep_targets in targets_by_timestep.values() 
         for target in timestep_targets},
        os.path.join(outputs_dir, 'rule_based_radar_gantt.png')
    )
    
    # 绘制目标甘特图
    plotter.plot_target_gantt(
        bfsa_history,
        (0, sim_total_time),
        {target['id']: target for timestep_targets in targets_by_timestep.values() 
         for target in timestep_targets},
        radars_dict,
        os.path.join(outputs_dir, 'bfsa_rho_target_gantt.png')
    )
    
    plotter.plot_target_gantt(
        rule_history,
        (0, sim_total_time),
        {target['id']: target for timestep_targets in targets_by_timestep.values() 
         for target in timestep_targets},
        radars_dict,
        os.path.join(outputs_dir, 'rule_based_target_gantt.png')
    )
    
    # 绘制所有算法的分配历史
    bfsa_rho_history_path = os.path.join(outputs_dir, 'bfsa_rho_assignment_history.json')
    rule_based_history_path = os.path.join(outputs_dir, 'rule_based_assignment_history.json')
    lns_history_path = os.path.join(outputs_dir, 'lns_assignment_history.json')

    assignment_histories = {}
    if os.path.exists(bfsa_rho_history_path):
        assignment_histories["BFSA-RHO"] = load_assignment_history(bfsa_rho_history_path)
    if os.path.exists(rule_based_history_path):
        assignment_histories["Rule-Based"] = load_assignment_history(rule_based_history_path)
    if os.path.exists(lns_history_path):
        assignment_histories["LNS"] = load_assignment_history(lns_history_path)
    
    if not assignment_histories:
        print("警告: 未找到任何分配历史文件。将跳过优先级满足度绘图。")
    else:
        # 为每个算法绘制优先级满足度图
        target_info = {}
        for timestep, targets in targets_by_timestep.items():
            for target in targets:
                target_id = str(target['id'])  # 转换为字符串以匹配JSON中的格式
                if target_id not in target_info:
                    target_info[target_id] = {
                        'priority': target['priority'],
                        'type': target['target_type']
                    }

    print(f"可视化结果已保存到目录: {outputs_dir}")

    # 删除未定义的函数调用
    """
    # 3. 优先级满足率柱状图
    plot_priority_satisfaction(
        assignment_histories,
        targets_by_timestep,
        save_path=os.path.join(vis_output_dir, 'priority_satisfaction.png')
    )
    
    # 4. 综合性能评分图
    plot_overall_performance(
        assignment_histories,
        targets_by_timestep,
        radar_info,
        save_path=os.path.join(vis_output_dir, 'overall_performance')
    )
    """

# In main function invoke visualization function
if __name__ == "__main__":
    output_dir, radar_dict, total_time = main()
    # print(f"Simulation finished. Output directory: {output_dir}")
    # print(f"Total simulation time: {total_time}")
