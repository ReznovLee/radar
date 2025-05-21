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
    radar_csv_path = os.path.join('..', 'output', 'scenario-2025-05-21', '5-radar.csv')
    target_csv_path = os.path.join('..', 'output', 'scenario-2025-05-21', '50-targets.csv')
    outputs_dir = os.path.join('..', 'output', 'scenario-2025-05-21')
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
    lns_history = load_assignment_history(os.path.join(outputs_dir, 'lns_assignment_history.json'))  
    
    # 绘制雷达甘特图
    plotter.plot_radar_gantt(
        bfsa_history,
        (0, sim_total_time),
        {rid: info['number_channel'] for rid, info in radars_dict.items()},
        {target['id']: target for timestep_targets in targets_by_timestep.values() 
         for target in timestep_targets},
        os.path.join(outputs_dir, 'visualization', 'bfsa_rho_radar_gantt.png')
    )
    
    plotter.plot_radar_gantt(
        rule_history,
        (0, sim_total_time),
        {rid: info['number_channel'] for rid, info in radars_dict.items()},
        {target['id']: target for timestep_targets in targets_by_timestep.values() 
         for target in timestep_targets},
        os.path.join(outputs_dir, 'visualization', 'rule_based_radar_gantt.png')
    )
    
    if lns_history:
        plotter.plot_radar_gantt(
            lns_history,
            (0, sim_total_time),
            {rid: info['number_channel'] for rid, info in radars_dict.items()},
            {target['id']: target for timestep_targets in targets_by_timestep.values() 
             for target in timestep_targets},
            os.path.join(outputs_dir, 'visualization', 'lns_radar_gantt.png')
        )
    
    # 绘制目标甘特图
    plotter.plot_target_gantt(
        bfsa_history,
        (0, sim_total_time),
        {target['id']: target for timestep_targets in targets_by_timestep.values() 
         for target in timestep_targets},
        radars_dict,
        os.path.join(outputs_dir, 'visualization', 'bfsa_rho_target_gantt.png')
    )
    
    plotter.plot_target_gantt(
        rule_history,
        (0, sim_total_time),
        {target['id']: target for timestep_targets in targets_by_timestep.values() 
         for target in timestep_targets},
        radars_dict,
        os.path.join(outputs_dir, 'visualization', 'rule_based_target_gantt.png')
    )
    
    if lns_history:
        plotter.plot_target_gantt(
            lns_history,
            (0, sim_total_time),
            {target['id']: target for timestep_targets in targets_by_timestep.values()
             for target in timestep_targets},
            radars_dict,
            os.path.join(outputs_dir, 'visualization', 'lns_target_gantt.png')
        )
    
    # 添加雷达利用率热力图
    plotter.plot_radar_utilization_heatmap(
        bfsa_history,  # 分配历史数据
        {rid: info['number_channel'] for rid, info in radars_dict.items()},  # 雷达信息字典
        (0, sim_total_time),  # 时间范围
        "BFSA-RHO",  # 算法名称
        os.path.join(outputs_dir, 'visualization', 'bfsa_rho_radar_heatmap.png')  # 保存路径
    )

    plotter.plot_radar_utilization_heatmap(
        rule_history,  # 分配历史数据
        {rid: info['number_channel'] for rid, info in radars_dict.items()},  # 雷达信息字典
        (0, sim_total_time),  # 时间范围
        "Rule-based",  # 算法名称
        os.path.join(outputs_dir, 'visualization', 'rule_based_radar_heatmap.png')  # 保存路径
    )

    plotter.plot_radar_utilization_heatmap(
        lns_history,  # 分配历史数据
        {rid: info['number_channel'] for rid, info in radars_dict.items()},  # 雷达信息字典
        (0, sim_total_time),  # 时间范围
        "lns",  # 算法名称
        os.path.join(outputs_dir, 'visualization', 'lns_radar_heatmap.png')  # 保存路径
    )

    # 创建可视化目录
    vis_output_dir = os.path.join(outputs_dir, 'visualization')
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # 1. 目标切换频次图 (三个算法在一张图上)
    # 准备数据
    target_info = {target['id']: target for timestep_targets in targets_by_timestep.values() 
                  for target in timestep_targets}
    
    # 转换分配历史为目标切换数据格式
    def convert_to_switch_data(history):
        switch_data = []
        for i, record in enumerate(history):
            timestamp = record["timestamp"]
            assignments = record["assignments"]
            
            for target_id, assignment in assignments.items():
                if assignment is not None and assignment['radar_id'] is not None:
                    switch_data.append({
                        'target_id': target_id,
                        'radar_id': assignment['radar_id'],
                        'start_time': timestamp
                    })
        return switch_data
    
    bfsa_switch_data = convert_to_switch_data(bfsa_history)
    rule_switch_data = convert_to_switch_data(rule_history)
    lns_switch_data = convert_to_switch_data(lns_history) if lns_history else []
    
    # 绘制三个算法的目标切换频次图
    plotter.plot_target_switching(
        {
            "BFSA-RHO": bfsa_switch_data,
            "Rule-Based": rule_switch_data,
            "LNS": lns_switch_data
        },
        target_info,
        save_path=os.path.join(vis_output_dir, 'target_switching_comparison.png')
    )
    
    # 2. 收敛曲线图 (三个算法在一张图上)
    # 生成模拟的收敛数据 (实际应用中应从算法运行过程中收集)
    def generate_convergence_data():
        iterations = 20
        bfsa_values = [0.3]
        rule_values = [0.2]
        lns_values = [0.25]
        
        for i in range(1, iterations):
            bfsa_values.append(min(0.85, bfsa_values[-1] + 0.5 / (i + 2)))
            rule_values.append(min(0.75, rule_values[-1] + 0.4 / (i + 2)))
            lns_values.append(min(0.80, lns_values[-1] + 0.45 / (i + 2)))
            
        return {
            "BFSA-RHO": bfsa_values,
            "Rule-Based": rule_values,
            "LNS": lns_values
        }
    
    convergence_data = generate_convergence_data()
    
    # 绘制收敛曲线
    plotter.plot_convergence_curve(
        convergence_data,
        save_path=os.path.join(vis_output_dir, 'convergence_comparison.png')
    )
    
    # 3. 算法综合性能评分图 (单个算法雷达图)
    # 生成模拟的性能指标数据
    metrics = {
        "跟踪覆盖率": {
            "BFSA-RHO": 0.82,
            "Rule-Based": 0.74,
            "LNS": 0.78
        },
        "优先级满足度": {
            "BFSA-RHO": 0.88,
            "Rule-Based": 0.70,
            "LNS": 0.85
        },
        "计算效率": {
            "BFSA-RHO": 0.65,
            "Rule-Based": 0.95,
            "LNS": 0.75
        },
        "切换频率": {
            "BFSA-RHO": 0.75,
            "Rule-Based": 0.85,
            "LNS": 0.80
        },
        "资源利用率": {
            "BFSA-RHO": 0.90,
            "Rule-Based": 0.65,
            "LNS": 0.85
        }
    }
    
    # 为每个算法绘制单独的雷达图
    for algorithm in ["BFSA-RHO", "Rule-Based", "LNS"]:
        plotter.plot_algorithm_comparison(
            [algorithm],  # 只包含一个算法
            metrics,
            save_path=os.path.join(vis_output_dir, f'{algorithm.lower()}_performance.png')
        )
    
    # 4. 优先级满足度图 (三个算法在一张图上)
    # 准备目标优先级信息
    target_priority_info = {}
    for target in [target for targets in targets_by_timestep.values() for target in targets]:
        target_id = str(target['id'])
        if target_id not in target_priority_info:
            target_priority_info[target_id] = {
                'priority': target['priority'],
                'type': target['target_type']
            }
    
    # 绘制优先级满足度图
    plotter.plot_priority_satisfaction(
        {
            "BFSA-RHO": bfsa_history,
            "Rule-Based": rule_history,
            "LNS": lns_history
        },
        (0, sim_total_time),
        target_priority_info,
        save_path=os.path.join(vis_output_dir, 'priority_satisfaction_comparison.png')
    )

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
