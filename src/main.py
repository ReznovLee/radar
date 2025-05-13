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

from core.models.radar_model import Radar, RadarNetwork
from core.algorithm.bfsa_rho import BFSARHO
from core.algorithm.rule_based import RuleBasedScheduler


def load_yaml_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_radar_csv(radar_csv_path):
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


def run_simulation(algorithm_name, algorithm_class, radar_network, targets_by_timestep, total_time, output_json_path):
    assignment_history = []
    # 初始化每部雷达的通道占用情况
    radar_channel_state = {rid: [None] * radar_network.radars[rid].num_channels for rid in radar_network.radars}
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
        if t == 0:
            algorithm = algorithm_class(radar_network)
        assignment_matrix = algorithm.solve(targets, observed_targets, t)
        assignments = {}
        channel_assignments = {}
        # 重置通道占用
        for rid in radar_channel_state:
            radar_channel_state[rid] = [None] * radar_network.radars[rid].num_channels
        if assignment_matrix is not None:
            for i, target in enumerate(targets):
                row = assignment_matrix.getrow(i).toarray().ravel()
                assigned_radar = None
                assigned_channel = None
                if np.any(row > 0):
                    assigned_radar = radar_network.radar_ids[np.argmax(row)]
                    # 分配空闲通道（简单轮询）
                    channels = radar_channel_state[assigned_radar]
                    for ch_idx in range(len(channels)):
                        if channels[ch_idx] is None:
                            assigned_channel = ch_idx
                            channels[ch_idx] = target['id']
                            break
                    if assigned_channel is None:
                        # 所有通道被占用，分配失败
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
    config_path = os.path.join("data\\config\\param_config.yaml")  # 'data', 'config', 'param_config.yaml'
    radar_csv_path = os.path.join('..', 'output', 'scenario-2025-05-13', '10-radar.csv')
    target_csv_path = os.path.join('..', 'output', 'scenario-2025-05-13', '100-targets.csv')
    output_dir = os.path.join('..', 'output', 'scenario-2025-05-13')
    os.makedirs(output_dir, exist_ok=True)

    config = load_yaml_config(config_path)
    total_time = int(config['simulation']['total_time'])
    radar_dict = load_radar_csv(radar_csv_path)
    targets_by_timestep = load_targets_csv(target_csv_path)
    radar_network = build_radar_network(radar_dict)

    # BFSA-RHO algorithm
    bfsa_rho_output = os.path.join(output_dir, 'bfsa_rho_assignment_history.json')
    run_simulation(
        algorithm_name='BFSA-Rho',
        algorithm_class=BFSARHO,
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        total_time=total_time,
        output_json_path=bfsa_rho_output
    )

    # Rule-Based algorithm
    """
    rule_based_output = os.path.join(output_dir, 'rule_based_assignment_history.json')
    run_simulation(
        algorithm_name='Rule-Based',
        algorithm_class=RuleBasedScheduler,
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        total_time=total_time,
        output_json_path=rule_based_output
    )
    """


if __name__ == '__main__':
    main()
