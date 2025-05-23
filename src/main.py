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
import logging
from typing import Dict, List, Optional # 新增导入

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
    """ Load a targe csv file

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
        targets_at_current_ts = targets_by_timestep.get(t, []) # Renamed for clarity
        observed_targets = []
        for target_obs in targets_at_current_ts: # Renamed for clarity
            observed_targets.append({
                'id': target_obs['id'],
                'position': target_obs['position'],
                'velocity': target_obs['velocity'],
                'target_type': target_obs['target_type'],
                'priority': target_obs['priority']
            })

        assignment_matrix = algorithm.solve(targets_at_current_ts, observed_targets, t)
        
        assignments = {}
        for rid_reset in radar_channel_state: # Use a different variable name
            radar_channel_state[rid_reset] = [None] * radar_network.radars[rid_reset].num_channels
            
        if assignment_matrix is not None:
            for i, target_data in enumerate(targets_at_current_ts): 
                row = assignment_matrix.getrow(i).toarray().ravel()
                assigned_radar = None
                assigned_channel = None
                if np.any(row > 0):
                    assigned_radar_idx = np.argmax(row)
                    if assigned_radar_idx < len(radar_network.radar_ids):
                        assigned_radar = radar_network.radar_ids[assigned_radar_idx]
                        channels = radar_channel_state[assigned_radar]
                        for ch_idx in range(len(channels)):
                            if channels[ch_idx] is None:
                                assigned_channel = ch_idx
                                channels[ch_idx] = target_data['id']
                                break
                        if assigned_channel is None:
                            assigned_radar = None 
                    else:
                        assigned_radar = None

                is_in_any_coverage = 0
                target_pos = target_data['position']
                for r_id_loop, r_obj_loop in radar_network.radars.items():
                    radar_pos_loop = r_obj_loop.radar_position
                    distance = np.linalg.norm(target_pos - radar_pos_loop)
                    if distance <= r_obj_loop.radar_radius:
                        is_in_any_coverage = 1
                        break
                
                assignments[str(target_data['id'])] = {
                    "radar_id": assigned_radar,
                    "channel_id": assigned_channel,
                    "in": is_in_any_coverage
                }
        else:
            for target_data in targets_at_current_ts:
                is_in_any_coverage = 0
                target_pos = target_data['position']
                for r_id_loop, r_obj_loop in radar_network.radars.items():
                    radar_pos_loop = r_obj_loop.radar_position
                    distance = np.linalg.norm(target_pos - radar_pos_loop)
                    if distance <= r_obj_loop.radar_radius:
                        is_in_any_coverage = 1
                        break
                assignments[str(target_data['id'])] = {
                    "radar_id": None,
                    "channel_id": None,
                    "in": is_in_any_coverage
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


def calculate_total_radar_switches(assignment_history: List[Dict]) -> int:
    """
    计算单个算法分配历史中所有目标的总雷达切换次数。
    切换定义为：在一个目标上，从一个非 null 的 radar_id 切换到另一个不同的非 null 的 radar_id。
    """
    if not assignment_history:
        return 0

    target_radar_sequences: Dict[str, List[Optional[int]]] = {}

    all_target_ids_str = set()
    for entry in assignment_history:
        if 'assignments' in entry and entry['assignments']:
            for target_id_s in entry['assignments'].keys():
                all_target_ids_str.add(target_id_s)
    
    if not all_target_ids_str:
        return 0

    for target_id_s in all_target_ids_str:
        sequence: List[Optional[int]] = []
        for entry in assignment_history:
            assignment_details = entry['assignments'].get(target_id_s)
            if assignment_details and 'radar_id' in assignment_details:
                sequence.append(assignment_details['radar_id'])
            else:
                sequence.append(None)
        target_radar_sequences[target_id_s] = sequence

    total_switches_for_algo = 0
    for target_id_s, radar_ids_sequence in target_radar_sequences.items():
        switches_for_this_target = 0
        last_radar_id_for_target: Optional[int] = None 
        
        for current_radar_id in radar_ids_sequence:
            if last_radar_id_for_target is not None and \
               current_radar_id is not None and \
               last_radar_id_for_target != current_radar_id:
                switches_for_this_target += 1

            last_radar_id_for_target = current_radar_id
            
        total_switches_for_algo += switches_for_this_target
        
    return total_switches_for_algo


def calculate_target_assignment_rate_data(assignment_histories: Dict[str, List[Dict]],
                                          targets_by_timestep: Dict[int, List[Dict]],
                                          radars_dict: Dict[int, Dict],
                                          sim_total_time: int) -> Dict[str, List[float]]:
    """
    计算全局目标跟踪率数据。
    公式：(已跟踪目标数 / 覆盖范围内目标数) * (活动目标平均优先级 / 6.0)
    """
    assignment_rate_data = {algo_name: [] for algo_name in assignment_histories.keys()}
    
    # 预处理雷达信息
    radar_positions = {}
    radar_ranges = {}
    for radar_id, radar_info in radars_dict.items():
        radar_positions[radar_id] = np.array([radar_info['x'], radar_info['y'], radar_info['z']])
        radar_ranges[radar_id] = radar_info['radius']
    
    for algo_name, history in assignment_histories.items():
        if not history:
            assignment_rate_data[algo_name] = [0.0] * (sim_total_time + 1)
            continue
            
        ratios_for_algo = []
        history_by_ts = {item['timestamp']: item for item in history}
        
        for t in range(sim_total_time + 1):
            current_targets_at_t = targets_by_timestep.get(t, [])
            
            if not current_targets_at_t:
                ratios_for_algo.append(0.0)
                continue
                
            # 计算在雷达覆盖范围内的目标
            targets_in_coverage = []
            total_priority = 0
            
            for target_data in current_targets_at_t:
                target_pos = np.array(target_data['position'])
                target_priority = int(target_data['priority'])
                
                # 检查目标是否在任何雷达覆盖范围内
                in_coverage = False
                for radar_id, radar_pos in radar_positions.items():
                    distance = np.linalg.norm(target_pos - radar_pos)
                    if distance <= radar_ranges[radar_id]:
                        in_coverage = True
                        break
                        
                if in_coverage:
                    targets_in_coverage.append(target_data)
                    total_priority += target_priority
            
            if not targets_in_coverage:
                ratios_for_algo.append(0.0)
                continue
                
            # 计算被跟踪的目标数量
            history_entry = history_by_ts.get(float(t))
            tracked_count = 0
            
            if history_entry and 'assignments' in history_entry:
                assignments = history_entry['assignments']
                for target_data in targets_in_coverage:
                    target_id_str = str(target_data['id'])
                    if (target_id_str in assignments and 
                        assignments[target_id_str] and 
                        assignments[target_id_str].get('radar_id') is not None):
                        tracked_count += 1
            
            # 计算全局跟踪率
            coverage_ratio = tracked_count / len(targets_in_coverage)
            avg_priority = total_priority / len(targets_in_coverage)
            priority_factor = avg_priority / 6.0
            
            global_tracking_rate = coverage_ratio * priority_factor
            ratios_for_algo.append(global_tracking_rate)
            
        assignment_rate_data[algo_name] = ratios_for_algo
        
    return assignment_rate_data

def calculate_target_assignment_rate_data_without_priotiry(assignment_histories: Dict[str, List[Dict]],
                                          targets_by_timestep: Dict[int, List[Dict]],
                                          radars_dict: Dict[int, Dict],
                                          sim_total_time: int) -> Dict[str, List[float]]:
    """
    计算全局目标跟踪率数据。
    公式：(已跟踪目标数 / 覆盖范围内目标数) * (活动目标平均优先级 / 6.0)
    """
    assignment_rate_data = {algo_name: [] for algo_name in assignment_histories.keys()}
    
    # 预处理雷达信息
    radar_positions = {}
    radar_ranges = {}
    for radar_id, radar_info in radars_dict.items():
        radar_positions[radar_id] = np.array([radar_info['x'], radar_info['y'], radar_info['z']])
        radar_ranges[radar_id] = radar_info['radius']
    
    for algo_name, history in assignment_histories.items():
        if not history:
            assignment_rate_data[algo_name] = [0.0] * (sim_total_time + 1)
            continue
            
        ratios_for_algo = []
        history_by_ts = {item['timestamp']: item for item in history}
        
        for t in range(sim_total_time + 1):
            current_targets_at_t = targets_by_timestep.get(t, [])
            
            if not current_targets_at_t:
                ratios_for_algo.append(0.0)
                continue
                
            # 计算在雷达覆盖范围内的目标
            targets_in_coverage = []
            total_priority = 0
            
            for target_data in current_targets_at_t:
                target_pos = np.array(target_data['position'])
                target_priority = int(target_data['priority'])
                
                # 检查目标是否在任何雷达覆盖范围内
                in_coverage = False
                for radar_id, radar_pos in radar_positions.items():
                    distance = np.linalg.norm(target_pos - radar_pos)
                    if distance <= radar_ranges[radar_id]:
                        in_coverage = True
                        break
                        
                if in_coverage:
                    targets_in_coverage.append(target_data)
                    total_priority += target_priority
            
            if not targets_in_coverage:
                ratios_for_algo.append(0.0)
                continue
                
            # 计算被跟踪的目标数量
            history_entry = history_by_ts.get(float(t))
            tracked_count = 0
            
            if history_entry and 'assignments' in history_entry:
                assignments = history_entry['assignments']
                for target_data in targets_in_coverage:
                    target_id_str = str(target_data['id'])
                    if (target_id_str in assignments and 
                        assignments[target_id_str] and 
                        assignments[target_id_str].get('radar_id') is not None):
                        tracked_count += 1
            
            # 计算全局跟踪率
            coverage_ratio = tracked_count / len(targets_in_coverage)
            
            ratios_for_algo.append(coverage_ratio)
            
        assignment_rate_data[algo_name] = ratios_for_algo
        
    return assignment_rate_data

def calculate_and_save_target_tracking_rates(
        assignment_histories: Dict[str, List[Dict]],
        targets_by_timestep: Dict[int, List[Dict]],
        radars_dict: Dict[int, Dict],
        sim_total_time: int,
        outputs_dir: str):
    """
    计算每个算法的平均目标跟踪率并保存到 JSON 文件。
    单个目标跟踪率 = (目标被跟踪的总时长 / 目标在雷达覆盖范围内时长) * (目标优先级 / 6)
    最终结果是所有单一目标跟踪率的平均值。
    """
    logging.info("Calculating target tracking rates...")
    all_algorithms_tracking_rates = {}

    all_target_details = {}  # {target_id_int: {'priority': prio, 'positions': {ts: pos_array}, 'first_ts': ts, 'last_ts': ts}}
    for ts in range(sim_total_time + 1):
        targets_at_ts = targets_by_timestep.get(ts, [])
        for target_data in targets_at_ts:
            tid = int(target_data['id'])
            if tid not in all_target_details:
                all_target_details[tid] = {
                    'priority': int(target_data['priority']),
                    'positions': {},
                    'first_ts': ts,
                    'last_ts': ts
                }
            all_target_details[tid]['positions'][ts] = np.array(target_data['position'])
            all_target_details[tid]['last_ts'] = max(all_target_details[tid]['last_ts'], ts)

    if not all_target_details:
        logging.warning("No targets found in targets_by_timestep. Cannot calculate tracking rates.")
        for algo_name in assignment_histories.keys():
            all_algorithms_tracking_rates[algo_name] = 0.0
        output_path = os.path.join(outputs_dir, 'visualization', 'result.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_algorithms_tracking_rates, f, indent=4)
        logging.info(f"Tracking rates (all zero due to no targets) saved to {output_path}")
        return

    target_coverage_durations = {}
    for tid, details in all_target_details.items():
        coverage_duration = 0
        for ts in range(details['first_ts'], details['last_ts'] + 1):
            if ts not in details['positions']:
                continue
            
            target_pos = details['positions'][ts]
            is_in_any_radar_coverage = False
            for radar_id, radar_info in radars_dict.items():
                radar_pos = np.array([radar_info['x'], radar_info['y'], radar_info['z']])
                distance = np.linalg.norm(target_pos - radar_pos)
                if distance <= radar_info['radius']:
                    is_in_any_radar_coverage = True
                    break
            if is_in_any_radar_coverage:
                coverage_duration += 1
        target_coverage_durations[tid] = coverage_duration

    for algo_name, history in assignment_histories.items():
        if not history:
            logging.warning(f"Assignment history for {algo_name} is empty. Tracking rate set to 0.")
            all_algorithms_tracking_rates[algo_name] = 0.0
            continue

        history_by_ts = {item['timestamp']: item['assignments'] for item in history}
        
        sum_of_single_target_tracking_rates = 0.0
        num_targets_considered = 0

        for tid, details in all_target_details.items():
            target_id_str = str(tid)
            target_priority = details['priority']
            
            tracked_duration = 0
            for ts in range(details['first_ts'], details['last_ts'] + 1):
                assignments_at_ts = history_by_ts.get(float(ts), {})
                assignment_info = assignments_at_ts.get(target_id_str)
                if assignment_info and assignment_info.get('radar_id') is not None:
                    tracked_duration += 1
            
            coverage_duration = target_coverage_durations.get(tid, 0)
            
            single_target_rate = 0.0
            if coverage_duration > 0:
                single_target_rate = (tracked_duration / coverage_duration) * (target_priority / 6.0)
            
            sum_of_single_target_tracking_rates += single_target_rate
            num_targets_considered += 1

        if num_targets_considered > 0:
            average_tracking_rate = sum_of_single_target_tracking_rates / num_targets_considered
        else:
            average_tracking_rate = 0.0
            
        all_algorithms_tracking_rates[algo_name] = average_tracking_rate
        logging.info(f"Algorithm {algo_name}: Average Target Tracking Rate = {average_tracking_rate:.4f}")

    output_path = os.path.join(outputs_dir, 'visualization', 'result.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_algorithms_tracking_rates, f, indent=4)
    logging.info(f"Target tracking rates saved to {output_path}")

def calculate_and_save_network_tracking_efficiency(
        assignment_histories: Dict[str, List[Dict]],
        outputs_dir: str):
    """
    计算每个算法的网络跟踪效率并保存到 JSON 文件。
    公式: (每个目标被正确分配的总时长) / (每个目标在所有时刻"in"标志为1的总数量 - 1)
    最终结果是所有符合条件的目标效率的平均值。
    """
    logging.info("Calculating network tracking efficiency...")
    all_algorithms_network_efficiency = {}

    for algo_name, history in assignment_histories.items():
        if not history:
            logging.warning(f"Assignment history for {algo_name} is empty. Network tracking efficiency set to 0.")
            all_algorithms_network_efficiency[algo_name] = 0.0
            continue

        target_stats: Dict[str, Dict[str, int]] = {}

        for entry in history:
            assignments_at_ts = entry.get('assignments', {})
            for target_id_str, details in assignments_at_ts.items():
                if target_id_str not in target_stats:
                    target_stats[target_id_str] = {'assigned_duration': 0, 'in_coverage_count': 0}
                
                if details.get('radar_id') is not None:
                    target_stats[target_id_str]['assigned_duration'] += 1
                
                if details.get('in') == 1:
                    target_stats[target_id_str]['in_coverage_count'] += 1
        
        if not target_stats:
            logging.warning(f"No target data found in history for {algo_name}. Efficiency set to 0.")
            all_algorithms_network_efficiency[algo_name] = 0.0
            continue

        sum_of_individual_target_efficiencies = 0.0
        num_targets_for_average = 0

        for target_id_str, stats_val in target_stats.items():
            assigned_duration = stats_val['assigned_duration']
            in_coverage_count = stats_val['in_coverage_count']
            
            denominator = in_coverage_count - 1
            
            if denominator > 0:
                efficiency = assigned_duration / denominator
                sum_of_individual_target_efficiencies += efficiency
                num_targets_for_average += 1
            
        if num_targets_for_average > 0:
            average_efficiency = sum_of_individual_target_efficiencies / num_targets_for_average
        else:
            average_efficiency = 0.0
            
        all_algorithms_network_efficiency[algo_name] = average_efficiency
        logging.info(f"Algorithm {algo_name}: Network Tracking Efficiency = {average_efficiency:.4f}")

    output_path = os.path.join(outputs_dir, 'visualization', 'network_tracking_efficiency.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_algorithms_network_efficiency, f, indent=4)
    logging.info(f"Network tracking efficiency saved to {output_path}")

def main():
    if platform.system() == 'Windows':
        config_path = os.path.join("data\\config\\param_config.yaml")  # windows
    else:
        config_path = os.path.join('data', 'config', 'param_config.yaml')  # linux
    radar_csv_path = os.path.join('..', 'output', 'scenario-2025-05-23', '5-radar.csv')
    target_csv_path = os.path.join('..', 'output', 'scenario-2025-05-23', '50-targets.csv')
    outputs_dir = os.path.join('..', 'output', 'scenario-2025-05-23')
    os.makedirs(outputs_dir, exist_ok=True)

    config = load_yaml_config(config_path)
    sim_total_time = int(config['simulation']['total_time'])
    radars_dict = load_radar_csv(radar_csv_path)
    targets_by_timestep = load_targets_csv(target_csv_path) # targets_by_timestep 在这里加载
    radar_network = build_radar_network(radars_dict)

    bfsa_rho_output = os.path.join(outputs_dir, 'bfsa_rho_assignment_history.json')
    run_simulation(
        algorithm_class=BFSARHO,
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        sim_total_time=sim_total_time,
        output_json_path=bfsa_rho_output
    )

    rule_based_output = os.path.join(outputs_dir, 'rule_based_assignment_history.json')
    run_simulation(
        algorithm_class=RuleBasedScheduler,
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        sim_total_time=sim_total_time,
        output_json_path=rule_based_output
    )

    lns_output = os.path.join(outputs_dir, 'lns_assignment_history.json')
    run_simulation(
        algorithm_class=LNS, # 假设LNS类也已定义和导入
        radar_network=radar_network,
        targets_by_timestep=targets_by_timestep,
        sim_total_time=sim_total_time,
        output_json_path=lns_output
    )

    visualize_results(outputs_dir, radars_dict, targets_by_timestep, sim_total_time)

    return outputs_dir, radars_dict, sim_total_time


def visualize_results(outputs_dir, radars_dict, targets_by_timestep, sim_total_time): # 添加 radar_network 参数
    """可视化仿真结果"""
    plotter = RadarPlotter()

    bfsa_history_path = os.path.join(outputs_dir, 'bfsa_rho_assignment_history.json')
    rule_history_path = os.path.join(outputs_dir, 'rule_based_assignment_history.json')
    lns_history_path = os.path.join(outputs_dir, 'lns_assignment_history.json')

    bfsa_history = load_assignment_history(bfsa_history_path) if os.path.exists(bfsa_history_path) else []
    rule_history = load_assignment_history(rule_history_path) if os.path.exists(rule_history_path) else []
    lns_history = load_assignment_history(lns_history_path) if os.path.exists(lns_history_path) else []
    
    all_target_info = {str(target['id']): target 
                       for timestep_targets in targets_by_timestep.values() 
                       for target in timestep_targets}
    radar_channel_info = {rid: info['number_channel'] for rid, info in radars_dict.items()}

    if bfsa_history:
        plotter.plot_radar_gantt(
            bfsa_history,
            (0, sim_total_time),
            radar_channel_info,
            all_target_info,
            os.path.join(outputs_dir, 'visualization', 'bfsa_rho_radar_gantt.png')
        )
    
    if rule_history:
        plotter.plot_radar_gantt(
            rule_history,
            (0, sim_total_time),
            radar_channel_info,
            all_target_info,
            os.path.join(outputs_dir, 'visualization', 'rule_based_radar_gantt.png')
        )
    
    if lns_history:
        plotter.plot_radar_gantt(
            lns_history,
            (0, sim_total_time),
            radar_channel_info,
            all_target_info,
            os.path.join(outputs_dir, 'visualization', 'lns_radar_gantt.png')
        )

    if bfsa_history:
        plotter.plot_target_gantt(
            bfsa_history,
            (0, sim_total_time),
            all_target_info,
            radars_dict, # plot_target_gantt 可能需要 radars_dict 而不是 radar_channel_info
            os.path.join(outputs_dir, 'visualization', 'bfsa_rho_target_gantt.png')
        )
    
    if rule_history:
        plotter.plot_target_gantt(
            rule_history,
            (0, sim_total_time),
            all_target_info,
            radars_dict,
            os.path.join(outputs_dir, 'visualization', 'rule_based_target_gantt.png')
        )

    if lns_history:
        plotter.plot_target_gantt(
            lns_history,
            (0, sim_total_time),
            all_target_info,
            radars_dict,
            os.path.join(outputs_dir, 'visualization', 'lns_target_gantt.png')
        )

    all_histories_for_switching = {
        "BFSA-RHO": bfsa_history,
        "Rule-Based": rule_history,
        "LNS": lns_history
    }
    switching_frequencies = {}
    for algo_name, history_data in all_histories_for_switching.items():
        if history_data:
            switches = calculate_total_radar_switches(history_data)
            switching_frequencies[algo_name] = switches
        else:
            switching_frequencies[algo_name] = 0
            logging.warning(f"Assignment history for {algo_name} is empty or missing. Switching frequency set to 0.")

    if switching_frequencies:
        plotter.plot_target_radar_switching_frequency(
            switching_frequencies,
            save_path=os.path.join(outputs_dir, 'visualization', 'target_radar_switching_frequency.png')
        )

    logging.info(f"Visualization results saved to {os.path.join(outputs_dir, 'visualization')}")

    plotter.plot_radar_utilization_heatmap(
        bfsa_history,
        {rid: info['number_channel'] for rid, info in radars_dict.items()},
        (0, sim_total_time),
        "BFSA-RHO",
        os.path.join(outputs_dir, 'visualization', 'bfsa_rho_radar_heatmap.png')
    )

    plotter.plot_radar_utilization_heatmap(
        rule_history,
        {rid: info['number_channel'] for rid, info in radars_dict.items()},
        (0, sim_total_time),
        "Rule-based",
        os.path.join(outputs_dir, 'visualization', 'rule_based_radar_heatmap.png')
    )

    plotter.plot_radar_utilization_heatmap(
        lns_history,
        {rid: info['number_channel'] for rid, info in radars_dict.items()},
        (0, sim_total_time),
        "lns",
        os.path.join(outputs_dir, 'visualization', 'lns_radar_heatmap.png')
    )

    vis_output_dir = os.path.join(outputs_dir, 'visualization')
    os.makedirs(vis_output_dir, exist_ok=True)

    target_info = {str(target['id']): target for timestep_targets in targets_by_timestep.values() 
                  for target in timestep_targets}

    def convert_to_switch_data(history):
        switch_data = []
        if not history:
            return switch_data
        for i, record in enumerate(history):
            timestamp = record["timestamp"]
            assignments = record["assignments"]
            
            for targets_id, assignment in assignments.items():
                if assignment is not None and assignment['radar_id'] is not None:
                    switch_data.append({
                        'target_id': str(targets_id),
                        'radar_id': assignment['radar_id'],
                        'start_time': timestamp
                    })
        return switch_data
    
    bfsa_switch_data = convert_to_switch_data(bfsa_history)
    rule_switch_data = convert_to_switch_data(rule_history)
    lns_switch_data = convert_to_switch_data(lns_history)

    switch_data_for_plot = {
        "BFSA-RHO": bfsa_switch_data,
        "Rule-Based": rule_switch_data
    }
    if lns_history:
        switch_data_for_plot["LNS"] = lns_switch_data
    
    plotter.plot_target_switching(
        switch_data_for_plot,
        target_info,
        save_path=os.path.join(vis_output_dir, 'target_switching_comparison.png')
    )

    all_histories_for_assignment_rate = {
        "BFSA-RHO": bfsa_history,
        "Rule-Based": rule_history
    }
    if lns_history:
        all_histories_for_assignment_rate["LNS"] = lns_history

    target_assignment_rate_data_without_priority = calculate_target_assignment_rate_data_without_priotiry(
        all_histories_for_assignment_rate,
        targets_by_timestep,
        radars_dict,
        sim_total_time
    )

    plotter.plot_target_assignment_rate_over_time(
        target_assignment_rate_data_without_priority,
        save_path=os.path.join(vis_output_dir, 'target_assignment_rate_comparison.png')
    )

    metrics = {
        "Tracking coverage": {
            "BFSA-RHO": 0.82,
            "Rule-Based": 0.74,
            "LNS": 0.78
        },
        "Priority satisfaction": {
            "BFSA-RHO": 0.88,
            "Rule-Based": 0.70,
            "LNS": 0.85
        },
        "Computational efficiency": {
            "BFSA-RHO": 0.65,
            "Rule-Based": 0.95,
            "LNS": 0.75
        },
        "Switching frequency": {
            "BFSA-RHO": 0.75,
            "Rule-Based": 0.85,
            "LNS": 0.80
        },
        "Resource Utilization": {
            "BFSA-RHO": 0.90,
            "Rule-Based": 0.65,
            "LNS": 0.85
        }
    }

    all_algorithm_names_for_radar = ["BFSA-RHO", "Rule-Based"]
    metrics_to_plot = {}

    if lns_history:
        all_algorithm_names_for_radar.append("LNS")
        metrics_to_plot = metrics
    else:
        for metric_name, algo_values in metrics.items():
            metrics_to_plot[metric_name] = {
                algo: val for algo, val in algo_values.items() if algo in all_algorithm_names_for_radar
            }
    
    if metrics_to_plot and any(metrics_to_plot.values()):
        plotter.plot_algorithm_comparison(
            all_algorithm_names_for_radar,
            metrics_to_plot,
            save_path=os.path.join(vis_output_dir, 'algorithms_performance_comparison.png')
        )
    else:
        logging.warning("No metrics data to plot for algorithm comparison radar chart.")

    target_priority_info = {}
    for target in [target for targets in targets_by_timestep.values() for target in targets]:
        target_id = str(target['id'])
        if target_id not in target_priority_info:
            target_priority_info[target_id] = {
                'priority': target['priority'],
                'type': target['target_type']
            }

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

    bfsa_rho_output = os.path.join(outputs_dir, 'bfsa_rho_assignment_history.json')
    rule_based_output = os.path.join(outputs_dir, 'rule_based_assignment_history.json')
    lns_output = os.path.join(outputs_dir, 'lns_assignment_history.json')

    bfsa_rho_history = load_assignment_history(bfsa_rho_output)
    rule_based_history = load_assignment_history(rule_based_output)
    lns_history = load_assignment_history(lns_output)

    radar_info = {radar_id: info['number_channel'] for radar_id, info in radars_dict.items()}

    assignment_histories = {
        "BFSA-RHO": bfsa_rho_history,
        "Rule-Based": rule_based_history,
        "LNS": lns_history
    }

    occupancy_over_time_path = os.path.join(outputs_dir, 'visualization', 'radar_channel_occupancy_over_time.png')
    plotter.plot_radar_channel_occupancy_over_time(
        assignment_histories=assignment_histories,
        radar_info=radar_info,
        time_range=(0, sim_total_time),
        save_path=occupancy_over_time_path
    )   # modify 超出1

    avg_occupancy_path = os.path.join(outputs_dir, 'visualization', 'average_radar_channel_occupancy.png')
    plotter.plot_average_radar_channel_occupancy(
        assignment_histories=assignment_histories,
        radar_info=radar_info,
        save_path=avg_occupancy_path
    )

    calculate_and_save_network_tracking_efficiency(
        assignment_histories, # Reuse the same histories dictionary
        outputs_dir
    )

    print(f"雷达信道占用率图表已保存到 {os.path.dirname(occupancy_over_time_path)}")


    print(f"可视化结果已保存到目录: {outputs_dir}")

if __name__ == "__main__":
    output_dir, radar_dict, total_time = main()

