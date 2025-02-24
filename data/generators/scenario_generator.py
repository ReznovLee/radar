# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : scenario_generator.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:27
"""
import csv
from datetime import datetime
import os
import numpy as np
import yaml
import pandas as pd

import core.models.target_model
from core.models.target_model import (
    BallisticMissileTargetModel,
    CruiseMissileTargetModel,
    AircraftTargetModel,
    MACH_2_MS
)


def load_config(yaml_file):
    """ Load configuration from yaml file

    The basic parameter information of the scene is in "./data/config/param_config.yaml ", including the number of
    targets/radars, the proportion of target types, simulation parameters and output file information.

    :param yaml_file: path to yaml file, Includes the basic parameters needed to generate the scene.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..\\config', yaml_file)
    with open(config_path, 'r', encoding='UTF-8') as stream:
        config = yaml.safe_load(stream)
    return config


def generate_radars(num_radars, radar_network_center_str, aggregation_rate):
    """ Generate radars using parameter configuration

    The radar list was generated according to the parameters and configuration information, and the center point of
    the radar network was taken as the center. The center coordinates of each radar in the radar network were
    constructed according to the aggregation rate of each radar, and then the corresponding channel number and
    radiation range were randomly generated.

    :param num_radars: number of radars parameter from yaml file
    :param radar_network_center_str: center of radar network, Represents the center coordinates of the radar network,
                                    which facilitates the construction of the radar scene
    :param aggregation_rate: aggregation rate, Represents the degree of radar aggregation, and the value ranges
                                from (0,1], where 0 means that all radars are in one point, and 1 means that all radars
                                are completely scattered without coincidence.
    :return: list of radars, each radar contains center coordinates, radiation range and number of channels.
    """
    if not 0 < aggregation_rate <= 1:
        raise ValueError('The aggregation rate must be between 0 and 1.')
    if num_radars <= 0:
        raise ValueError('The number of radars must be greater than 0.')

    distribution_range = 1000 * aggregation_rate

    radar_network_center = np.array([float(x.strip()) for x in radar_network_center_str.strip('()').split(',')])

    x_coordinates = np.random.uniform(
        radar_network_center[0] - distribution_range,
        radar_network_center[0] + distribution_range,
        num_radars)

    y_coordinates = np.random.uniform(
        radar_network_center[1] - distribution_range,
        radar_network_center[1] + distribution_range,
        num_radars
    )

    z_coordinates = 0

    min_radius_range = 30000
    max_radius_range = 50000
    min_channel_number = 7
    max_channel_number = 10

    radars = []
    for index in range(num_radars):
        radar_label = {
            'radar_id': index,
            'center': (int(x_coordinates[index]), int(y_coordinates[index]), int(z_coordinates)),
            'radius': int(np.random.uniform(min_radius_range, max_radius_range)),
            'num_channels': int(np.random.uniform(min_channel_number, max_channel_number + 1))
        }
        radars.append(radar_label)

    return radars


def save_radars_2_csv(radars, radar_folder_path, radar_file_name):
    """ Save radars to csv file

    Save the generated radar data to a csv file with the header id,x,y,z,radius,number_channel.

    :param radars: list of radars
    :param radar_folder_path: path to save the csv file
    :param radar_file_name: name of the csv file
    """
    os.makedirs(radar_folder_path, exist_ok=True)
    radar_folder_path = os.path.join(radar_folder_path, radar_file_name)

    with open(radar_folder_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "x", "y", "z", "radius", "number_channel"])

        for radar in radars:
            writer.writerow([radar["radar_id"], radar["center"][0], radar["center"][1], radar["center"][2], radar[
                'radius'], radar["num_channels"]])


def compute_target_counts(num_targets, target_ratio):
    """ Compute target counts

    According to the proportion of each type of target, the actual number of targets in each category is calculated.

    :param num_targets: number of targets
    :param target_ratio: ratio of targets
    :return: list of target counts
    """
    ballistic_count = int(num_targets * target_ratio["ballistic_missile"])
    cruise_count = int(num_targets * target_ratio["cruise_missile"])
    fighter_count = num_targets - ballistic_count - cruise_count  # 确保总数匹配
    return {
        "ballistic_missile": ballistic_count,
        "cruise_missile": cruise_count,
        "fighter_jet": fighter_count
    }


def generate_random_targets(center_drop_position_str, dispersion_rate):
    """ Generate random targets

    The actual targets are generated based on the number of targets in each category. For the generation of the initial
    point of the target, the backward method is used, assuming that the target's landing point is distributed around a
    center point (the center point is tentatively considered as the center point of the radar network), and then the
    initial point 100 moments ago is roughly inverted according to the motion equations of ballistic missiles,
    cruise missiles and fighter jets. Then, the target id, the target's path point coordinate and speed at each moment
    are derived by time stepping, and the target list is returned.

    :param center_drop_position_str: the landing center point coordinates of the target, and the path coordinates of the
                                    aircraft in the x_axis and y_axis direction
    :param dispersion_rate: the dispersion of the target from the central drop point, the larger the value, the more
                                dispersed.
    :return: list of random targets
    """
    if not 0 < dispersion_rate <= 1:
        raise ValueError("dispersion_rate must between 0 to 1")

    BALLISTIC_MISSILE_MACH = 8.0
    CRUISE_MISSILE_MACH = 0.8
    AIRCRAFT_MACH = 1.5
    ballistic_missile_ms = BALLISTIC_MISSILE_MACH * MACH_2_MS
    cruise_missile_ms = CRUISE_MISSILE_MACH * MACH_2_MS
    aircraft_ms = AIRCRAFT_MACH * MACH_2_MS

    BASE_SAMPLE_POINTS = 100
    ballistic_samples = BASE_SAMPLE_POINTS
    cruise_samples = int(BASE_SAMPLE_POINTS * (BALLISTIC_MISSILE_MACH / CRUISE_MISSILE_MACH))
    aircraft_samples = int(BASE_SAMPLE_POINTS * (BALLISTIC_MISSILE_MACH / AIRCRAFT_MACH))

    config = load_config("param_config.yaml")
    num_targets = config["num_targets"]
    target_ratio = config["target_ratio"]
    num_counts = compute_target_counts(num_targets, target_ratio)
    ballistic_counts = num_counts["ballistic_missile"]
    cruise_counts = num_counts["cruise_missile"]
    aircraft_counts = num_counts["fighter_jet"]

    distribution_range = 1000 * dispersion_rate
    targets_data = []
    current_id = 1

    center_drop_position = np.array([float(x.strip()) for x in center_drop_position_str.strip('()').split(',')])

    for _ in range(ballistic_counts):
        drop_point = np.array([
            center_drop_position[0] + np.random.uniform(-distribution_range, distribution_range),
            center_drop_position[1] + np.random.uniform(-distribution_range, distribution_range),
            0
        ])

        time_to_impact = 100
        gravity = core.models.target_model.BallisticMissileTargetModel.GRAVITY
        initial_z = 20000
        initial_position = np.array([
            drop_point[0] - ballistic_missile_ms * time_to_impact,
            drop_point[1],
            initial_z
        ])
        initial_velocity = np.array([
            ballistic_missile_ms,
            0,
            -np.sqrt(2 * gravity[2] * initial_z)
        ])

        target = BallisticMissileTargetModel(current_id, initial_position, initial_velocity)
        ballistic_dt = time_to_impact / ballistic_samples
        generate_trajectory_points(target, ballistic_samples, ballistic_dt, targets_data)
        current_id += 1

    for _ in range(cruise_counts):
        drop_point = np.array([
            center_drop_position[0] + np.random.uniform(-distribution_range, distribution_range),
            center_drop_position[1] + np.random.uniform(-distribution_range, distribution_range),
            0
        ])

        cruise_altitude = 8000
        initial_position = np.array([
            drop_point[0] - cruise_missile_ms * 100,
            drop_point[1],
            cruise_altitude
        ])
        initial_velocity = np.array([cruise_missile_ms, 0, 0])

        target = CruiseMissileTargetModel(current_id, initial_position, initial_velocity)
        cruise_dt = 100 / cruise_samples
        generate_trajectory_points(target, cruise_samples, cruise_dt, targets_data)
        current_id += 1

    for _ in range(aircraft_counts):
        target_point = np.array([
            center_drop_position[0] + np.random.uniform(-distribution_range, distribution_range),
            center_drop_position[1] + np.random.uniform(-distribution_range, distribution_range),
            np.random.uniform(900, 8100)
        ])

        initial_position = np.array([
            target_point[0] - aircraft_ms * 100,
            target_point[1],
            target_point[2]
        ])
        initial_velocity = np.array([aircraft_ms, 0, 0])

        target = AircraftTargetModel(current_id, initial_position, initial_velocity)
        aircraft_dt = 100 / aircraft_samples
        generate_trajectory_points(target, aircraft_samples, aircraft_dt, targets_data)
        current_id += 1

    targets_data.sort(key=lambda x: (x['id'], x['timestep']))
    return targets_data


def generate_trajectory_points(target, samples, dt, targets_data):
    """ Generates trajectory points based on target and sample points.

    Generate the target list as required by the csv file.

    :param target: target set
    :param samples: sample points
    :param dt: time step
    :param targets_data: target data
    """
    for t in range(samples):
        timestamp = t * dt
        state = target.get_state(timestamp)
        targets_data.append({
            'id': state[0],
            'timestep': state[1],
            'position': state[2],
            'velocity': state[3],
            'target_type': state[4],
            'priority': state[5]
        })
        target.update_position(dt)


def save_targets_2_csv(targets_data, target_folder_path, target_file_name):
    """ Save targets to csv file

        Save the generated target data to a csv file.

        :param targets_data: list of targets
        :param target_folder_path: path to save the csv file
        :param target_file_name: name of the csv file
    """
    if target_folder_path:
        os.makedirs(target_folder_path, exist_ok=True)
        target_file_path = os.path.join(target_folder_path, target_file_name)
    else:
        target_file_path = target_file_name
    # os.makedirs(target_folder_path, exist_ok=True)
    # target_file_path = os.path.join(target_folder_path, target_file_name)

    expanded_data = []
    for target in targets_data:
        expanded_target = {
            'id': target['id'],
            'timestep': target['timestep'],
            'position_x': target['position'][0],
            'position_y': target['position'][1],
            'position_z': target['position'][2],
            'velocity_x': target['velocity'][0],
            'velocity_y': target['velocity'][1],
            'velocity_z': target['velocity'][2],
            'target_type': target['target_type'],
            'priority': target['priority']
        }
        expanded_data.append(expanded_target)

    df = pd.DataFrame(expanded_data)
    df = df.sort_values(['id', 'timestep'])
    df.to_csv(target_file_path, index=False)
    print(f'The target data has been saved to: {target_file_path}')


def generate_scenario():
    """ Generates scenario based on target and sample points.
    Generate the target list as required by the csv file.
    """
    config = load_config("param_config.yaml")

    num_targets = config['num_targets']
    num_radars = config['num_radars']
    radar_center = config['radar_network_center']
    radar_aggregation_rate = config['radar_aggregation_rate']
    target_drop_position = config["radar_network_center"]
    target_aggregation_rate = config["target_aggregation_rate"]

    # 创建输出文件夹
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_folder_path = f"scenario-{current_date}"

    # 生成并保存雷达数据
    radars = generate_radars(num_radars, radar_center, radar_aggregation_rate)
    radar_file_name = config["output"]["radar_filename_template"].format(num_radars=num_radars)
    save_radars_2_csv(radars, output_folder_path, radar_file_name)

    # 生成并保存目标数据
    targets = generate_random_targets(target_drop_position, target_aggregation_rate)
    target_file_name = config["output"]["target_filename_template"].format(num_targets=num_targets)
    save_targets_2_csv(targets, output_folder_path, target_file_name)


if __name__ == '__main__':
    generate_scenario()
