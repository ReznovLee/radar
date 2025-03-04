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


from core.models.target_model import (
    BallisticMissileTargetModel,
    CruiseMissileTargetModel,
    AircraftTargetModel,
    MACH_2_MS,
    GRAVITY
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


def generate_radars():
    """ Generate radars using parameter configuration

    The radar list was generated according to the parameters and configuration information, and the center point of
    the radar network was taken as the center. The center coordinates of each radar in the radar network were
    constructed according to the aggregation rate of each radar, and then the corresponding channel number and
    radiation range were randomly generated.

    :return: list of radars, each radar contains center coordinates, radiation range and number of channels.
    """
    config = load_config('param_config.yaml')
    num_radars = config["num_radars"]
    radar_network_center_str = config["radar_network_center_str"]
    radar_aggregation_rate = config["radar_aggregation_rate"]

    if not 0 < radar_aggregation_rate <= 1:
        raise ValueError('The aggregation rate must be between 0 and 1.')
    if num_radars <= 0:
        raise ValueError('The number of radars must be greater than 0.')

    distribution_range = 1000 * radar_aggregation_rate

    radar_network_center = np.array([float(x.strip()) for x in radar_network_center_str.strip('()').split(',')])

    x_coordinates = np.random.uniform(
        radar_network_center[0] - distribution_range,
        radar_network_center[0] + distribution_range,
        num_radars
    )

    y_coordinates = np.random.uniform(
        radar_network_center[1] - distribution_range,
        radar_network_center[1] + distribution_range,
        num_radars
    )

    z_coordinates = 0

    min_radius_range = 30000  # Min radius range of radars
    max_radius_range = 50000  # Max radius range of radars
    min_channel_number = 7  # Min channel number of radars
    max_channel_number = 10  # Max channel number of radars

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
    aircraft_count = num_targets - ballistic_count - cruise_count  # 确保总数匹配
    return {
        "ballistic_missile": ballistic_count,
        "cruise_missile": cruise_count,
        "aircraft": aircraft_count
    }


def generate_random_targets(center_drop_position_str, dispersion_rate, time_to_impact):
    """ Generate random targets

    The actual targets are generated based on the number of targets in each category. For the generation of the initial
    point of the target, the backward method is used, assuming that the target's landing point is distributed around a
    center point (the center point is tentatively considered as the center point of the radar network), and then the
    initial point 100 moments ago is roughly inverted according to the motion equations of ballistic missiles,
    cruise missiles and fighter jets. Then, the target id, the target's path point coordinate and speed at each moment
    are derived by time stepping, and the target list is returned.

    :param time_to_impact: time to impact
    :param center_drop_position_str: the landing center point coordinates of the target, and the path coordinates of the
                                    aircraft in the x_axis and y_axis direction
    :param dispersion_rate: the dispersion of the target from the central drop point, the larger the value, the more
                                dispersed.
    :return: list of random targets
    """
    if not 0 < dispersion_rate <= 1:
        raise ValueError("dispersion_rate must between 0 to 1")

    # The mach speed of 3 type of targets
    config = load_config("param_config.yaml")
    BALLISTIC_MISSILE_MACH = config["speed"]["ballistic_speed"]
    CRUISE_MISSILE_MACH = config["speed"]["cruise_speed"]
    AIRCRAFT_MACH = config["speed"]["aircraft_speed"]

    # The standard speed of 3 type of targets
    ballistic_missile_ms = BALLISTIC_MISSILE_MACH * MACH_2_MS
    cruise_missile_ms = CRUISE_MISSILE_MACH * MACH_2_MS
    aircraft_ms = AIRCRAFT_MACH * MACH_2_MS

    # Sample param data
    BASE_SAMPLE_POINTS = 100
    ballistic_samples = BASE_SAMPLE_POINTS
    cruise_samples = int(BASE_SAMPLE_POINTS * (BALLISTIC_MISSILE_MACH / CRUISE_MISSILE_MACH))
    aircraft_samples = int(BASE_SAMPLE_POINTS * (BALLISTIC_MISSILE_MACH / AIRCRAFT_MACH))

    # Load scenario config
    num_targets = config["num_targets"]
    target_ratio = config["target_ratio"]
    num_counts = compute_target_counts(num_targets, target_ratio)
    ballistic_counts = num_counts["ballistic_missile"]
    cruise_counts = num_counts["cruise_missile"]
    aircraft_counts = num_counts["aircraft"]

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

        gravity = GRAVITY
        min_elevation = np.pi / 6
        max_elevation = np.pi / 3

        elevation_angle = - np.random.uniform(
            min_elevation,
            max_elevation
        )

        min_azimuth = - np.pi / 12
        max_azimuth = - np.pi / 3

        azimuth_angle = - np.random.uniform(
            min_azimuth,
            max_azimuth
        )

        initial_velocity = np.array([
            ballistic_missile_ms * np.cos(elevation_angle) * np.cos(azimuth_angle),
            ballistic_missile_ms * np.cos(elevation_angle) * np.sin(azimuth_angle),
            ballistic_missile_ms * np.sin(elevation_angle)
        ])

        initial_position = np.array([
            drop_point[0] - initial_velocity[0] * time_to_impact,
            drop_point[1] - initial_velocity[1] * time_to_impact,
            (initial_velocity[2] + gravity[2]) * time_to_impact
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

        cruise_altitude = 3000
        dive_distance_horizontal = 500
        rocket_acceleration = MACH_2_MS

        delta_x = drop_point[0] - center_drop_position[0]
        delta_y = drop_point[1] - center_drop_position[1]
        direction_2d = np.array([delta_x, delta_y]) / np.sqrt(delta_x ** 2 + delta_y ** 2)
        direction = np.array([direction_2d[0], direction_2d[1], 0])  # 扩展为3D向量，z分量为0
        cruise_end_point = np.array([
            drop_point[0] - direction[0] * dive_distance_horizontal,
            drop_point[1] - direction[1] * dive_distance_horizontal,
            cruise_altitude
        ])

        dive_distance = np.sqrt(cruise_altitude**2 + dive_distance_horizontal**2)  # 俯冲阶段运动距离
        dive_time = np.sqrt(2 * dive_distance / rocket_acceleration)  # 俯冲时间

        # 计算巡航阶段时间和初始位置
        cruise_time = time_to_impact - dive_time
        cruise_distance = np.linalg.norm([delta_x, delta_y]) - dive_distance_horizontal  # 巡航总距离
        
        # 初始位置（从落点反推）
        initial_position = np.array([
            cruise_end_point[0] - direction[0] * cruise_distance,
            cruise_end_point[1] - direction[1] * cruise_distance,
            cruise_altitude
        ])

        # 初始速度（巡航阶段平均速度）
        initial_velocity = np.array([
            cruise_missile_ms * direction[0],
            cruise_missile_ms * direction[1],
            0
        ])

        # 创建目标并生成轨迹
        target = CruiseMissileTargetModel(
            current_id, 
            initial_position, 
            initial_velocity,
            cruise_end_point=cruise_end_point,
            dive_time=dive_time,
            cruise_time=cruise_time,
            rocket_acceleration=rocket_acceleration
        )
        cruise_dt = time_to_impact / cruise_samples
        generate_trajectory_points(target, cruise_samples, cruise_dt, targets_data)
        current_id += 1

    for _ in range(aircraft_counts):
        target_end_point = np.array([
            center_drop_position[0] + np.random.uniform(-distribution_range, distribution_range),
            center_drop_position[1] + np.random.uniform(-distribution_range, distribution_range),
            np.random.uniform(5000, 13000)
        ])

        # 计算水平面上的飞行方向（只考虑x-y平面）
        delta_x = target_end_point[0] - center_drop_position[0]
        delta_y = target_end_point[1] - center_drop_position[1]
        direction = np.array([delta_x, delta_y]) / np.sqrt(delta_x**2 + delta_y**2)

        initial_position = np.array([
            target_end_point[0] - direction[0] * aircraft_ms * time_to_impact,
            target_end_point[1] - direction[1] * aircraft_ms * time_to_impact,
            np.random.uniform(5000, 13000)
        ])

        initial_velocity = np.array([
            aircraft_ms * direction[0],
            aircraft_ms * direction[1],
            np.random.uniform(-50, 50)
        ])

        target = AircraftTargetModel(
            current_id, 
            initial_position, 
            initial_velocity
        )
        aircraft_dt = time_to_impact / aircraft_samples
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
    target_drop_position = config["radar_network_center_str"]
    target_aggregation_rate = config["target_aggregation_rate"]

    # 创建输出文件夹
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_folder_path = f"scenario-{current_date}"

    # 生成并保存雷达数据
    radars = generate_radars()
    radar_file_name = config["output"]["radar_filename_template"].format(num_radars=num_radars)
    save_radars_2_csv(radars, output_folder_path, radar_file_name)

    # 生成并保存目标数据
    targets = generate_random_targets(target_drop_position, target_aggregation_rate, time_to_impact=100)
    target_file_name = config["output"]["target_filename_template"].format(num_targets=num_targets)
    save_targets_2_csv(targets, output_folder_path, target_file_name)


if __name__ == '__main__':
    generate_scenario()
