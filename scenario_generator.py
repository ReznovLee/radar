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
import random
import sys
from datetime import datetime
import os
import numpy as np
import yaml
import pandas as pd
import math

from src.core.models.target_model import (
    BallisticMissileTargetModel,
    CruiseMissileTargetModel,
    AircraftTargetModel,
    GRAVITY
)


def load_config(yaml_file):
    """ Load configuration from yaml file

    The basic parameter information of the scene is in "./data/config/param_config.yaml", including the number of
    targets/radars, the proportion of target types, simulation parameters and output file information.

    :param yaml_file: path to yaml file, Includes the basic parameters needed to generate the scene.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == 'win32':
        config_path = os.path.join(current_dir, 'src\data\config', yaml_file)
    else:
        config_path = os.path.join(current_dir, 'src/data/config', yaml_file)
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

    distribution_range = 10000 * radar_aggregation_rate

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


def generate_random_targets(center_drop_position_str, target_dispersion_rate, time_to_impact):
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
    :param target_dispersion_rate: the dispersion of the target from the central drop point, the larger the value, the
                                    more dispersed.
    :return: list of random targets
    """
    if not 0 < target_dispersion_rate <= 1:
        raise ValueError("dispersion_rate must between 0 to 1")

    # The speed of 3 type of targets
    config = load_config("param_config.yaml")
    ballistic_missile_speed = config["speed"]["ballistic_speed"]
    cruise_missile_speed = config["speed"]["cruise_speed"]
    aircraft_speed = config["speed"]["aircraft_speed"]

    # Sample param date
    TOTAL_SAMPLE = 1000
    dt = TOTAL_SAMPLE / time_to_impact

    # Load scenario config
    num_targets = config["num_targets"]
    target_ratio = config["target_ratio"]
    num_counts = compute_target_counts(num_targets, target_ratio)
    ballistic_counts = num_counts["ballistic_missile"]
    cruise_counts = num_counts["cruise_missile"]
    aircraft_counts = num_counts["aircraft"]

    target_distribution_range = 1000 * target_dispersion_rate
    targets_data = []
    current_id = 1

    center_drop_position = np.array([float(x.strip()) for x in center_drop_position_str.strip('()').split(',')])

    for _ in range(ballistic_counts):
        # Every ballistic missiles' drop points confirm
        drop_point = np.array([
            center_drop_position[0] + np.random.uniform(-target_distribution_range, target_distribution_range),
            center_drop_position[1] + np.random.uniform(-target_distribution_range, target_distribution_range),
            0
        ])

        alpha = random.uniform(4 * math.pi / 9, math.pi / 3)
        beta = random.uniform(-math.pi / 3, math.pi / 3)
        vz = ballistic_missile_speed * np.sin(alpha)
        v_xy = ballistic_missile_speed * np.cos(alpha)
        vx = v_xy * math.cos(beta)
        vy = v_xy * math.sin(beta)
        ballistic_missile_time_to_impact = time_to_impact / 10  # TODO: Debug param, when finish the code, delete it

        initial_x = drop_point[0] + vx * ballistic_missile_time_to_impact
        initial_y = drop_point[1] + vy * ballistic_missile_time_to_impact
        initial_z = vz * ballistic_missile_time_to_impact + 0.5 * GRAVITY[2] * ballistic_missile_time_to_impact ** 2

        initial_vz = vz + 0.5 * GRAVITY[2] * ballistic_missile_time_to_impact

        # Velocity initial
        initial_velocity = np.array([
            -vx,
            -vy,
            -initial_vz
        ])

        # Position initial
        initial_position = np.array([
            initial_x,
            initial_y,
            initial_z
        ])

        target = BallisticMissileTargetModel(current_id, initial_position, initial_velocity)
        generate_trajectory_points(target, TOTAL_SAMPLE, dt, targets_data)
        current_id += 1

    for _ in range(cruise_counts):
        drop_point = np.array([
            center_drop_position[0] + np.random.uniform(-target_distribution_range, target_distribution_range),
            center_drop_position[1] + np.random.uniform(-target_distribution_range, target_distribution_range),
            0
        ])

        # Create cruise missile target class
        cruise_missile = CruiseMissileTargetModel

        # Initial parameters of cruise missile
        cruise_altitude = cruise_missile.CRUISE_ALTITUDE
        initial_cruise_altitude = cruise_altitude + np.random.uniform(-100, 100)
        dive_distance_horizontal = cruise_missile.TRANSITION_DISTANCE
        rocket_acceleration_magnitude = 5

        theta = random.uniform(math.pi / 4, 3 * math.pi / 4)
        cruise_end_point = np.array([
            drop_point[0] - dive_distance_horizontal * math.cos(theta),
            drop_point[1] - dive_distance_horizontal * math.sin(theta),
            initial_cruise_altitude
        ])

        delta_x = drop_point[0] - cruise_end_point[0]
        delta_y = drop_point[1] - cruise_end_point[1]
        delta_z = -cruise_end_point[2]  # 从巡航高度到地面
        dive_direction = np.array([delta_x, delta_y, delta_z])
        dive_direction = dive_direction / np.linalg.norm(dive_direction)

        dive_distance = np.sqrt(dive_distance_horizontal ** 2 + initial_cruise_altitude ** 2)
        dive_time = dive_distance / cruise_missile_speed
        cruise_time = time_to_impact - dive_time

        if dive_time <= time_to_impact:
            if dive_time < time_to_impact * 0.3:
                dive_time = time_to_impact * 0.3
                cruise_time = time_to_impact * 0.7
        else:
            raise ValueError("dive_time must be smaller than time_to_impact")

        cruise_direction = np.array([-dive_direction[0], -dive_direction[1], 0])
        cruise_direction = cruise_direction / np.linalg.norm(cruise_direction)

        cruise_distance = cruise_missile_speed * cruise_time * 0.9

        initial_position = np.array([
            cruise_end_point[0] - cruise_direction[0] * cruise_distance,
            cruise_end_point[1] - cruise_direction[1] * cruise_distance,
            initial_cruise_altitude
        ])

        # 将标量加速度轉換為向量形式
        rocket_acceleration = rocket_acceleration_magnitude * dive_direction

        # 初始速度（巡航阶段平均速度）
        initial_velocity = cruise_missile_speed * cruise_direction

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

        generate_trajectory_points(target, TOTAL_SAMPLE, dt, targets_data)
        current_id += 1

    for _ in range(aircraft_counts):
        # 1. Generate initial position far from the center
        initial_distance = random.uniform(50000, 100000) # Start 50-100km away
        initial_angle = random.uniform(0, 2 * math.pi)
        initial_position = np.array([
            center_drop_position[0] + initial_distance * math.cos(initial_angle),
            center_drop_position[1] + initial_distance * math.sin(initial_angle),
            random.uniform(AircraftTargetModel.MIN_ALTITUDE, AircraftTargetModel.MAX_ALTITUDE) # Start within altitude limits
        ])

        # 2. Define a target point near the center drop position to aim towards
        target_point_near_center = np.array([
            center_drop_position[0] + np.random.uniform(-target_distribution_range, target_distribution_range),
            center_drop_position[1] + np.random.uniform(-target_distribution_range, target_distribution_range),
            # Z component of target point doesn't strictly matter for initial horizontal velocity
            random.uniform(AircraftTargetModel.MIN_ALTITUDE, AircraftTargetModel.MAX_ALTITUDE)
        ])

        # 3. Calculate initial velocity direction towards the target point
        direction_vector = target_point_near_center - initial_position
        direction_xy = direction_vector[:2] # Horizontal direction

        # Normalize horizontal direction
        norm_xy = np.linalg.norm(direction_xy)
        if norm_xy < 1e-6: # Avoid division by zero if start point is too close horizontally
            # Default to a random horizontal direction if norm is too small
             random_angle = random.uniform(0, 2 * math.pi)
             direction_xy_norm = np.array([math.cos(random_angle), math.sin(random_angle)])
        else:
             direction_xy_norm = direction_xy / norm_xy

        # Set initial velocity
        initial_velocity = np.array([
            aircraft_speed * direction_xy_norm[0],
            aircraft_speed * direction_xy_norm[1],
            0.0  # Start with zero vertical velocity, altitude control is handled later
        ])

        # 4. Create AircraftTargetModel and generate trajectory
        target = AircraftTargetModel(
            current_id,
            initial_position,
            initial_velocity
        )
        # Use the dedicated aircraft trajectory generator which handles altitude constraints
        generate_aircraft_trajectory_points(target, TOTAL_SAMPLE, dt, targets_data)
        current_id += 1

    targets_data.sort(key=lambda x: (x['id'], x['timestep']))
    return targets_data


def generate_trajectory_points(target, samples, dt, targets_data):
    """ Function that generates trajectory points for targets.

    This function is used to generate trajectory data points. 
    For the specific data format, refer to the TargetModel class. 
    The way to generate data is mainly to advance the simulation time and obtain the target state 
    with different timestamps from the TargetModel class. The target object is used to access 
    class-related methods. samples indicates the number of samples, which is synchronized with 
    the situation update frequency. dt indicates the update time interval. target_data indicates 
    the temporarily stored target trajectory point data.

    :param target: target object initialized from core.models.target_model
    :param samples: samples that need sampling from time step
    :param dt: time step
    :param targets_data: list of target data
    """
    current_time = 0
    last_position = None

    for _ in range(samples):
        state = target.get_state(current_time)
        current_position = np.array(state[2], dtype=np.float64)
        if current_position[2] > 0:
            targets_data.append({
                'id': state[0],
                'timestep': current_time,
                'position': current_position.copy(),
                'velocity': np.array(state[3], dtype=np.float64).copy(),
                'target_type': state[4],
                'priority': state[5]
            })
            target.update_state(dt)
            last_position = current_position.copy()
            current_time += dt
        else:
            xp, yp, zp = last_position
            xc, yc, zc = current_position
            t = - zp / (zc - zp)
            cross_x = xp + t * (xc - xp)
            cross_y = yp + t * (yc - yp)
            landing_position = [cross_x, cross_y, 0]
            targets_data.append({
                'id': state[0],
                'timestep': current_time,
                'position': landing_position.copy(),
                'velocity': np.zeros(3),
                'target_type': state[4],
                'priority': state[5]
            })
            break


def generate_aircraft_trajectory_points(target, samples, dt, targets_data):
    """ Function that generates trajectory points for aircraft.

    This function generates aircraft trajectory points. If the aircraft's altitude
    goes outside the defined MIN/MAX range, it adjusts the vertical velocity
    to guide it back smoothly, avoiding abrupt position changes.

    :param target: aircraft target object initialized from AircraftTargetModel (assumes target.velocity is accessible)
    :param samples: number of samples needed
    :param dt: time step
    :param targets_data: list to store trajectory data
    """
    current_time = 0
    # Define vertical correction speed (adjust as needed for smoothness)
    CORRECTION_VZ_UP = 20.0   # Speed to apply when below min altitude
    CORRECTION_VZ_DOWN = -20.0 # Speed to apply when above max altitude

    for _ in range(samples):
        # 1. Get current state
        state = target.get_state(current_time)
        current_position = np.array(state[2], dtype=np.float64)
        current_velocity = np.array(state[3], dtype=np.float64) # Keep current velocity for recording

        # 2. Record the current state *before* applying corrections for the next step
        targets_data.append({
            'id': state[0],
            'timestep': current_time,
            'position': current_position.copy(),
            'velocity': current_velocity.copy(), # Record the actual velocity at this time step
            'target_type': state[4],
            'priority': state[5]
        })

        # 3. Check altitude and determine if correction is needed for the *next* state update
        needs_correction = False
        target_vz = current_velocity[2] # Default to current vz if no correction needed

        if current_position[2] < target.MIN_ALTITUDE:
            # Apply upward correction velocity for the next update step
            target_vz = CORRECTION_VZ_UP
            needs_correction = True
            # Optional: print a warning
            # print(f"Warning: Aircraft {state[0]} below min altitude ({current_position[2]:.2f}m). Applying upward correction.")
        elif current_position[2] > target.MAX_ALTITUDE:
            # Apply downward correction velocity for the next update step
            target_vz = CORRECTION_VZ_DOWN
            needs_correction = True
            # Optional: print a warning
            # print(f"Warning: Aircraft {state[0]} above max altitude ({current_position[2]:.2f}m). Applying downward correction.")

        # 4. Apply correction to the target's internal state *before* the update
        #    This assumes direct modification of target.velocity is possible and intended.
        if needs_correction:
            # Directly modify the target's internal velocity for the next update calculation
            # Ensure target object allows this modification.
            try:
                 target.velocity[2] = target_vz
                 # If the model also uses acceleration, consider resetting vertical acceleration
                 # if hasattr(target, 'acceleration'):
                 #    target.acceleration[2] = 0.0
            except AttributeError:
                 print(f"Error: Cannot directly modify target.velocity for Aircraft {state[0]}. Model modification might be needed.")
                 # Decide how to handle this - maybe break or continue without correction?
                 pass # Continue without correction for now if attribute doesn't exist

        # 5. Update the target's state using the (potentially modified) velocity
        target.update_state(dt)

        # 6. Increment time
        current_time += dt


def generate_trajectory_points1(target, samples, dt, targets_data):
    """
    生成目标的轨迹数据，直到目标落地后保持静止。

    参数：
    - target: 目标对象，具有 update_state(dt) 方法
    - samples: 采样次数
    - dt: 初始时间间隔
    - targets_data: 轨迹数据列表
    """
    current_time = 0  # 当前时间

    last_position = None  # 记录上一时刻的位置

    # last_state = None  # 记录上一时刻的状态

    for _ in range(samples):
        state = target.get_state(current_time)  # 获取当前状态
        current_position = np.array(state[2], dtype=np.float64)
        current_velocity = np.array(state[3], dtype=np.float64)

        # Debug 输出
        print(f"Timestep: {current_time}, Z: {state[2][2]}")

        # if state[2][2] > 0:  # **z > 0，正常存储**
        if current_position[2] > 0:  # **z > 0，正常存储**
            targets_data.append({
                'id': state[0],
                'timestep': current_time,
                'position': current_position.copy(),
                'velocity': current_velocity.copy(),
                'target_type': state[4],
                'priority': state[5]
            })
            # last_state = state.copy()  # 记录上一时刻的状态
            last_position = current_position.copy()  # 记录上一时刻的位置
            target.update_state(dt)
            current_time += dt
        else:  # **z ≤ 0，计算交点**
            if last_position is None:
                print("警告：没有上一时刻的状态，无法计算精确落地点")
                # **修正落地点**
                fixed_position = [current_position[0], current_position[1], 0]
                # **落地后速度归零**
                fixed_velocity = [0, 0, 0]
                # **落地后速度归零**
            else:
                x1, y1, z1 = last_position
                x2, y2, z2 = current_position

                if abs(z2 - z1) > 1e-6:  # 避免除零错误
                    # 计算线段与平面交点的参数t
                    t = -z1 / (z2 - z1)  # 计算时间比例
                    # 使用线性插值计算交点的x和y坐标
                    intersection_x = x1 + t * (x2 - x1)
                    intersection_y = y1 + t * (y2 - y1)
                    # 计算落地时刻（在当前时间步内的精确时刻）
                    landing_time = current_time - dt + t * dt
                else:
                    # 如果z方向变化很小，直接使用上一时刻的位置
                    intersection_x, intersection_y = x1, y1
                    landing_time = current_time - dt

                # 修正落地点
                fixed_position = [intersection_x, intersection_y, 0]
                # 落地后速度归零
                fixed_velocity = [0, 0, 0]
                # 使用精确地落地时刻
                current_time = landing_time
            # **记录修正后的落地状态**
            targets_data.append({
                'id': state[0],
                'timestep': current_time,
                'position': fixed_position,
                'velocity': fixed_velocity,
                'target_type': state[4],
                'priority': state[5]
            })
            print(f"Target landed at timestep {current_time}, final position: {fixed_position}")
            break  # **目标落地后，停止记录**

        current_time += dt


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
    output_folder_path = f"output/scenario-{current_date}"

    # 生成并保存雷达数据
    radars = generate_radars()
    radar_file_name = config["output"]["radar_filename_template"].format(num_radars=num_radars)
    save_radars_2_csv(radars, output_folder_path, radar_file_name)

    # 生成并保存目标数据
    targets = generate_random_targets(target_drop_position, target_aggregation_rate, time_to_impact=1000)
    target_file_name = config["output"]["target_filename_template"].format(num_targets=num_targets)
    save_targets_2_csv(targets, output_folder_path, target_file_name)


if __name__ == '__main__':
    generate_scenario()
