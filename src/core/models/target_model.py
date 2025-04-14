# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : target_model.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:25
"""
import math
import numpy as np

GRAVITY = np.array([0, 0, -9.81])
SEA_LEVEL_AIR_DENSITY = 1.225
ATMOSPHERIC_SCALE_HEIGHT = 8500


class TargetModel:
    """Basic target model

    The base class of the target model, inherited from the three target classes.

    Attributes:
        target_id:          Each target has its own unique ID (int) that distinguishes it from each other.
        target_position:    The 3D coordinates of the target at any time, and the specific kinematic equations follow
                                the description in the paper.
        velocity:           Here param_config.yaml takes an input parameter of M/S, which was used in paper to visualize
                                the speeds of the three targets.
        target_type:        It mainly includes three types: ballistic missiles, cruise missiles and fighter jets.
        priority:           Because of their different speeds and functions in combat,
                                they are simply given corresponding priorities. It is mainly divided into three levels:
                                level 1 (the most priority), Level 2 (the second priority),
                                and Level 3 (the lowest level).
    """

    def __init__(self, target_id, target_position, velocity, target_type, priority):
        """ Initializes the target model.

        Initialization of the target model class, including properties and methods of the target.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity: Velocity of the target
        :param target_type: Unity type of the target
        :param priority: Priority of the target
        """
        self.target_id = target_id
        self.target_position = np.array(target_position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.target_type = target_type
        self.priority = priority
        self.acceleration = np.zeros(3)
        self.velocity_disturbance = np.zeros(3)

    def update_state(self, delta_time):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param delta_time: The time interval between sampling points
        """
        self.velocity = self.acceleration * delta_time + self.velocity_disturbance
        self.target_position += delta_time * self.velocity

    def get_state(self, timestamp):
        """Gets state of the target model.

        The target state is updated at each timestamp, including all attributes of the target class.

        :param timestamp: Timestamp
        :return: State of the target model at a given timestamp -> list
        """
        return [
            self.target_id,
            timestamp,
            self.target_position,
            self.velocity,
            self.target_type,
            self.priority]


class BallisticMissileTargetModel(TargetModel):
    """Ballistic Missile target model

    Ballistic missile model class, inherited from TargetModel, whose trajectory is approximately parabolic.

    Attributes:
        target_id:          The unique ID of the target, inherited from the TargetModel class.
        target_position:    The 3D coordinates of the target, inherited from the TargetModel class.
        velocity:           The target's velocity (in M/S), inherited from the TargetModel class.
    """

    PRIORITY = 1
    AIR_RESISTANCE_COEF = 0.5
    BALLISTIC_MISSILE_MASS = 5000
    BALLISTIC_MISSILE_AREA = 1.1

    def __init__(self, target_id, target_position, velocity_ms):
        """ Initializes the target model.

        Initialization of the ballistic missile target model class, including properties and methods of the target.
        The cruise missile is divided into active phase, interruption phase and reentry phase. Since the missile range
        is generally long, only part of the trajectory of the reentry phase is considered in this project.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity_ms: Target speed
        """
        super().__init__(target_id, target_position, velocity_ms, "Ballistic_Missile", self.PRIORITY)

    def _calculate_air_resistance_acceleration(self):
        """Calculates the air resistance acceleration of the missile.

        The acceleration of air resistance is approximately calculated by the classical formula of air resistance,
        and the formula of air density is obtained by ISA empirical formula.

        :return: Air resistance acceleration
        """
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0:
            rho = SEA_LEVEL_AIR_DENSITY * math.exp(-self.target_position[2] / ATMOSPHERIC_SCALE_HEIGHT)
            resistance_magnitude = (0.5 * self.AIR_RESISTANCE_COEF * rho * velocity_magnitude * velocity_magnitude
                                    * self.BALLISTIC_MISSILE_AREA / self.BALLISTIC_MISSILE_MASS)
            velocity_direction = -self.velocity / velocity_magnitude
            return resistance_magnitude * velocity_direction
        return np.zeros(3)

    def update_state(self, delta_time):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param delta_time: Time step
        """
        air_resistance = self._calculate_air_resistance_acceleration()
        self.acceleration = GRAVITY + air_resistance

        self.velocity += self.acceleration * delta_time
        self.target_position += self.velocity * delta_time


class CruiseMissileTargetModel(TargetModel):
    """ Cruise Missile target model

    Initialization of the cruise missile target model class, including properties and methods of the target.
    The cruise missile is divided into climbing phase, cruising phase and diving phase. The focus of this project is on
    radar tracking, so only the cruise phase and diving phase of the cruise missile are involved.

    Attributes:
        target_id:          The unique ID of the target, inherited from the TargetModel class.
        target_position:    The 3D coordinates of the target, inherited from the TargetModel class.
        velocity:           The target velocity (in M/S), inherited from the TargetModel class.
        cruise_end_point:   The cruise end point of the cruise missile phase.
        dive_time:          The dive time of the cruise phase.
        cruise_time:        The cruise time of the cruise phase.
        rocket_acceleration:The rocket acceleration of the cruise phase.
    """
    PRIORITY = 2
    CRUISE_ALTITUDE = 8000
    TRANSITION_DISTANCE = 500  # Horizontal distance of the subduction section
    DISTURBANCE_SCALE = 0.8  # Disturbance factor

    def __init__(self, target_id, target_position, velocity, cruise_end_point, dive_time, cruise_time,
                 rocket_acceleration):
        """ Initializes the target model.

        Initialization of the cruise missile target model class, including properties and methods of the target.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity: M/S is the speed in units
        :param cruise_end_point: Cruise end point
        :param dive_time: Dive time
        :param cruise_time: Cruise time
        :param rocket_acceleration: Rocket acceleration in dive phase
        """
        super().__init__(target_id, target_position, velocity, "cruise_missile", self.PRIORITY)
        self.current_phase = "cruise"
        self.cruise_end_point = np.array(cruise_end_point)
        self.dive_time = dive_time
        self.cruise_time = cruise_time
        self.rocket_acceleration = rocket_acceleration
        self.acceleration = np.zeros(3)

    def _apply_cruise_control(self):
        """ Apply cruise control to the cruise phase.

        Cruise missile needs to add disturbance in both cruise and reentry phase to meet the actual operational
        requirements.
        """
        """
        height_error = self.CRUISE_ALTITUDE - self.target_position[2]
        normalized_height_error = np.clip(height_error / 100, -1, 1)
        height_correction = np.array([0, 0, normalized_height_error * self.DISTURBANCE_SCALE])

        horizontal_disturbance = np.random.normal(0, self.DISTURBANCE_SCALE, 2)
        disturbance = np.array([horizontal_disturbance[0], horizontal_disturbance[1], 0])

        return height_correction + disturbance
        """
        height_error = self.CRUISE_ALTITUDE - self.target_position[2]  # Height error calculation
        normalized_height_error = np.clip(height_error / 50, -1, 1)  # Normalized height error, reducing the denominator has increased the response to the error
        time_based_oscillation = np.sin(self.cruise_time * 0.1) * 0.3  # Adding sinusoidal oscillations in the height direction
        height_correction = np.array([0, 0, (normalized_height_error + time_based_oscillation) * self.DISTURBANCE_SCALE])  # Composite altitude correction
        horizontal_disturbance = np.random.normal(0, self.DISTURBANCE_SCALE * 0.7, 2)  # Adding random horizontal disturbances
        disturbance = np.array([horizontal_disturbance[0], horizontal_disturbance[1], 0]) 
        return height_correction + disturbance

    def _apply_dive_control(self):
        """ Apply dive control to the cruise phase.

        Cruise missile needs to add disturbance in both cruise and reentry phase to meet the actual operational
        requirements.
        """
        acceleration = np.random.normal(0, self.DISTURBANCE_SCALE, 3)
        return GRAVITY + self.rocket_acceleration + acceleration

    def _check_phase_transition(self, current_position):
        """ Check if missile should transition from cruise to dive phase

        :param current_position: Current missile position
        :return: True if missile should transition, False otherwise
        """
        horizontal_distance = np.linalg.norm(current_position[:2] - self.cruise_end_point[:2])
        return horizontal_distance <= self.TRANSITION_DISTANCE

    def update_state(self, delta_time):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param delta_time: Time step
        """
        if self.current_phase == "cruise":
            if self._check_phase_transition(self.target_position):
                self.current_phase = "dive"
                dive_control = self._apply_dive_control()
                self.acceleration = dive_control
        else:  # dive phase
            dive_control = self._apply_dive_control()
            self.acceleration = dive_control

        self.velocity += self.acceleration * delta_time
        self.target_position += self.velocity * delta_time


class AircraftTargetModel(TargetModel):
    """Aircraft target model

    Initialization of the aircraft target model class, including properties and methods of the target.

    Attributes:
        target_id:          The unique ID of the aircraft target, inherited from the TargetModel class.
        target_position:    The 3D coordinates of the aircraft target, inherited from the TargetModel class.
        velocity:           The aircraft target velocity (in M/S), inherited from the TargetModel class.

    """
    PRIORITY = 3
    MIN_ALTITUDE = 5000
    MAX_ALTITUDE = 10000
    AIR_RESISTANCE_COEF = 0.1
    TURN_RATE_MAX = 0.1
    VERTICAL_ACCELERATION = 5

    def __init__(self, target_id, target_position, velocity_ms):
        """ Initializes the aircraft target model.

        Initialization of the aircraft target model class, including properties and methods of the target.

        :param target_id: The unique ID of the aircraft target.
        :param target_position: The 3D coordinates of the aircraft target.
        :param velocity_ms: M/S is the speed in units.
        """
        super().__init__(target_id, target_position, velocity_ms, "Aircraft", self.PRIORITY)
        self.min_altitude = self.MIN_ALTITUDE
        self.max_altitude = self.MAX_ALTITUDE
        self.acceleration = np.zeros(3)
        self.yaw = np.random.uniform(0, 2 * np.pi)
        self.pitch = np.random.uniform(-np.pi / 6, np.pi / 6)

    def _calculate_air_resistance(self):
        """ Calculate air resistance.

        The air resistance is calculated based on the current velocity of the target.

        :return: Air resistance
        """
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 0:
            resistance = -self.AIR_RESISTANCE_COEF * velocity_magnitude * self.velocity
            return resistance
        return np.zeros(3)

    def _apply_altitude_control(self):
        """ Apply altitude control to the aircraft target.

        :return: Altitude control acceleration
        """
        height_margin = 200
        height = self.target_position[2]

        if height < self.MIN_ALTITUDE + height_margin:
            return np.array([0, 0, self.VERTICAL_ACCELERATION])
        elif height > self.MAX_ALTITUDE - height_margin:
            return np.array([0, 0, -self.VERTICAL_ACCELERATION])
        return np.zeros(3)

    def _apply_maneuver(self, time_step):
        """ Apply maneuver to the aircraft target.

        :param time_step: Time step
        :return: Maneuver acceleration
        """
        self.yaw += np.random.uniform(-self.TURN_RATE_MAX, self.TURN_RATE_MAX)
        self.pitch += np.random.uniform(-self.TURN_RATE_MAX / 2, self.TURN_RATE_MAX / 2)

        direction = np.array([
            np.cos(self.pitch) * np.cos(self.yaw),
            np.cos(self.pitch) * np.sin(self.yaw),
            np.sin(self.pitch)
        ])

        disturbance = np.array([
            np.random.normal(0, 0.3),  # x方向扰动
            np.random.normal(0, 0.3),  # y方向扰动
            np.random.normal(0, 0.1)  # z方向扰动（较小）
        ]) * time_step  # 扰动与速度和时间步长相关

        return direction, disturbance

    def update_state(self, time_step):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param time_step: Time step
        """
        air_resistance = self._calculate_air_resistance()
        direction, disturbance = self._apply_maneuver(time_step)
        altitude_control = self._apply_altitude_control()

        speed = np.linalg.norm(self.velocity)

        # 分别处理每个加速度分量，避免直接相加
        if speed > 0 and time_step > 0:
            desired_velocity = direction * speed
            velocity_diff = desired_velocity - self.velocity
            maneuver_acceleration = np.nan_to_num(velocity_diff / time_step, nan=0.0, posinf=0.0, neginf=0.0)

            # 分别处理每个加速度分量
            disturbance_acceleration = np.nan_to_num(disturbance / time_step, nan=0.0, posinf=0.0, neginf=0.0)
            air_resistance = np.nan_to_num(air_resistance, nan=0.0, posinf=0.0, neginf=0.0)
            altitude_control = np.nan_to_num(altitude_control, nan=0.0, posinf=0.0, neginf=0.0)

            # 安全地合成总加速度
            self.acceleration = (
                    maneuver_acceleration +
                    altitude_control +
                    air_resistance +
                    disturbance_acceleration
            )
        else:
            self.acceleration = np.nan_to_num(
                altitude_control + air_resistance,
                nan=0.0, posinf=0.0, neginf=0.0
            )

        # 更新速度和位置
        self.velocity += self.acceleration * time_step
        self.target_position += self.velocity * time_step
