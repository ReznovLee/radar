# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : target_model.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:25
"""
import numpy as np

GRAVITY = np.array([0, 0, -9.81])


class TargetModel:
    """Basic target model

    The base class of the target model, inherited from the three target classes.

    Attributes:
        target_id:      Each target has its own unique ID (int) that distinguishes it from each other.
        target_position: The 3D coordinates of the target at any time, and the specific kinematic equations follow
                            the description in the paper.
        velocity_ms:    Here param_config.yaml takes an input parameter of M/S, which was used in paper to visualize
                            the speeds of the three targets.
        target_type:    It mainly includes three types: ballistic missiles, cruise missiles and fighter jets.
        priority:       Because of their different speeds and functions in combat,
                            they are simply given corresponding priorities. It is mainly divided into three levels:
                            level 1 (the most priority), Level 2 (the second priority), and Level 3 (the lowest level).
    """

    def __init__(self, target_id, target_position, velocity_ms, target_type, priority):
        """ Initializes the target model.

        Initialization of the target model class, including properties and methods of the target.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity_ms: Velocity ms
        :param target_type: Target type
        :param priority: Priority
        """
        self.target_id = target_id
        self.target_position = np.array(target_position, dtype=np.float64)
        self.velocity_ms = np.array(velocity_ms, dtype=np.float64)
        self.target_type = target_type
        self.priority = priority

    def update_state(self, time_step):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param time_step: Time step
        """
        self.target_position += time_step * self.velocity_ms

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
            self.velocity_ms,
            self.target_type,
            self.priority]


class BallisticMissileTargetModel(TargetModel):
    """Ballistic Missile target model

    Ballistic missile model class, inherited from TargetModel, whose trajectory is approximately parabolic.

    Attributes:
        target_id:          The unique ID of the target, inherited from the TargetModel class.
        target_position:    The 3D coordinates of the target, inherited from the TargetModel class.
        velocity_ms:        The target's velocity (in M/S), inherited from the TargetModel class.
    """

    PRIORITY = 1
    AIR_RESISTANCE_COEF = 0.5

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
        self.acceleration = np.zeros(3)

    def _calculate_air_resistance(self):
        """ Calculate air resistance.

        The air resistance is calculated based on the current velocity of the target.

        :return: Air resistance
        """
        velocity_magnitude = np.linalg.norm(self.velocity_ms)
        if velocity_magnitude > 0:
            resistance = -self.AIR_RESISTANCE_COEF * velocity_magnitude
            return resistance
        return np.zeros(3)

    def update_state(self, time_step):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param time_step: Time step
        """
        air_resistance = self._calculate_air_resistance()
        self.acceleration = GRAVITY + air_resistance

        self.velocity_ms += self.acceleration * time_step
        self.target_position += self.velocity_ms * time_step


class CruiseMissileTargetModel(TargetModel):
    """ Cruise Missile target model

    Initialization of the cruise missile target model class, including properties and methods of the target.
    The cruise missile is divided into climbing phase, cruising phase and diving phase. The focus of this project is on
    radar tracking, so only the cruise phase and diving phase of the cruise missile are involved.

    Attributes:
        target_id:          The unique ID of the target, inherited from the TargetModel class.
        target_position:    The 3D coordinates of the target, inherited from the TargetModel class.
        velocity_ms:        The target velocity (in M/S), inherited from the TargetModel class.
        cruise_end_point:   The cruise end point of the cruise missile phase.
        dive_time:          The dive time of the cruise phase.
        cruise_time:        The cruise time of the cruise phase.
        rocket_acceleration:The rocket acceleration of the cruise phase.
    """
    PRIORITY = 2
    CRUISE_ALTITUDE = 8000
    TRANSITION_DISTANCE = 500
    AIR_RESISTANCE_COEF = 0.2
    DISTURBANCE_SCALE = 0.8

    def __init__(self, target_id, target_position, velocity_ms, cruise_end_point, dive_time, cruise_time,
                 rocket_acceleration):
        """ Initializes the target model.

        Initialization of the cruise missile target model class, including properties and methods of the target.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity_ms: M/S is the speed in units
        :param cruise_end_point: Cruise end point
        :param dive_time: Dive time
        :param cruise_time: Cruise time
        """
        super().__init__(target_id, target_position, velocity_ms, "cruise_missile", self.PRIORITY)
        self.current_phase = "cruise"
        self.cruise_end_point = np.array(cruise_end_point)
        self.dive_time = dive_time
        self.cruise_time = cruise_time
        self.rocket_acceleration = rocket_acceleration
        self.acceleration = np.zeros(3)

    def _calculate_air_resistance(self):
        """ Calculate air resistance.

        The air resistance is calculated based on the current velocity of the target.

        :return: Air resistance
        """
        velocity_magnitude = np.linalg.norm(self.velocity_ms)
        if velocity_magnitude > 0:
            resistance = -self.AIR_RESISTANCE_COEF * velocity_magnitude * self.velocity_ms
            return resistance
        return np.zeros(3)

    def _apply_cruise_control(self):
        """ Apply cruise control to the cruise phase.

        Cruise missile needs to add disturbance in both cruise and reentry phase to meet the actual operational
        requirements.
        """
        height_error = self.CRUISE_ALTITUDE - self.target_position[2]
        height_correction = np.array([0, 0, height_error * 0.1])

        horizontal_disturbance = np.random.normal(0, self.DISTURBANCE_SCALE, 2)
        disturbance = np.array([horizontal_disturbance[0], horizontal_disturbance[1], 0])

        return height_correction + disturbance

    def _apply_dive_control(self):
        """ Apply dive control to the cruise phase.

        Cruise missile needs to add disturbance in both cruise and reentry phase to meet the actual operational
        requirements.
        """
        direction_to_target = self.cruise_end_point - self.target_position
        distance = np.linalg.norm(direction_to_target)
        if distance > 0:
            direction_to_target = direction_to_target / distance

        dive_acceleration = (self.rocket_acceleration * direction_to_target)

        return dive_acceleration

    def _check_phase_transition(self, current_position):
        """ Check if missile should transition from cruise to dive phase

        :param current_position: Current missile position
        :return: True if missile should transition, False otherwise
        """
        horizontal_distance = np.linalg.norm(current_position[:2] - self.cruise_end_point[:2])
        return horizontal_distance <= self.TRANSITION_DISTANCE

    def update_position(self, time_step):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param time_step: Time step
        """
        air_resistance = self._calculate_air_resistance()

        if self.current_phase == "cruise":
            control_acceleration = self._apply_cruise_control()
            if self._check_phase_transition(self.target_position):
                self.current_phase = "dive"
        else:  # dive phase
            control_acceleration = self._apply_dive_control()

        self.acceleration = control_acceleration + air_resistance
        self.velocity_ms += self.acceleration * time_step
        self.target_position += self.velocity_ms * time_step


class AircraftTargetModel(TargetModel):
    """Aircraft target model

    Initialization of the aircraft target model class, including properties and methods of the target.

    Attributes:
        target_id:          The unique ID of the aircraft target, inherited from the TargetModel class.
        target_position:    The 3D coordinates of the aircraft target, inherited from the TargetModel class.
        velocity_ms:        The aircraft target velocity (in M/S), inherited from the TargetModel class.

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
        velocity_magnitude = np.linalg.norm(self.velocity_ms)
        if velocity_magnitude > 0:
            resistance = -self.AIR_RESISTANCE_COEF * velocity_magnitude * self.velocity_ms
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

    def update_position(self, time_step):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param time_step: Time step
        """
        air_resistance = self._calculate_air_resistance()
        direction, disturbance = self._apply_maneuver(time_step)
        altitude_control = self._apply_altitude_control()

        speed = np.linalg.norm(self.velocity_ms)

        # 分别处理每个加速度分量，避免直接相加
        if speed > 0 and time_step > 0:
            desired_velocity = direction * speed
            velocity_diff = desired_velocity - self.velocity_ms
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
        self.velocity_ms += self.acceleration * time_step
        self.target_position += self.velocity_ms * time_step
