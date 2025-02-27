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

MACH_2_MS = 340  # Mach conversion standard unit


class TargetModel:
    """Basic target model

    The base class of the target model, inherited from the three target classes.

    Attributes:
        target_id:      Each target has its own unique ID (int) that distinguishes it from each other.
        target_position: The 3D coordinates of the target at any time, and the specific kinematic equations follow
                            the description in the paper.
        velocity_mach:  Here param_config.yaml takes an input parameter of Mach, which was used in paper to visualize
                            the speeds of the three targets.
        target_type:    It mainly includes three types: ballistic missiles, cruise missiles and fighter jets.
        priority:       Because of their different speeds and functions in combat,
                            they are simply given corresponding priorities. It is mainly divided into three levels:
                            level 1 (the most priority), Level 2 (the second priority), and Level 3 (the lowest level).
    """

    def __init__(self, target_id, target_position, velocity_mach, target_type, priority):
        """ Initializes the target model.

        Initialization of the target model class, including properties and methods of the target.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity_mach: Velocity mach
        :param target_type: Target type
        :param priority: Priority
        """
        self.target_id = target_id
        self.target_position = np.array(target_position, dtype=np.float64)
        self.velocity_mach = np.array(velocity_mach, dtype=np.float64)
        self.velocity_ms = self.velocity_mach * MACH_2_MS
        self.target_type = target_type
        self.priority = priority

    def update_position(self, time_step):
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
            self.velocity_mach,
            self.target_type,
            self.priority]


class BallisticMissileTargetModel(TargetModel):
    """Ballistic Missile target model

    Ballistic missile model class, inherited from TargetModel, whose trajectory is approximately parabolic.

    Attributes:
        target_id:          The unique ID of the target, inherited from the TargetModel class.
        target_position:    The 3D coordinates of the target, inherited from the TargetModel class.
        velocity_mach:      The target's velocity (in Mach), inherited from the TargetModel class.
    """

    PRIORITY = 1
    GRAVITY = np.array([0, 0, 9.81])

    def __init__(self, target_id, target_position, velocity_mach):
        """ Initializes the target model.

        Initialization of the ballistic missile target model class, including properties and methods of the target.
        The cruise missile is divided into active phase, interruption phase and reentry phase. Since the missile range
        is generally long, only part of the trajectory of the reentry phase is considered in this project.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity_mach: Mach is the speed in units
        """
        super().__init__(target_id, target_position, velocity_mach, "Ballistic_Missile", self.PRIORITY)

    def update_position(self, time_step):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param time_step: Time step
        """
        self.velocity_ms = self.velocity_mach * MACH_2_MS
        self.velocity_ms += self.GRAVITY * time_step
        self.target_position += self.velocity_ms * time_step


class CruiseMissileTargetModel(TargetModel):
    """ Cruise Missile target model

    Initialization of the cruise missile target model class, including properties and methods of the target.
    The cruise missile is divided into climbing phase, cruising phase and diving phase. The focus of this project is on
    radar tracking, so only the cruise phase and diving phase of the cruise missile are involved.

    Attributes:
        target_id:          The unique ID of the target, inherited from the TargetModel class.
        target_position:    The 3D coordinates of the target, inherited from the TargetModel class.
        velocity_mach:      The target velocity (in Mach), inherited from the TargetModel class.
        cruise_end_point:   The cruise end point of the cruise missile phase.
        dive_time:          The dive time of the cruise phase.
        cruise_time:        The cruise time of the cruise phase.
        rocket_acceleration:The rocket acceleration of the cruise phase.
    """
    PRIORITY = 2
    CRUISE_ALTITUDE = 8000
    GRAVITY = np.array([0, 0, 9.81])
    TRANSITION_DISTANCE = 500

    def __init__(self, target_id, target_position, velocity_mach, cruise_end_point, dive_time, cruise_time,
                 rocket_acceleration):
        """ Initializes the target model.

        Initialization of the cruise missile target model class, including properties and methods of the target.

        :param target_id: Target ID
        :param target_position: Target position
        :param velocity_mach: Mach is the speed in units
        :param cruise_end_point: Cruise end point
        :param dive_time: Dive time
        :param cruise_time: Cruise time
        """
        super().__init__(target_id, target_position, velocity_mach, "cruise_missile", self.PRIORITY)
        self.current_phase = "cruise"
        self._disturbance_cache = np.zeros(2)
        self.cruise_end_point = np.array(cruise_end_point)
        self.dive_time = dive_time
        self.cruise_time = cruise_time
        self.rocket_acceleration = rocket_acceleration

    def _apply_disturbance(self, time_step):
        """ Apply disturbance to the cruise phase.

        Cruise missile needs to add disturbance in both cruise and reentry phase to meet the actual operational
        requirements.

        :param time_step: Time step
        """
        self._disturbance_cache = np.random.normal(0, 0.8, 2) * time_step
        self.velocity_ms[:2] += self._disturbance_cache
        self.target_position[2] = self.CRUISE_ALTITUDE

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
        self.velocity_ms = self.velocity_mach / MACH_2_MS
        if self.current_phase == "cruise":
            self._apply_disturbance(time_step)
            self.target_position += self.velocity_ms * time_step

            if self._check_phase_transition(self.target_position):
                self.current_phase = "dive"
        elif self.current_phase == "dive":
            total_acceleration = self.GRAVITY + np.array([0, 0, -self.rocket_acceleration])
            self.velocity_ms += total_acceleration * time_step
            self.target_position += self.velocity_ms * time_step


class AircraftTargetModel(TargetModel):
    """Aircraft target model

    Initialization of the aircraft target model class, including properties and methods of the target.

    Attributes:
        target_id:          The unique ID of the aircraft target, inherited from the TargetModel class.
        target_position:    The 3D coordinates of the aircraft target, inherited from the TargetModel class.
        velocity_mach:      The aircraft target velocity (in Mach), inherited from the TargetModel class.

    """
    PRIORITY = 2
    MIN_ALTITUDE = 5000
    MAX_ALTITUDE = 10000

    def __init__(self, target_id, target_position, velocity_mach):
        """ Initializes the aircraft target model.

        Initialization of the aircraft target model class, including properties and methods of the target.

        :param target_id: The unique ID of the aircraft target.
        :param target_position: The 3D coordinates of the aircraft target.
        :param velocity_mach: Mach is the speed in units.
        """
        super().__init__(target_id, target_position, velocity_mach, "Aircraft", self.PRIORITY)
        self.min_altitude = self.MIN_ALTITUDE
        self.max_altitude = self.MAX_ALTITUDE

    def update_position(self, time_step):
        """Updates the target position based on the time step.

        The position coordinates are updated in a linear manner.

        :param time_step: Time step
        """
        self.velocity_ms = self.velocity_mach * MACH_2_MS
        disturbance = np.random.normal(0, 0.5, 3) * time_step
        self.velocity_ms += disturbance

        height_margin = 200
        adjustment_acceleration = 5

        if self.target_position[2] < self.MIN_ALTITUDE + height_margin:
            self.velocity_ms[2] += adjustment_acceleration * time_step
        elif self.target_position[2] > self.MAX_ALTITUDE - height_margin:
            self.velocity_ms[2] -= adjustment_acceleration * time_step

        self.target_position += self.velocity_ms * time_step
