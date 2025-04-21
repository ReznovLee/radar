#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : kalman_filter.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:25
"""

import numpy as np
from enum import Enum


class MotionModel(Enum):
    """Motion models supported by the Kalman filter"""
    BM = "ballistic_motion"  # Ballistic Missile Motion
    CM_CRUISE = "cruise_missile_cruise"  # Cruise Missile Cruise Phase
    CM_DIVE = "cruise_missile_dive"  # Cruise Missile Dive Phase
    RANDOM = "random_motion"  # Random Motion (Aircraft)


class KalmanFilter:
    """
    Kalman Filter for target state estimation and prediction.
    State vector dimensions vary by motion model:
    - BM: [x, y, z, vx, vy, vz, ax, ay, az]
    - CM_CRUISE/CM_DIVE: [x, y, z, vx, vy, vz, ax, ay, az]
    - RANDOM: [x, y, z, vx, vy, vz, ax, ay, az, jx, jy, jz]
    """

    def __init__(self, motion_model=MotionModel.BM, dt=1.0, process_noise_std=0.1, F=None, H=None, Q=None):
        self.motion_model = motion_model
        self.dt = dt

        # Set dimensions based on motion model
        if motion_model in [MotionModel.BM, MotionModel.CM_CRUISE, MotionModel.CM_DIVE]:
            self.dim_state = 9  # [x, y, z, vx, vy, vz, ax, ay, az]
            self.dim_obs = 3  # [x, y, z]
        elif motion_model == MotionModel.RANDOM:
            self.dim_state = 12  # [x, y, z, vx, vy, vz, ax, ay, az, jx, jy, jz]
            self.dim_obs = 3  # [x, y, z]
        else:
            raise ValueError(f"Unsupported motion model: {motion_model}")

        # Initialize state vector and covariance matrix
        self.x = np.zeros(self.dim_state)
        self.P = np.eye(self.dim_state) * 1e2

        # Initialize history variables
        self._prev_pos = None
        self._prev_vel = None
        self._prev_acc = None

        # Set matrices based on motion model
        self._setup_matrices(F, H, Q, process_noise_std)

    def _setup_matrices(self, F, H, Q, process_noise_std):
        """Setup state transition and measurement matrices based on motion model"""
        if self.motion_model == MotionModel.BM:
            self._setup_ballistic_matrices(F, H, Q, process_noise_std)
        elif self.motion_model == MotionModel.CM_CRUISE:
            self._setup_cruise_matrices(F, H, Q, process_noise_std)
        elif self.motion_model == MotionModel.CM_DIVE:
            self._setup_dive_matrices(F, H, Q, process_noise_std)
        elif self.motion_model == MotionModel.RANDOM:
            self._setup_random_matrices(F, H, Q, process_noise_std)

    def _setup_base_matrices(self, F, H, Q, process_noise_std):
        """Setup basic matrices common to all models"""
        dt = self.dt
        dt2 = dt ** 2 / 2
        dt3 = dt ** 3 / 6

        # State transition matrix
        self.F = F if F is not None else np.eye(self.dim_state)
        self.F[:3, 3:6] = np.eye(3) * dt
        self.F[:3, 6:9] = np.eye(3) * dt2
        self.F[3:6, 6:] = np.eye(3) * dt

        # Measurement matrix
        self.H = H if H is not None else np.zeros((self.dim_obs, self.dim_state))
        self.H[:3, :3] = np.eye(3)

        # Base process noise matrix
        if Q is None:
            q = process_noise_std ** 2
            self.Q = np.zeros((self.dim_state, self.dim_state))
            base_matrix = np.array([[dt3, dt2, dt],
                                    [dt2, dt, 1],
                                    [dt, 1, 1 / dt]])
            for i in range(3):
                if self.dim_state == 9:  # BM, CM_CRUISE, CM_DIVE
                    self.Q[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = q * base_matrix
                elif self.dim_state == 12:  # RANDOM
                    dt4 = dt ** 4 / 24
                    extended_matrix = np.array([[dt4, dt3, dt2, dt],
                                                [dt3, dt2, dt, 1],
                                                [dt2, dt, 1, 0],
                                                [dt, 1, 0, 0]])
                    self.Q[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = q * 10 * extended_matrix
        else:
            self.Q = Q

    def _setup_ballistic_matrices(self, F, H, Q, process_noise_std):
        """Setup matrices for Ballistic Motion model"""
        self._setup_base_matrices(F, H, Q, process_noise_std)
        self.F[5, 8] = self.dt  # Add vertical gravity acceleration
        # Increase vertical uncertainty
        self.Q[2, 2] *= 2
        self.Q[5, 5] *= 2
        self.Q[8, 8] *= 2

    def _setup_cruise_matrices(self, F, H, Q, process_noise_std):
        """Setup matrices for Cruise Missile Cruise Phase"""
        self._setup_base_matrices(F, H, Q, process_noise_std)
        # Increase lateral uncertainty
        self.Q[1, 1] *= 2
        self.Q[4, 4] *= 2
        self.Q[7, 7] *= 2

    def _setup_dive_matrices(self, F, H, Q, process_noise_std):
        """Setup matrices for Cruise Missile Dive Phase"""
        self._setup_base_matrices(F, H, Q, process_noise_std)
        # Increase vertical uncertainty
        self.Q[2, 2] *= 2
        self.Q[5, 5] *= 2
        self.Q[8, 8] *= 2

    def _setup_random_matrices(self, F, H, Q, process_noise_std):
        """Setup matrices for Random Motion (Aircraft)"""
        self._setup_base_matrices(F, H, Q, process_noise_std)

    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """
        Update state estimate based on measurement
        :param z: Measurement vector [x, y, z]
        """
        assert z.shape[0] == self.dim_obs, f"Measurement dimension should be {self.dim_obs}"

        # 标准卡尔曼滤波更新
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.dim_state) - K @ self.H) @ self.P

        # 更新速度和加速度估计
        if self._prev_pos is not None:
            curr_vel = (z - self._prev_pos) / self.dt
            self.x[3:6] = curr_vel

            if self._prev_vel is not None:
                curr_acc = (curr_vel - self._prev_vel) / self.dt
                if self.dim_state >= 9:
                    self.x[6:9] = curr_acc

                if self.dim_state == 12:
                    if self._prev_acc is not None:
                        self.x[9:12] = (curr_acc - self._prev_acc) / self.dt
                    self._prev_acc = curr_acc.copy()

                self._prev_vel = curr_vel.copy()
            else:
                self._prev_vel = curr_vel.copy()

        self._prev_pos = z.copy()
        return self.x.copy()

    def get_state(self):
        """Return current state estimate"""
        return self.x.copy()

    def get_position(self):
        """Return estimated position [x, y, z]"""
        return self.x[:3].copy()

    def get_velocity(self):
        """Return estimated velocity [vx, vy, vz]"""
        return self.x[3:6].copy()

    def get_acceleration(self):
        """Return estimated acceleration [ax, ay, az] if available"""
        if self.dim_state >= 9:
            return self.x[6:9].copy()
        return None

    def get_jerk(self):
        """Return estimated jerk [jx, jy, jz] if available"""
        if self.dim_state == 12:
            return self.x[9:12].copy()
        return None


class IMMFilter:
    """Interactive Multiple Model Filter"""

    def __init__(self, dt=1.0):
        self.dt = dt  # 添加dt属性
        self.filters = {
            MotionModel.BM: KalmanFilter(MotionModel.BM, dt, process_noise_std=0.5),
            MotionModel.CM_CRUISE: KalmanFilter(MotionModel.CM_CRUISE, dt, process_noise_std=0.5),
            MotionModel.CM_DIVE: KalmanFilter(MotionModel.CM_DIVE, dt, process_noise_std=0.5),
            MotionModel.RANDOM: KalmanFilter(MotionModel.RANDOM, dt, process_noise_std=2.0)
        }
        self.model_probs = np.ones(len(self.filters)) / len(self.filters)
        self._prev_pos = None  # 添加位置历史
        self._prev_vel = None  # 添加速度历史
        self.Q = np.eye(3) * 0.1  # 添加过程噪声矩阵

    def update(self, z):
        """Update state estimates using all models"""
        predictions = []
        likelihoods = np.zeros(len(self.filters))
        
        for i, (model, kf) in enumerate(self.filters.items()):
            pred = kf.predict()
            kf.update(z)
            predictions.append(pred)
            
            # 计算似然度
            innovation = z - pred[:3]  # 只比较位置
            innovation_cov = kf.H @ kf.P @ kf.H.T
            likelihood = self._compute_likelihood(innovation, innovation_cov)
            likelihoods[i] = likelihood

        self.model_probs = self._update_model_probabilities(likelihoods)
        return self._combine_estimates()

    @staticmethod
    def _compute_likelihood(innovation, innovation_cov):
        """Compute measurement likelihood for a model"""
        dim = len(innovation)
        det = np.linalg.det(innovation_cov)
        inv_cov = np.linalg.inv(innovation_cov)
        exponent = -0.5 * innovation.T @ inv_cov @ innovation
        likelihood = 1.0 / np.sqrt((2 * np.pi) ** dim * det) * np.exp(exponent)
        return likelihood

    def _update_model_probabilities(self, likelihoods):
        """Update model probabilities based on measurement likelihood"""
        # 使用贝叶斯更新
        c = np.sum(likelihoods * self.model_probs)
        updated_probs = likelihoods * self.model_probs / c
        return updated_probs

    def _combine_estimates(self):
        """Combine state estimates from all models"""
        combined_state = np.zeros(12)  # 使用最大维度
        combined_covariance = np.zeros((12, 12))

        # 组合状态估计
        for (model, kf), prob in zip(self.filters.items(), self.model_probs):
            state = kf.get_state()
            # 填充较小维度的状态向量
            if len(state) < 12:
                state = np.pad(state, (0, 12 - len(state)))
            combined_state += state * prob

        # 组合协方差
        for (model, kf), prob in zip(self.filters.items(), self.model_probs):
            state = kf.get_state()
            state_diff = state - combined_state[:len(state)]
            
            # 扩展协方差矩阵
            P = np.pad(kf.P, ((0, 12 - len(state)), (0, 12 - len(state))))
            
            # 计算交叉项
            spread = np.outer(state_diff, state_diff)
            combined_covariance += prob * (P + spread)

        return combined_state, combined_covariance

    def detect_motion_type(self, z):
        """Detect target motion characteristics and adjust process noise"""
        curr_vel = None
        
        if self._prev_pos is not None and self._prev_vel is not None:
            curr_vel = (z - self._prev_pos) / self.dt
            acc = (curr_vel - self._prev_vel) / self.dt
        
            vel_change = np.linalg.norm(curr_vel - self._prev_vel)
            acc_change = np.linalg.norm(acc)
        
            # 调整过程噪声
            if acc_change > 5.0:  # 高机动
                self.Q *= 2.0
            elif vel_change > 2.0:  # 中等机动
                self.Q *= 1.5
        
        # 更新历史值
        self._prev_pos = z.copy()
        if curr_vel is not None:
            self._prev_vel = curr_vel
