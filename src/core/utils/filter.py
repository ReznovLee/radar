# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : filter.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/04/22 10:51
"""
import numpy as np
from enum import Enum
from scipy.linalg import block_diag


class MotionModel(Enum):
    """运动模型类型"""
    CV = "constant_velocity"
    CA = "constant_acceleration"
    CT = "coordinated_turn"
    BM = "ballistic_motion"
    CM_CRUISE = "cruise_missile_cruise"
    CM_DIVE = "cruise_missile_dive"


class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器基类"""

    def __init__(self, dt, state_dim, measurement_dim):
        self.dt = dt
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 100
        self.Q = np.eye(state_dim) * 0.1
        self.R = np.eye(measurement_dim) * 1

    def f(self, x, dt):
        """状态转移函数"""
        raise NotImplementedError

    def h(self, x):
        """观测函数"""
        return x[:3]  # 默认只观测位置

    def Jacobian_F(self, x, dt):
        """状态转移矩阵的雅可比矩阵"""
        raise NotImplementedError

    def Jacobian_H(self, x):
        """观测矩阵的雅可比矩阵"""
        H = np.zeros((3, self.state_dim))
        H[:3, :3] = np.eye(3)
        return H

    def predict(self):
        """预测步骤"""
        self.x = self.f(self.x, self.dt)
        F = self.Jacobian_F(self.x, self.dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """更新步骤"""
        H = self.Jacobian_H(self.x)
        y = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P


class BallisticMissileEKF(ExtendedKalmanFilter):
    """弹道导弹EKF"""

    def __init__(self, dt):
        super().__init__(dt, 9, 3)  # 状态：[x, y, z, vx, vy, vz, ax, ay, az]
        self.air_resistance_coef = 0.1
        self.g = 9.81

    def f(self, x, dt):
        """非线性状态转移"""
        pos = x[:3] + x[3:6] * dt + 0.5 * x[6:9] * dt ** 2
        vel = x[3:6] + x[6:9] * dt

        # 考虑重力和空气阻力的加速度
        v_mag = np.linalg.norm(vel)
        if v_mag > 0:
            air_resistance = -self.air_resistance_coef * v_mag * vel
        else:
            air_resistance = np.zeros(3)

        acc = air_resistance
        acc[2] -= self.g  # 添加重力

        return np.concatenate([pos, vel, acc])

    def Jacobian_F(self, x, dt):
        """状态转移雅可比矩阵"""
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * dt
        F[:3, 6:9] = np.eye(3) * (dt ** 2 / 2)
        F[3:6, 6:9] = np.eye(3) * dt

        # 添加空气阻力的雅可比项
        vel = x[3:6]
        v_mag = np.linalg.norm(vel)
        if v_mag > 0:
            J_air = -self.air_resistance_coef * (np.eye(3) * v_mag +
                                                 np.outer(vel, vel) / v_mag)
            F[6:9, 3:6] = J_air

        return F


class CruiseMissileEKF(ExtendedKalmanFilter):
    """巡航导弹EKF"""

    def __init__(self, dt):
        super().__init__(dt, 9, 3)
        self.phase = "cruise"
        self.height_threshold = 1000  # 切换阈值
        self.dive_angle = np.pi / 4  # 俯冲角

    def f(self, x, dt):
        if self.phase == "cruise":
            # CV模型
            pos = x[:3] + x[3:6] * dt
            vel = x[3:6]
            acc = np.zeros(3)
        else:  # dive phase
            # 带加速度的斜向运动
            pos = x[:3] + x[3:6] * dt + 0.5 * x[6:9] * dt ** 2
            vel = x[3:6] + x[6:9] * dt
            # 保持俯冲角加速度
            acc_mag = np.linalg.norm(x[6:9])
            acc = acc_mag * np.array([
                np.cos(self.dive_angle) * x[3] / np.linalg.norm(x[3:5]),
                np.cos(self.dive_angle) * x[4] / np.linalg.norm(x[3:5]),
                -np.sin(self.dive_angle)
            ])
        return np.concatenate([pos, vel, acc])

    def Jacobian_F(self, x, dt):
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * dt
        if self.phase == "dive":
            F[:3, 6:9] = np.eye(3) * (dt ** 2 / 2)
            F[3:6, 6:9] = np.eye(3) * dt
        return F

    def check_phase(self, z):
        """检查是否需要切换阶段"""
        if z[2] < self.height_threshold and self.phase == "cruise":
            self.phase = "dive"
            # 调整过程噪声
            self.Q[6:9, 6:9] *= 5  # 增加加速度不确定性


class AircraftIMMEKF:
    """飞机IMM-EKF"""

    def __init__(self, dt):
        self.dt = dt
        # 初始化多个模型
        self.filters = {
            MotionModel.CV: self._create_cv_filter(),
            MotionModel.CT: self._create_ct_filter(),
            MotionModel.CA: self._create_ca_filter()
        }
        # 模型转移概率矩阵
        self.transition_matrix = np.array([
            [0.95, 0.025, 0.025],
            [0.025, 0.95, 0.025],
            [0.025, 0.025, 0.95]
        ])
        self.model_probs = np.ones(3) / 3

    def _create_cv_filter(self):
        """匀速运动模型"""
        class CVFilter(ExtendedKalmanFilter):
            def f(self, x, dt):
                # 实现匀速运动模型
                pos = x[:3] + x[3:6] * dt
                vel = x[3:6]
                return np.concatenate([pos, vel])
                
            def Jacobian_F(self, x, dt):
                F = np.eye(6)
                F[:3, 3:6] = np.eye(3) * dt
                return F

        ekf = CVFilter(self.dt, 6, 3)
        ekf.Q = block_diag(
            np.eye(3) * 0.1,  # 位置噪声
            np.eye(3) * 1.0   # 速度噪声
        )
        return ekf

    def _create_ct_filter(self):
        """协调转弯模型"""
        class CTFilter(ExtendedKalmanFilter):
            def __init__(self, dt, state_dim, measurement_dim):
                super().__init__(dt, state_dim, measurement_dim)
                self.turn_rate = 0.1  # 初始转弯角速度

            def f(self, x, dt):
                # 实现协调转弯模型
                pos = x[:3]
                vel = x[3:6]
                
                # 更新位置和速度
                pos_new = pos + vel * dt
                vel_new = np.array([
                    vel[0] * np.cos(self.turn_rate * dt) - vel[1] * np.sin(self.turn_rate * dt),
                    vel[0] * np.sin(self.turn_rate * dt) + vel[1] * np.cos(self.turn_rate * dt),
                    vel[2]
                ])
                return np.concatenate([pos_new, vel_new])
                
            def Jacobian_F(self, x, dt):
                F = np.eye(6)
                F[:3, 3:6] = np.eye(3) * dt
                
                # 速度部分的雅可比矩阵
                omega = self.turn_rate
                F[3:6, 3:6] = np.array([
                    [np.cos(omega * dt), -np.sin(omega * dt), 0],
                    [np.sin(omega * dt), np.cos(omega * dt), 0],
                    [0, 0, 1]
                ])
                return F

        ekf = CTFilter(self.dt, 6, 3)
        ekf.Q = block_diag(
            np.eye(3) * 0.1,  # 位置噪声
            np.eye(3) * 2.0   # 速度噪声（转弯时不确定性更大）
        )
        return ekf

    def _create_ca_filter(self):
        """匀加速模型"""
        class CAFilter(ExtendedKalmanFilter):
            def f(self, x, dt):
                # 实现匀加速运动模型
                pos = x[:3] + x[3:6] * dt + 0.5 * x[6:9] * dt**2
                vel = x[3:6] + x[6:9] * dt
                acc = x[6:9]
                return np.concatenate([pos, vel, acc])
                
            def Jacobian_F(self, x, dt):
                F = np.eye(9)
                F[:3, 3:6] = np.eye(3) * dt
                F[:3, 6:9] = np.eye(3) * (dt**2/2)
                F[3:6, 6:9] = np.eye(3) * dt
                return F

        ekf = CAFilter(self.dt, 9, 3)
        ekf.Q = block_diag(
            np.eye(3) * 0.1,  # 位置噪声
            np.eye(3) * 1.0,  # 速度噪声
            np.eye(3) * 2.0   # 加速度噪声
        )
        return ekf

    def predict(self):
        """IMM预测步骤"""
        # 模型交互
        for i, (model_type, filter) in enumerate(self.filters.items()):
            mixed_mean = np.zeros_like(filter.x)
            mixed_cov = np.zeros_like(filter.P)

            # 混合状态
            for j, (other_type, other_filter) in enumerate(self.filters.items()):
                if j != i:
                    # 将状态向量调整为相同维度
                    if len(other_filter.x) > len(filter.x):
                        # 如果其他模型维度更高，截取相同维度的部分
                        mixed_mean += self.model_probs[j] * other_filter.x[:len(filter.x)]
                    else:
                        # 如果其他模型维度更低，补零
                        padded_state = np.pad(other_filter.x, (0, len(filter.x) - len(other_filter.x)))
                        mixed_mean += self.model_probs[j] * padded_state

            filter.x = mixed_mean
            filter.P = mixed_cov

            # 预测
            filter.predict()

    def update(self, z):
        """IMM更新步骤"""
        likelihoods = np.zeros(len(self.filters))

        # 更新每个模型
        for i, (_, filter) in enumerate(self.filters.items()):
            filter.update(z)

            # 计算似然度
            innovation = z - filter.h(filter.x)
            S = filter.Jacobian_H(filter.x) @ filter.P @ filter.Jacobian_H(filter.x).T + filter.R
            likelihoods[i] = self._compute_likelihood(innovation, S)

        # 更新模型概率
        c = np.sum(likelihoods * self.model_probs)
        self.model_probs = likelihoods * self.model_probs / c

        # 输出组合估计
        return self._combine_estimates()

    def _compute_likelihood(self, innovation, S):
        """计算似然度"""
        n = len(innovation)
        det = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
        exp_term = -0.5 * innovation.T @ inv_S @ innovation
        return 1.0 / np.sqrt((2 * np.pi) ** n * det) * np.exp(exp_term)

    def _combine_estimates(self):
        """组合各模型估计"""
        combined_state = np.zeros(9)  # 使用最大维度
        combined_cov = np.zeros((9, 9))

        for i, (_, filter) in enumerate(self.filters.items()):
            # 填充较小维度的状态向量
            state = np.pad(filter.x, (0, 9 - len(filter.x)))
            combined_state += state * self.model_probs[i]

        return combined_state, combined_cov
