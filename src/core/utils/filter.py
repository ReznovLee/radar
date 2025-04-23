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
    """ Motion model

    List all motion models supported by radar.

    Attribute:
        - CV: continuous velocity model
        - CA: continuous acceleration model
        - CT: coordinated turn model
        - BM: Ballistic motion model
        - CM_CRUISE: Cruise phase of cruise missile motion model
        - CM_DIVE: Dive phase of cruise missile motion model
    """
    CV = "constant_velocity"
    CA = "constant_acceleration"
    CT = "coordinated_turn"
    BM = "ballistic_motion"
    CM_CRUISE = "cruise_missile_cruise"
    CM_DIVE = "cruise_missile_dive"


class ExtendedKalmanFilter:
    """ Extended Kalman Filter

    Extended Kalman filter base class, used to define EKF basic properties and methods. 
    Together, these variables form the basic elements of the Extended Kalman Filter, which is used for:
        - State prediction: using dt, x, P, Q
        - State update: using x, P, R
        - Uncertainty propagation: using P, Q, R

    Attributes:
        - dt: The time interval between two consecutive measurements
        - state_dim: The dimension of the system state vector. 
                     The CA model has 9 dimensions, and the other models have 6 dimensions.
        - measurement_dim: Dimensions of the observation vector, used to initialize the measurement noise matrix.
        - x: State vector, Estimate the current state of the storage system.
        - P: The state covariance matrix represents the uncertainty of the state estimate.
        - Q: The process noise covariance matrix represents the uncertainty of the system dynamics model.
        - R: The measurement noise covariance matrix represents the noise level in the measurement process.
    """

    def __init__(self, dt, state_dim, measurement_dim):
        """ Initializes the Extended Kalman Filter

        The extended Kalman filter initializer is used to declare the relevant property values of
        the extended Kalman filter.

        :param dt:
        :param state_dim:
        :param measurement_dim:
        """
        self.dt = dt
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 100  # High uncertainty about the initial state
        self.Q = np.eye(state_dim) * 0.1  # Have a certain degree of confidence in the system dynamics model
        
        # Diagonalizing assumes that the observation noise is independent in each dimension and that
        # there is a moderate amount of trust in the observations.
        self.R = np.eye(measurement_dim) * 1  

    def f(self, x, dt):
        """ State transfer function

        There will be other inherited classes implemented to clarify the target state migration under different motion
        states.

        :param x: The state vector.
        :param dt: The time interval between two consecutive measurements.
        """
        raise NotImplementedError

    def h(self, x):
        """ Observation function

        There will be implementations of other inherited classes to clarify the observation results of EKF on
        target coordinates under different models.

        :param x: The state vector.
        :return: The observation result, which only return coordination of the targets.
        """
        return x[:3]  # 默认只观测位置

    def Jacobian_F(self, x, dt):
        """ Jacobian matrix of the state transfer matrix

        The Jacobian matrix of the state transfer matrix, that is, the partial derivative matrix of 
        the state transfer equation with respect to the state vector, is used to linearize the state 
        transfer equation and transfer the uncertainty of the state estimation. It will be inherited 
        and implemented by other motion models.

        :param x: The state vector.
        :param dt: The time interval between two consecutive measurements.
        """
        raise NotImplementedError

    def Jacobian_H(self, x):
        """ Jacobian matrix of the measurement matrix
        
        The partial derivative matrix of the observation equation with respect to the state vector 
        is used to linearize the observation equation and establish the mapping relationship between 
        the state space and the observation space. It will be inherited and implemented by other motion 
        models.
        
        : param x: The state vector.
        """
        H = np.zeros((3, self.state_dim))
        H[:3, :3] = np.eye(3)
        return H

    def predict(self):
        """ Prediction function
        
        The state prediction and covariance prediction will be inherited and implemented by different motion models.
        
        """
        self.x = self.f(self.x, self.dt)
        F = self.Jacobian_F(self.x, self.dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """ Update function

        Update the state and covariance, and calculate the measurement error and Kalman gain based on the
        actual observations.

        :param z: The actual observation vector.
        """
        H = self.Jacobian_H(self.x)
        y = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P


class BallisticMissileEKF(ExtendedKalmanFilter):
    """ Ballistic Missile extended Kalman Filter class

    The ballistic missile extended Kalman filter model class inherits from the ExtendedKalmanFilter class and
    rewrites the two methods of State transfer function and Jacobian matrix of the state transfer matrix.

    Attributes:
        - dt: The time interval between two consecutive measurements.
    """

    def __init__(self, dt):
        """ Initializes the Ballistic Missile Extended Kalman Filter class

        The BallisticMissileEKF class initializer, inherited from ExtendedKalmanFilter,
        defines a drag coefficient for a ballistic missile.

        :param dt: The time interval between two consecutive measurements.
        """
        super().__init__(dt, 9, 3)  # State: [x, y, z, vx, vy, vz, ax, ay, az]
        self.air_resistance_coef = 0.1
        self.g = 9.81

    def f(self, x, dt):
        """ Nonlinear state transfer

        The state transition equation of a ballistic missile when it receives basic gravity and wind resistance
        facing the cross-section.

        :param x: The state vector.
        :param dt: The time interval between two consecutive measurements.
        """
        pos = x[:3] + x[3:6] * dt + 0.5 * x[6:9] * dt ** 2
        vel = x[3:6] + x[6:9] * dt

        # Acceleration taking into account gravity and air resistance
        v_mag = np.linalg.norm(vel)
        if v_mag > 0:
            air_resistance = -self.air_resistance_coef * v_mag * vel
        else:
            air_resistance = np.zeros(3)

        acc = air_resistance
        acc[2] -= self.g  # Add gravity

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
