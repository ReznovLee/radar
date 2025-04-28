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
    """
    CV = "constant_velocity"
    CA = "constant_acceleration"
    CT = "coordinated_turn"


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
        
        The state prediction and covariance prediction will be inherited and implemented by different 
        motion models.
        """
        self.x = self.f(self.x, self.dt)
        F = self.Jacobian_F(self.x, self.dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """ Update function

        Update the state and covariance, and calculate the measurement error and Kalman gain based on the
        actual observations. Uses Joseph form for covariance update for better numerical stability.

        :param z: The actual observation vector.
        """
        H = self.Jacobian_H(self.x)
        y = z - self.h(self.x) # Innovation (measurement residual)
        
        # Calculate Innovation Covariance (S) and handle potential singularity
        PHT = self.P @ H.T
        S = H @ PHT + self.R
        try:
            inv_S = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Handle singular S matrix, e.g., by adding small identity matrix
            print(f"Warning: S matrix is singular or near-singular. Adding epsilon.")
            inv_S = np.linalg.inv(S + np.eye(S.shape[0]) * 1e-9)
            
        K = PHT @ inv_S # Kalman Gain

        # Update state estimate
        self.x = self.x + K @ y

        # Update covariance estimate using Joseph form P = (I - KH)P(I - KH)' + KRK'
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Ensure P remains symmetric
        self.P = (self.P + self.P.T) / 2.0


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
        self.air_resistance_coef = 0.01  # 降低空气阻力系数
        self.g = 9.81
        
        # --- 修改点：显著增加加速度的过程噪声 ---
        # 调整过程噪声 Q
        self.Q = np.diag([
            0.1, 0.1, 0.1,  # 位置噪声 (较小)
            1.0, 1.0, 1.0,  # 速度噪声 (中等)
            50.0, 50.0, 50.0 # 加速度噪声 (显著增大，允许滤波器快速调整加速度估计)
        ])
        # --- 修改结束 ---

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
        """State transition Jacobian matrix
        
        The state transfer Jacobian matrix of the BallisticMissileEKF class is inherited 
        from the ExtendedKalmanFilter class, and mainly takes into account the Jacobian 
        term when air resistance exists.
        
        :param x: The state vector.
        :param dt: The time interval between two consecutive measurements.
        """
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * dt
        F[:3, 6:9] = np.eye(3) * (dt ** 2 / 2)
        F[3:6, 6:9] = np.eye(3) * dt

        # Adding the Jacobian term for air resistance
        vel = x[3:6]
        v_mag = np.linalg.norm(vel)
        if v_mag > 0:
            J_air = -self.air_resistance_coef * (np.eye(3) * v_mag +
                                                 np.outer(vel, vel) / v_mag)
            F[6:9, 3:6] = J_air

        return F


class CruiseMissileEKF(ExtendedKalmanFilter):
    """Cruise Missile Extended Kalman Filter Class

    The cruise missile extended Kalman filter model class inherits from the ExtendedKalmanFilter 
    class and rewrites the state transfer function and the Jacobian matrix of the state transfer 
    matrix, and adds a phase determination function to determine which EKF to use.

    Attributes:
        - dt: The time interval between two consecutive measurements.
    """

    def __init__(self, dt):
        """ Class initializer

        The CruiseMissileEKF class initializer inherits from the ExtendedKalmanFilter class 
        and mainly initializes the height threshold and dive angle.

        :param dt: The time interval between two consecutive measurements.
        """
        super().__init__(dt, 9, 3)
        self.phase = "cruise"
        self.g = 9.81
        
        self.height_history = []
        self.window_size = 20  # 滑动窗口大小
        self.decline_threshold = 5  # 连续下降次数阈值

        self.R = np.eye(3) * 0.5 # 增加观测噪声，适应可能的机动

    def f(self, x, dt):
        """ State transfer function.
        
        Rewrite the state transfer equation, inherit from the ExtendedKalmanFilter class, 
        and divide the two motion model classes of the cruise missile.
        
        :param x: The state vector.
        :param dt: The time interval between two consecutive measurements.
        """
        pos = x[:3] + x[3:6] * dt + 0.5 * x[6:9] * dt**2
        vel = x[3:6] + x[6:9] * dt
        acc = x[6:9].copy() # Use the estimated acceleration

        if self.phase == "dive":
            acc[2] -= self.g  # Add gravity effect during dive

        return np.concatenate([pos, vel, acc])

    def Jacobian_F(self, x, dt):
        """ State transition Jacobian matrix

        The Jacobian matrix method of overriding state transfer of the CruiseMissileEKF class 
        is inherited from the ExtendedKalmanFilter class, and solves the Jacobian matrix of 
        position to velocity and position to acceleration in the cruise and dive phases respectively.
        
        :param x: The state vector.
        :param dt: The time interval between two consecutive measurements.
        """
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * dt
        F[:3, 6:9] = np.eye(3) * (0.5 * dt**2)
        F[3:6, 6:9] = np.eye(3) * dt
        return F

    def check_phase(self, z):
        """ Check if a phase switch to dive is required based on altitude trend.

        Uses a sliding window to detect sustained altitude decrease.
        Once switched to dive, it remains in the dive phase.

        :param z: The measurement vector [x, y, z].
        """
        # Only check if currently in cruise phase
        if self.phase == "cruise":
            current_height = z[2]
            self.height_history.append(current_height)

            # Keep the window size fixed
            if len(self.height_history) > self.window_size:
                self.height_history.pop(0)

            # Only make judgments after collecting enough samples
            if len(self.height_history) >= self.window_size:
                # Check for a sustained downward trend
                decline_count = 0
                for i in range(len(self.height_history) - 1):
                    # Check for decrease, allowing for small noise fluctuations
                    if self.height_history[i] > self.height_history[i+1] + 1.0: # Add tolerance
                        decline_count += 1
                    else:
                        # Reset count if not decreasing significantly
                        decline_count = 0 

                # Switch if decline count reaches threshold
                if decline_count >= self.decline_threshold:
                    print(f"Switching to DIVE phase at height: {current_height:.2f}")
                    self.phase = "dive"
                    # Increase process noise for acceleration when diving
                    # to allow the filter to adapt faster to new dynamics (gravity, thrust)
                    q_factor = 10 
                    self.Q[6, 6] *= q_factor
                    self.Q[7, 7] *= q_factor
                    self.Q[8, 8] *= q_factor * 2 # Increase Z-acceleration noise more


class AircraftIMMEKF:
    """ Aircraft Interacting Multiple Model Extended Kalman Filter class
    
    AircraftIMM-EKF class uses IMM-EKF to build aircraft dynamics models, 
    mainly including CV model, CA model and CT model.

    Attribute:
        - dt: The time interval between two consecutive measurements.
    """

    def __init__(self, dt):
        """ Class initializer
        Class initializer, used to declare multiple model classes and state 
        transition probability matrices

        :param dt: The time interval between two consecutive measurements.
        """
        self.dt = dt
        # Initializing multiple models
        self.filters = {
            MotionModel.CV: self._create_cv_filter(),
            MotionModel.CT: self._create_ct_filter(),
            MotionModel.CA: self._create_ca_filter()
        }
        # Model transition probability matrix
        self.transition_matrix = np.array([
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90]
        ])
        self.model_probs = np.ones(len(self.filters)) / len(self.filters)

    def _create_cv_filter(self):
        """ Uniform motion model
        
        Uniform linear motion without considering acceleration.
        """
        class CVFilter(ExtendedKalmanFilter):
            def f(self, x, dt):
                # Implementing a uniform motion model
                pos = x[:3] + x[3:6] * dt
                vel = x[3:6]
                return np.concatenate([pos, vel])
                
            def Jacobian_F(self, x, dt):
                F = np.eye(6)
                F[:3, 3:6] = np.eye(3) * dt
                return F

        ekf = CVFilter(self.dt, 6, 3)
        ekf.Q = block_diag(
            np.eye(3) * 0.5,  # Position noise
            np.eye(3) * 2.0   # Speed ​​noise
        )
        return ekf

    def _create_ct_filter(self):
        """Coordinated Turn Model"""
        class CTFilter(ExtendedKalmanFilter):
            def __init__(self, dt, state_dim, measurement_dim):
                super().__init__(dt, state_dim, measurement_dim)
                self.turn_rate = 0.1  # Initial turning angular velocity

            def f(self, x, dt):
                # Implementing a coordinated turning model
                pos = x[:3]
                vel = x[3:6]
                
                # Update position and velocity
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
                
                # Jacobian matrix of the velocity part
                omega = self.turn_rate
                F[3:6, 3:6] = np.array([
                    [np.cos(omega * dt), -np.sin(omega * dt), 0],
                    [np.sin(omega * dt), np.cos(omega * dt), 0],
                    [0, 0, 1]
                ])
                return F

        ekf = CTFilter(self.dt, 6, 3)
        ekf.Q = block_diag(
            np.eye(3) * 0.5,  # Position noise
            np.diag([10.0, 10.0, 5.0])   # Speed ​​noise (greater uncertainty when turning)
        )
        return ekf

    def _create_ca_filter(self):
        """Uniform acceleration model"""
        class CAFilter(ExtendedKalmanFilter):
            def f(self, x, dt):
                # Implementing a uniformly accelerated motion model
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
            np.eye(3) * 0.5,  # Position noise
            np.eye(3) * 5.0,  # Speed ​​noise
            np.eye(3) * 15.0   # Acceleration noise
        )
        return ekf

    def predict(self):
        """IMM预测方法"""
        num_models = len(self.filters)
        model_keys = list(self.filters.keys())
        
        # 1. 计算混合概率 (c_bar, mu_ij)
        c_bar = np.sum(self.transition_matrix * self.model_probs, axis=1) # shape (num_models,)
        # 防止除零
        c_bar[c_bar == 0] = np.finfo(float).eps 
        mixing_probs = np.zeros((num_models, num_models))
        for i in range(num_models):
            for j in range(num_models):
                mixing_probs[i, j] = (self.transition_matrix[j, i] * self.model_probs[j]) / c_bar[i]

        # 2. 状态和协方差混合
        mixed_states = {}
        mixed_covs = {}

        for i, model_type_i in enumerate(model_keys):
            filter_i = self.filters[model_type_i]
            target_dim = filter_i.state_dim
            
            # 初始化混合状态和协方差
            mixed_x_i = np.zeros(target_dim)
            mixed_P_i = np.zeros((target_dim, target_dim))

            for j, model_type_j in enumerate(model_keys):
                filter_j = self.filters[model_type_j]
                
                # --- 状态混合 ---
                # 处理维度不匹配：截断或填充
                if filter_j.state_dim >= target_dim:
                    state_j = filter_j.x[:target_dim]
                else:
                    # 如果源维度小于目标维度，用零填充（或更复杂的映射）
                    state_j = np.pad(filter_j.x, (0, target_dim - filter_j.state_dim)) 
                mixed_x_i += mixing_probs[i, j] * state_j
            
            mixed_states[model_type_i] = mixed_x_i

            # --- 协方差混合 ---
            for j, model_type_j in enumerate(model_keys):
                filter_j = self.filters[model_type_j]
                
                # 处理维度不匹配的状态和协方差
                if filter_j.state_dim >= target_dim:
                    state_j = filter_j.x[:target_dim]
                    cov_j = filter_j.P[:target_dim, :target_dim]
                else:
                    state_j = np.pad(filter_j.x, (0, target_dim - filter_j.state_dim))
                    # 填充协方差矩阵，对角线填充一个较大的值表示不确定性
                    cov_j = np.eye(target_dim) * 1e6 # Start with high uncertainty for padded dims
                    cov_j[:filter_j.state_dim, :filter_j.state_dim] = filter_j.P 
                
                # 计算状态差
                state_diff = state_j - mixed_x_i
                # 累加混合协方差
                mixed_P_i += mixing_probs[i, j] * (cov_j + np.outer(state_diff, state_diff))

            mixed_covs[model_type_i] = mixed_P_i

        # 3. 更新每个滤波器的状态和协方差，然后执行各自的预测
        for i, model_type in enumerate(model_keys):
            filter = self.filters[model_type]
            filter.x = mixed_states[model_type]
            filter.P = mixed_covs[model_type]
            filter.predict() # 每个滤波器独立预测

    def update(self, z):
        """IMM Update Method"""
        likelihoods = np.zeros(len(self.filters))
        innovations = {} # Store innovations for likelihood calculation
        innovation_covs = {} # Store innovation covariances (S)

        # Update each model using the base class update method
        for i, (model_type, filter_obj) in enumerate(self.filters.items()):
            # Store state before update for likelihood calculation consistency
            x_predict = filter_obj.x.copy() 
            P_predict = filter_obj.P.copy()
            
            # Perform the update using the (now more stable) base class method
            filter_obj.update(z) 

            # Calculate innovation and S based on predicted state/covariance
            H = filter_obj.Jacobian_H(x_predict) # Use Jacobian at predicted state
            innovation = z - filter_obj.h(x_predict) # Innovation based on prediction
            S = H @ P_predict @ H.T + filter_obj.R # S based on predicted P
            
            innovations[model_type] = innovation
            innovation_covs[model_type] = S
            
            # Calculate likelihood using the dedicated method
            likelihoods[i] = self._compute_likelihood(innovation, S)

        # Update model probability - Add epsilon to prevent division by zero
        c = np.sum(likelihoods * self.model_probs)
        # Ensure c is not zero or too small
        c = max(c, np.finfo(float).eps) 
        self.model_probs = (likelihoods * self.model_probs) / c
        # Normalize probabilities again to ensure they sum to 1 after potential epsilon addition
        self.model_probs /= np.sum(self.model_probs) 

        # Output combination estimation
        return self._combine_estimates()

    def _compute_likelihood(self, innovation, S):
        """Calculate likelihood using Gaussian PDF, with stabilization."""
        n = len(innovation)
        try:
            # Add epsilon to determinant calculation for stability
            det_S = np.linalg.det(S)
            # Ensure determinant is positive and non-zero
            if det_S <= 0:
                 print(f"Warning: Non-positive determinant encountered ({det_S}). Using epsilon.")
                 det_S = np.finfo(float).eps

            inv_S = np.linalg.inv(S)
            
            # Calculate exponent term
            maha_dist_sq = innovation.T @ inv_S @ innovation
            # Prevent overflow in exp() for large Mahalanobis distance
            if maha_dist_sq > 700: # exp(700) is already very large
                return np.finfo(float).eps # Return a very small number instead of 0 or NaN

            exp_term = np.exp(-0.5 * maha_dist_sq)
            
            # Calculate normalization factor
            norm_factor = np.sqrt((2 * np.pi) ** n * det_S)
            if norm_factor == 0:
                 print(f"Warning: Zero normalization factor. Using epsilon.")
                 norm_factor = np.finfo(float).eps

            likelihood = exp_term / norm_factor
            # Ensure likelihood is not NaN or inf
            if not np.isfinite(likelihood):
                print(f"Warning: Non-finite likelihood calculated. Returning epsilon.")
                return np.finfo(float).eps
            return max(likelihood, np.finfo(float).eps) # Return at least epsilon

        except np.linalg.LinAlgError:
            # Handle cases where S is singular even after potential fixes
            print(f"Error: Singular matrix S in likelihood calculation. Returning epsilon.")
            return np.finfo(float).eps

    def _combine_estimates(self):
        """Combining model estimates for state and covariance."""
        max_dim = 9 # Assuming CA model has the maximum dimension
        combined_state = np.zeros(max_dim)
        combined_cov = np.zeros((max_dim, max_dim))
        
        # 1. Combine state estimates
        for i, (model_type, filter_obj) in enumerate(self.filters.items()):
            # Pad state vector if necessary
            state_i = filter_obj.x
            if len(state_i) < max_dim:
                state_i = np.pad(state_i, (0, max_dim - len(state_i)))
            combined_state += self.model_probs[i] * state_i

        # 2. Combine covariance estimates
        for i, (model_type, filter_obj) in enumerate(self.filters.items()):
            state_i = filter_obj.x
            cov_i = filter_obj.P
            
            # Pad state and covariance if necessary
            if len(state_i) < max_dim:
                state_i = np.pad(state_i, (0, max_dim - len(state_i)))
                # Pad covariance: Use large diagonal values for uncertainty in padded dimensions
                padded_cov = np.eye(max_dim) * 1e6 
                padded_cov[:cov_i.shape[0], :cov_i.shape[1]] = cov_i
                cov_i = padded_cov

            # Difference between individual model state and combined state
            state_diff = state_i - combined_state
            
            # Add weighted covariance term
            combined_cov += self.model_probs[i] * (cov_i + np.outer(state_diff, state_diff))

        # Ensure combined covariance is symmetric
        combined_cov = (combined_cov + combined_cov.T) / 2.0

        return combined_state, combined_cov
