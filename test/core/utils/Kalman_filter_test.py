# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : Kalman_filter_test.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/04/21 17:22
"""
import numpy as np
import matplotlib.pyplot as plt
from src.core.utils.filter import BallisticMissileEKF, CruiseMissileEKF, AircraftIMMEKF


def test_ballistic_missile():
    """测试弹道导弹EKF"""
    # 初始化参数
    dt = 0.1
    t = np.arange(0, 100, dt)
    ekf = BallisticMissileEKF(dt)

    # 生成真实轨迹
    initial_pos = np.array([0., 0., 30000.])  # 初始高度30km
    initial_vel = np.array([300., 0., 100.])  # 初始速度

    true_pos = []
    measured_pos = []
    estimated_pos = []

    pos = initial_pos
    vel = initial_vel

    # 设置观测噪声
    R = np.diag([100., 100., 100.])  # 观测标准差：10m

    for _ in t:
        # 更新真实轨迹
        acc = np.array([0., 0., -9.81])  # 重力加速度
        vel = vel + acc * dt
        pos = pos + vel * dt + 0.5 * acc * dt ** 2

        # 生成带噪声的观测
        z = pos + np.random.multivariate_normal(np.zeros(3), R)

        # EKF估计
        ekf.predict()
        ekf.update(z)

        true_pos.append(pos.copy())
        measured_pos.append(z)
        estimated_pos.append(ekf.x[:3])  # 直接从状态向量中获取位置信息

    return np.array(true_pos), np.array(measured_pos), np.array(estimated_pos)


def test_cruise_missile():
    """测试巡航导弹EKF"""
    dt = 0.1
    t = np.arange(0, 150, dt)
    ekf = CruiseMissileEKF(dt)

    # 初始参数
    initial_pos = np.array([0., 0., 5000.])  # 5km高度
    initial_vel = np.array([250., 0., 0.])  # 巡航速度

    true_pos = []
    measured_pos = []
    estimated_pos = []

    pos = initial_pos
    vel = initial_vel
    phase_changed = False

    R = np.diag([100., 100., 100.])

    for t_val in t:
        # 更新真实轨迹
        if pos[2] < 1000 and not phase_changed:  # 切换到俯冲阶段
            phase_changed = True
            vel = np.array([200., 0., -100.])  # 俯冲速度

        if phase_changed:
            acc = np.array([0., 0., -5.])  # 俯冲加速度
        else:
            acc = np.zeros(3)

        vel = vel + acc * dt
        pos = pos + vel * dt + 0.5 * acc * dt ** 2

        # 生成观测
        z = pos + np.random.multivariate_normal(np.zeros(3), R)

        # EKF估计
        ekf.check_phase(z)  # 检查是否需要切换阶段
        ekf.predict()
        ekf.update(z)

        true_pos.append(pos.copy())
        measured_pos.append(z)
        estimated_pos.append(ekf.x[:3])  # 直接从状态向量中获取位置信息

    return np.array(true_pos), np.array(measured_pos), np.array(estimated_pos)


def test_aircraft():
    """测试飞机IMM-EKF"""
    dt = 0.1
    t = np.arange(0, 200, dt)
    imm = AircraftIMMEKF(dt)

    # 初始参数
    initial_pos = np.array([0., 0., 8000.])  # 8km高度
    initial_vel = np.array([200., 0., 0.])  # 初始速度

    true_pos = []
    measured_pos = []
    estimated_pos = []

    pos = initial_pos
    vel = initial_vel
    turn_rate = 0.0

    R = np.diag([100., 100., 100.])

    for t_val in t:
        # 生成机动轨迹
        if 50 < t_val < 70:  # 转弯
            turn_rate = 0.1
        elif 100 < t_val < 120:  # 加速
            acc = np.array([10., 0., 5.])
            vel = vel + acc * dt
        else:
            turn_rate = 0.0

        # 更新位置
        if turn_rate != 0:
            vel_new = np.array([
                vel[0] * np.cos(turn_rate * dt) - vel[1] * np.sin(turn_rate * dt),
                vel[0] * np.sin(turn_rate * dt) + vel[1] * np.cos(turn_rate * dt),
                vel[2]
            ])
            vel = vel_new

        pos = pos + vel * dt

        # 生成观测
        z = pos + np.random.multivariate_normal(np.zeros(3), R)

        # IMM-EKF估计
        imm.predict()
        estimated_state = imm.update(z)

        true_pos.append(pos.copy())
        measured_pos.append(z)
        estimated_pos.append(estimated_state[0][:3])

    return np.array(true_pos), np.array(measured_pos), np.array(estimated_pos)


def plot_results(true_pos, measured_pos, estimated_pos, title):
    """绘制结果"""
    fig = plt.figure(figsize=(15, 5))

    # 3D轨迹
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], 'b-', label='real trajectories')
    ax1.plot(measured_pos[:, 0], measured_pos[:, 1], measured_pos[:, 2], 'g.', label='Observation trajectories')
    ax1.plot(estimated_pos[:, 0], estimated_pos[:, 1], estimated_pos[:, 2], 'r--', label='Estimation trajectories')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()

    # 高度随时间变化
    ax2 = fig.add_subplot(132)
    t = np.arange(len(true_pos))
    ax2.plot(t, true_pos[:, 2], 'b-', label='real height')
    ax2.plot(t, measured_pos[:, 2], 'g.', label='Observation height')
    ax2.plot(t, estimated_pos[:, 2], 'r--', label='Estimation height')
    ax2.set_xlabel('time step')
    ax2.set_ylabel('height (m)')
    ax2.legend()

    # 误差分析
    ax3 = fig.add_subplot(133)
    error_measured = np.linalg.norm(measured_pos - true_pos, axis=1)
    error_estimated = np.linalg.norm(estimated_pos - true_pos, axis=1)
    ax3.plot(t, error_measured, 'g-', label='Observation Error')
    ax3.plot(t, error_estimated, 'r-', label='Estimation Error')
    ax3.set_xlabel('time step')
    ax3.set_ylabel('error (m)')
    ax3.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    # 测试弹道导弹
    true_pos, measured_pos, estimated_pos = test_ballistic_missile()
    plot_results(true_pos, measured_pos, estimated_pos, 'Ballistic EKF tracking')

    # 测试巡航导弹
    true_pos, measured_pos, estimated_pos = test_cruise_missile()
    plot_results(true_pos, measured_pos, estimated_pos, 'Cruise EKF tracking')

    # 测试飞机
    true_pos, measured_pos, estimated_pos = test_aircraft()
    plot_results(true_pos, measured_pos, estimated_pos, 'Aircraft IMM-EKF tracking')


if __name__ == '__main__':
    main()
