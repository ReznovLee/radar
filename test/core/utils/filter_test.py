# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : Kalman_filter_test.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/04/24 10:53
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.core.utils.filter import BallisticMissileEKF, CruiseMissileEKF, AircraftIMMEKF, Filter


class FilterTester:
    def __init__(self, data_path):
        """初始化测试器
        
        :param data_path: 轨迹数据文件路径
        """
        self.data = pd.read_csv(data_path)
        self.dt = 1.0  # 采样间隔为1秒
        
    def calculate_rmse(self, true_positions, estimated_positions, skip_initial=5):
        """计算RMSE误差
        
        :param true_positions: 真实轨迹点
        :param estimated_positions: 估计轨迹点
        :return: 位置RMSE误差
        """
        if len(true_positions) <= skip_initial:
            return np.nan # 数据不足以计算
            
        # --- 修改点：在计算前确保数组为数值类型 ---
        try:
            # 截取数据段
            true_pos_segment = np.asarray(true_positions[skip_initial:], dtype=np.float64)
            est_pos_segment = np.asarray(estimated_positions[skip_initial:], dtype=np.float64)
        except ValueError as e:
            print(f"Error converting position data to float64 after skipping {skip_initial} points: {e}")
            # 可以打印出问题数据帮助调试
            # print("Problematic true_positions slice:", true_positions[skip_initial:])
            # print("Problematic estimated_positions slice:", estimated_positions[skip_initial:])
            return np.nan # 如果转换失败，无法计算RMSE
        # --- 修改结束 ---

        # 确保数据类型和维度正确 (现在使用转换后的 segment)
        if true_pos_segment.shape != est_pos_segment.shape or true_pos_segment.ndim != 2:
             print(f"Error: Shape mismatch or incorrect dimensions. True: {true_pos_segment.shape}, Est: {est_pos_segment.shape}")
             return np.nan
        if true_pos_segment.size == 0:
            return np.nan # Avoid division by zero if arrays become empty
        
        # 计算平方误差 (使用转换后的 segment)
        squared_errors = np.sum((true_pos_segment - est_pos_segment) ** 2, axis=1)
        
        # 检查是否有 NaN 或 Inf 值 (现在 squared_errors 应该是 float 类型)
        if np.any(np.isnan(squared_errors)) or np.any(np.isinf(squared_errors)):
            print(f"Warning: NaN or Inf detected in squared errors after skipping {skip_initial} points.")
            # ... (处理 NaN/Inf 的逻辑保持不变) ...
            return np.nan # 简单起见，直接返回 NaN

        return np.sqrt(np.mean(squared_errors))

        # param "skip_initial" is for skip start point, if u need all data, 
        # delete the param and using code down below

        # return np.sqrt(np.mean(np.sum((true_positions - estimated_positions) ** 2, axis=1)))
    
    def test_ballistic_missile(self, target_id):
        """测试弹道导弹EKF
        
        :param target_id: 目标ID
        """
        # 获取目标数据
        target_data = self.data[self.data['id'] == target_id]
        
        # 初始化EKF
        ekf = BallisticMissileEKF(self.dt)
        
        # 第一个时刻：使用零加速度
        init_pos = target_data.iloc[0][['position_x', 'position_y', 'position_z']].values
        init_vel = target_data.iloc[0][['velocity_x', 'velocity_y', 'velocity_z']].values
        init_acc = np.zeros(3)
        ekf.x = np.concatenate([init_pos, init_vel, init_acc])
        
        # 存储结果
        true_positions = []
        estimated_positions = []
        
        # 记录第一个时刻的结果
        measurement = target_data.iloc[0][['position_x', 'position_y', 'position_z']].values
        true_positions.append(measurement)
        estimated_positions.append(ekf.x[:3])
        
        # 从第二个时刻开始
        prev_vel = init_vel
        for i in range(1, len(target_data)):
            # 预测: 得到 x_{k|k-1}
            ekf.predict()
            
            # 获取当前真实状态用于计算加速度和更新
            curr_pos = target_data.iloc[i][['position_x', 'position_y', 'position_z']].values
            curr_vel = target_data.iloc[i][['velocity_x', 'velocity_y', 'velocity_z']].values
            
            # 计算观测到的加速度 (来自区间 [k-1, k])
            curr_acc = (curr_vel - prev_vel) / self.dt
            
            # --- 修改点：在 Update 之前，将观测加速度注入到预测状态中 ---
            # ekf.x 此刻存储的是预测状态 x_{k|k-1}
            ekf.x[6:] = curr_acc
            # ----------------------------------------------------------
            
            # 更新: 使用 curr_pos 和修正后的 x_{k|k-1} 计算 x_{k|k}
            ekf.update(curr_pos)
            
            # 记录结果 (ekf.x 现在是 x_{k|k})
            true_positions.append(curr_pos)
            estimated_positions.append(ekf.x[:3])
            
            # 更新前一时刻速度，为下一次循环计算加速度做准备
            prev_vel = curr_vel
        
        return np.array(true_positions), np.array(estimated_positions)
        
    def test_cruise_missile(self, target_id):
        """测试巡航导弹EKF
        
        :param target_id: 目标ID
        """
        target_data = self.data[self.data['id'] == target_id]
        
        # 初始化EKF，设置合适的高度阈值和俯冲角
        ekf = CruiseMissileEKF(self.dt)
        
        # 初始状态
        init_pos = target_data.iloc[0][['position_x', 'position_y', 'position_z']].values
        init_vel = target_data.iloc[0][['velocity_x', 'velocity_y', 'velocity_z']].values
        init_acc = np.zeros(3) # Start with zero acceleration assumption
        ekf.x = np.concatenate([init_pos, init_vel, init_acc])
        
        true_positions = []
        estimated_positions = []
        phases = []
        
        measurement = target_data.iloc[0][['position_x', 'position_y', 'position_z']].values
        true_positions.append(measurement)
        estimated_positions.append(ekf.x[:3])
        phases.append(ekf.phase)

        for i in range(1, len(target_data)):
            row = target_data.iloc[i]
            ekf.predict()
            measurement = row[['position_x', 'position_y', 'position_z']].values.astype(np.float64)
            ekf.update(measurement)
            ekf.check_phase(measurement)  # 检查是否需要切换阶段

            true_positions.append(measurement)
            estimated_positions.append(ekf.x[:3])
            phases.append(ekf.phase) # Optional: track phase changes

        # Optional: Print phase transition info
        # print(f"Target {target_id} phase history: {phases}")
            
        return np.array(true_positions), np.array(estimated_positions)
    
    def test_aircraft(self, target_id):
        """测试飞机IMM-EKF
        
        :param target_id: 目标ID
        """
        target_data = self.data[self.data['id'] == target_id]
        
        if len(target_data) < 2:
            print(f"警告: 飞机目标 {target_id} 数据点不足，无法计算初始加速度。")
            # 处理数据不足的情况，例如使用零加速度或跳过测试
            init_acc_ca = np.zeros(3) 
        else:
            vel_t0 = target_data.iloc[0][['velocity_x', 'velocity_y', 'velocity_z']].values
            vel_t1 = target_data.iloc[1][['velocity_x', 'velocity_y', 'velocity_z']].values
            init_acc_ca = (vel_t1 - vel_t0) / self.dt

        # 初始化IMM-EKF
        ekf = AircraftIMMEKF(self.dt)
        
        # 初始状态
        init_pos = target_data.iloc[0][['position_x', 'position_y', 'position_z']].values
        init_vel = target_data.iloc[0][['velocity_x', 'velocity_y', 'velocity_z']].values
        
        # 为每个子滤波器设置初始状态
        for filter_name, filter_obj in ekf.filters.items():
            if filter_name in ['CV', 'CT']:  # CV和CT模型是6维状态
                filter_obj.x = np.array(np.concatenate([init_pos, init_vel]), dtype=np.float64)
                filter_obj.P = np.eye(6) * 100  # 6x6协方差矩阵
            elif filter_name == 'CA':  # CA模型是9维状态
                # filter_obj.x = np.array(np.concatenate([init_pos, init_vel, np.zeros(3)]), dtype=np.float64)
                filter_obj.x = np.array(np.concatenate([init_pos, init_vel, init_acc_ca]), dtype=np.float64)
                filter_obj.P = np.eye(9) * 100  # 9x9协方差矩阵
        
        true_positions = []
        estimated_positions = []
        
        for _, row in target_data.iterrows():
            ekf.predict()
            measurement = row[['position_x', 'position_y', 'position_z']].values.astype(np.float64)
            ekf.update(measurement)
            
            # 获取组合后的状态估计
            combined_state, _ = ekf._combine_estimates()  # 使用_combine_estimates方法
            state_estimate = combined_state[:3]  # 提取位置分量
            
            true_positions.append(measurement)
            estimated_positions.append(state_estimate)
            
        return np.array(true_positions), np.array(estimated_positions)
    
    def plot_results(self, true_pos, est_pos, target_type, skip_initial=5):
        """绘制跟踪结果 (忽略初始点)
        
        :param true_pos: 真实轨迹
        :param est_pos: 估计轨迹
        :param target_type: 目标类型
        :param skip_initial: 忽略开头的点数
        """
        # --- 这部分与上次修改一致，用于截断数据 ---
        if len(true_pos) <= skip_initial or len(est_pos) <= skip_initial:
            print(f"Warning: Not enough data points to plot after skipping {skip_initial} for {target_type}.")
            return # 数据不足，不绘图

        try:
            true_pos_plot = np.asarray(true_pos[skip_initial:], dtype=np.float64)
            est_pos_plot = np.asarray(est_pos[skip_initial:], dtype=np.float64)
        except ValueError as e:
             print(f"Error converting position data to float64 for plotting (skipped): {e}")
             return

        if true_pos_plot.size == 0 or est_pos_plot.size == 0 or true_pos_plot.shape != est_pos_plot.shape:
             print(f"Warning: Data is empty or shape mismatch after skipping {skip_initial} for {target_type}. Skipping plot.")
             return
        # --- 截断数据结束 ---

        fig = plt.figure(figsize=(15, 5))
        
        # 3D轨迹图 (使用截断后的数据)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(true_pos_plot[:, 0], true_pos_plot[:, 1], true_pos_plot[:, 2], 'b-', label='True')
        ax1.plot(est_pos_plot[:, 0], est_pos_plot[:, 1], est_pos_plot[:, 2], 'r--', label='Estimated')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'{target_type} Tracking - 3D Trajectory (after step {skip_initial})')
        ax1.legend()
        
        # 误差图 (使用截断后的数据)
        ax2 = fig.add_subplot(122)
        errors = np.linalg.norm(true_pos_plot - est_pos_plot, axis=1)
        time_steps = np.arange(skip_initial, skip_initial + len(errors))
        ax2.plot(time_steps, errors)
        ax2.set_xlabel(f'Time Step (starting from {skip_initial})')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Tracking Error (after convergence)')
        
        plt.tight_layout()
        plt.show()

    # --- 新增方法：绘制完整轨迹 ---
    def plot_results_full(self, true_pos, est_pos, target_type):
        """绘制完整的跟踪结果 (从头到尾)
        
        :param true_pos: 真实轨迹
        :param est_pos: 估计轨迹
        :param target_type: 目标类型
        """
        # 确保数据为 NumPy 数组且类型正确
        try:
            true_pos_plot = np.asarray(true_pos, dtype=np.float64)
            est_pos_plot = np.asarray(est_pos, dtype=np.float64)
        except ValueError as e:
             print(f"Error converting position data to float64 for plotting (full): {e}")
             return

        if true_pos_plot.size == 0 or est_pos_plot.size == 0 or true_pos_plot.shape != est_pos_plot.shape:
             print(f"Warning: Data is empty or shape mismatch for {target_type}. Skipping full plot.")
             return

        fig = plt.figure(figsize=(15, 5))
        
        # 3D轨迹图 (使用完整数据)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(true_pos_plot[:, 0], true_pos_plot[:, 1], true_pos_plot[:, 2], 'b-', label='True')
        ax1.plot(est_pos_plot[:, 0], est_pos_plot[:, 1], est_pos_plot[:, 2], 'r--', label='Estimated')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'{target_type} Tracking - Full 3D Trajectory')
        ax1.legend()
        
        # 误差图 (使用完整数据)
        ax2 = fig.add_subplot(122)
        errors = np.linalg.norm(true_pos_plot - est_pos_plot, axis=1)
        ax2.plot(errors) # X 轴为 0 到 N-1
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Full Tracking Error')
        
        plt.tight_layout()
        plt.show()
    # --- 新增方法结束 ---


def main():
    # 定义统一的 skip_initial 值 (用于 RMSE 计算和跳点绘图)
    SKIP_POINTS_FOR_RMSE_PLOT = 10 

    # 初始化测试器
    tester = FilterTester("output/scenario-2025-05-21/50-targets.csv")

    # --- 测试弹道导弹 ---
    true_pos_bm, est_pos_bm = tester.test_ballistic_missile(1)
    # 计算跳过初始点的 RMSE
    rmse_bm = tester.calculate_rmse(true_pos_bm, est_pos_bm, skip_initial=SKIP_POINTS_FOR_RMSE_PLOT)
    print(f"Ballistic Missile RMSE (after {SKIP_POINTS_FOR_RMSE_PLOT} steps): {rmse_bm:.2f} meters")
    # 绘制跳过初始点的结果
    tester.plot_results(true_pos_bm, est_pos_bm, "Ballistic Missile", skip_initial=SKIP_POINTS_FOR_RMSE_PLOT)
    # (可选) 绘制完整结果
    # tester.plot_results_full(true_pos_bm, est_pos_bm, "Ballistic Missile")
    
    # --- 测试巡航导弹 ---
    true_pos_cm, est_pos_cm = tester.test_cruise_missile(2)
    rmse_cm = tester.calculate_rmse(true_pos_cm, est_pos_cm, skip_initial=SKIP_POINTS_FOR_RMSE_PLOT)
    print(f"Cruise Missile RMSE (after {SKIP_POINTS_FOR_RMSE_PLOT} steps): {rmse_cm:.2f} meters")
    tester.plot_results(true_pos_cm, est_pos_cm, "Cruise Missile", skip_initial=SKIP_POINTS_FOR_RMSE_PLOT)
    # (可选) 绘制完整结果
    # tester.plot_results_full(true_pos_cm, est_pos_cm, "Cruise Missile")
    
    # --- 测试飞机 ---
    true_pos_ac, est_pos_ac = tester.test_aircraft(7)
    rmse_ac = tester.calculate_rmse(true_pos_ac, est_pos_ac, skip_initial=SKIP_POINTS_FOR_RMSE_PLOT)
    print(f"Aircraft RMSE (after {SKIP_POINTS_FOR_RMSE_PLOT} steps): {rmse_ac:.2f} meters")
    tester.plot_results(true_pos_ac, est_pos_ac, "Aircraft", skip_initial=SKIP_POINTS_FOR_RMSE_PLOT)
    # (可选) 绘制完整结果
    # tester.plot_results_full(true_pos_ac, est_pos_ac, "Aircraft")


if __name__ == "__main__":
    main()
    test_param = Filter("output/scenario-2025-05-21/50-targets.csv")
    true_BM, perdit_BM = test_param.test_ballistic_missile(1)
    true_CM, perdit_CM = test_param.test_cruise_missile(2)
    true_AC, perdit_AC = test_param.test_aircraft(7)
    print(perdit_BM)
    print(perdit_CM)
    print(perdit_AC)
    
