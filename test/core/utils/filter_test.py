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
from src.core.utils.filter import BallisticMissileEKF, CruiseMissileEKF, AircraftIMMEKF


class FilterTester:
    def __init__(self, data_path):
        """初始化测试器
        
        :param data_path: 轨迹数据文件路径
        """
        self.data = pd.read_csv(data_path)
        self.dt = 1.0  # 采样间隔为1秒
        
    def calculate_rmse(self, true_positions, estimated_positions):
        """计算RMSE误差
        
        :param true_positions: 真实轨迹点
        :param estimated_positions: 估计轨迹点
        :return: 位置RMSE误差
        """
        return np.sqrt(np.mean(np.sum((true_positions - estimated_positions) ** 2, axis=1)))
    
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
        
        # 从第二个时刻开始，使用速度差计算加速度
        prev_vel = init_vel
        for i in range(1, len(target_data)):
            # 预测
            ekf.predict()
            
            # 获取当前真实状态用于更新
            curr_pos = target_data.iloc[i][['position_x', 'position_y', 'position_z']].values
            curr_vel = target_data.iloc[i][['velocity_x', 'velocity_y', 'velocity_z']].values
            
            # 更新
            ekf.update(curr_pos)
            
            # 计算真实加速度并更新状态向量中的加速度部分
            curr_acc = (curr_vel - prev_vel) / self.dt
            ekf.x[6:] = curr_acc  # 只更新加速度分量
            
            # 记录结果
            true_positions.append(curr_pos)
            estimated_positions.append(ekf.x[:3])
            
            # 更新前一时刻速度
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
                filter_obj.x = np.array(np.concatenate([init_pos, init_vel, np.zeros(3)]), dtype=np.float64)
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
    
    def plot_results(self, true_pos, est_pos, target_type):
        """绘制跟踪结果
        
        :param true_pos: 真实轨迹
        :param est_pos: 估计轨迹
        :param target_type: 目标类型
        """
        fig = plt.figure(figsize=(15, 5))
        
        # 3D轨迹图
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], 'b-', label='True')
        ax1.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], 'r--', label='Estimated')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'{target_type} Tracking - 3D Trajectory')
        ax1.legend()
        
        # 误差图
        ax2 = fig.add_subplot(122)
        # 确保数据类型为numpy数组并计算误差
        true_pos = np.array(true_pos, dtype=np.float64)
        est_pos = np.array(est_pos, dtype=np.float64)
        errors = np.linalg.norm(true_pos - est_pos, axis=1)
        ax2.plot(errors)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Tracking Error')
        
        plt.tight_layout()
        plt.show()


def main():
    # 初始化测试器
    tester = FilterTester("output/scenario-2025-04-17/10-targets.csv")
    
    # 测试弹道导弹跟踪 (ID=1)
    true_pos_bm, est_pos_bm = tester.test_ballistic_missile(1)
    rmse_bm = tester.calculate_rmse(true_pos_bm, est_pos_bm)
    print(f"Ballistic Missile RMSE: {rmse_bm:.2f} meters")
    tester.plot_results(true_pos_bm, est_pos_bm, "Ballistic Missile")
    
    # 测试巡航导弹跟踪 (ID=2)
    true_pos_cm, est_pos_cm = tester.test_cruise_missile(2)
    rmse_cm = tester.calculate_rmse(true_pos_cm, est_pos_cm)
    print(f"Cruise Missile RMSE: {rmse_cm:.2f} meters")
    tester.plot_results(true_pos_cm, est_pos_cm, "Cruise Missile")
    
    # 测试飞机跟踪 (ID=6)
    true_pos_ac, est_pos_ac = tester.test_aircraft(7)  # 修改为ID=7
    rmse_ac = tester.calculate_rmse(true_pos_ac, est_pos_ac)
    print(f"Aircraft RMSE: {rmse_ac:.2f} meters")
    tester.plot_results(true_pos_ac, est_pos_ac, "Aircraft")


if __name__ == "__main__":
    main()
