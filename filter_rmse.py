# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : filter_rmse.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/05/27 18:22
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.core.utils.filter import BallisticMissileEKF, CruiseMissileEKF, AircraftIMMEKF


class FilterRMSETester:
    def __init__(self, data_path):
        """初始化RMSE测试器
        
        :param data_path: 轨迹数据文件路径
        """
        self.data = pd.read_csv(data_path)
        self.dt = 1.0  # 采样间隔为1秒
        
    def calculate_rmse(self, true_positions, estimated_positions, skip_initial=5):
        """计算RMSE误差
        
        :param true_positions: 真实轨迹点
        :param estimated_positions: 估计轨迹点
        :param skip_initial: 跳过初始点数
        :return: 位置RMSE误差
        """
        if len(true_positions) <= skip_initial:
            return np.nan
            
        try:
            true_pos_segment = np.asarray(true_positions[skip_initial:], dtype=np.float64)
            est_pos_segment = np.asarray(estimated_positions[skip_initial:], dtype=np.float64)
        except ValueError as e:
            print(f"Error converting position data to float64 after skipping {skip_initial} points: {e}")
            return np.nan

        if true_pos_segment.shape != est_pos_segment.shape or true_pos_segment.ndim != 2:
            print(f"Error: Shape mismatch or incorrect dimensions. True: {true_pos_segment.shape}, Est: {est_pos_segment.shape}")
            return np.nan
        if true_pos_segment.size == 0:
            return np.nan
        
        squared_errors = np.sum((true_pos_segment - est_pos_segment) ** 2, axis=1)
        
        if np.any(np.isnan(squared_errors)) or np.any(np.isinf(squared_errors)):
            print(f"Warning: NaN or Inf detected in squared errors after skipping {skip_initial} points.")
            return np.nan

        return np.sqrt(np.mean(squared_errors))
    
    def test_ballistic_missile(self, target_id):
        """测试弹道导弹EKF
        
        :param target_id: 目标ID
        :return: 真实轨迹和估计轨迹
        """
        target_data = self.data[self.data['id'] == target_id]
        
        if len(target_data) == 0:
            print(f"Warning: No data found for Ballistic Missile target {target_id}")
            return None, None
        
        ekf = BallisticMissileEKF(self.dt)
        
        init_pos = target_data.iloc[0][['position_x', 'position_y', 'position_z']].values
        init_vel = target_data.iloc[0][['velocity_x', 'velocity_y', 'velocity_z']].values
        init_acc = np.zeros(3)
        ekf.x = np.concatenate([init_pos, init_vel, init_acc])
        
        true_positions = []
        estimated_positions = []
        
        measurement = target_data.iloc[0][['position_x', 'position_y', 'position_z']].values
        true_positions.append(measurement)
        estimated_positions.append(ekf.x[:3])
        
        prev_vel = init_vel
        for i in range(1, len(target_data)):
            ekf.predict()
            
            curr_pos = target_data.iloc[i][['position_x', 'position_y', 'position_z']].values
            curr_vel = target_data.iloc[i][['velocity_x', 'velocity_y', 'velocity_z']].values
            
            curr_acc = (curr_vel - prev_vel) / self.dt
            ekf.x[6:] = curr_acc
            
            ekf.update(curr_pos)
            
            true_positions.append(curr_pos)
            estimated_positions.append(ekf.x[:3])
            
            prev_vel = curr_vel
        
        return np.array(true_positions), np.array(estimated_positions)
        
    def test_cruise_missile(self, target_id):
        """测试巡航导弹EKF
        
        :param target_id: 目标ID
        :return: 真实轨迹和估计轨迹
        """
        target_data = self.data[self.data['id'] == target_id]
        
        if len(target_data) == 0:
            print(f"Warning: No data found for Cruise Missile target {target_id}")
            return None, None
        
        ekf = CruiseMissileEKF(self.dt)
        
        init_pos = target_data.iloc[0][['position_x', 'position_y', 'position_z']].values
        init_vel = target_data.iloc[0][['velocity_x', 'velocity_y', 'velocity_z']].values
        init_acc = np.zeros(3)
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
            ekf.check_phase(measurement)

            true_positions.append(measurement)
            estimated_positions.append(ekf.x[:3])
            phases.append(ekf.phase)
            
        return np.array(true_positions), np.array(estimated_positions)
    
    def test_aircraft(self, target_id):
        """测试飞机IMM-EKF
        
        :param target_id: 目标ID
        :return: 真实轨迹和估计轨迹
        """
        target_data = self.data[self.data['id'] == target_id]
        
        if len(target_data) == 0:
            print(f"Warning: No data found for Aircraft target {target_id}")
            return None, None
        
        if len(target_data) < 2:
            print(f"Warning: Aircraft target {target_id} has insufficient data points.")
            init_acc_ca = np.zeros(3) 
        else:
            vel_t0 = target_data.iloc[0][['velocity_x', 'velocity_y', 'velocity_z']].values
            vel_t1 = target_data.iloc[1][['velocity_x', 'velocity_y', 'velocity_z']].values
            init_acc_ca = (vel_t1 - vel_t0) / self.dt

        ekf = AircraftIMMEKF(self.dt)
        
        init_pos = target_data.iloc[0][['position_x', 'position_y', 'position_z']].values
        init_vel = target_data.iloc[0][['velocity_x', 'velocity_y', 'velocity_z']].values
        
        for filter_name, filter_obj in ekf.filters.items():
            if filter_name in ['CV', 'CT']:
                filter_obj.x = np.array(np.concatenate([init_pos, init_vel]), dtype=np.float64)
                filter_obj.P = np.eye(6) * 100
            elif filter_name == 'CA':
                filter_obj.x = np.array(np.concatenate([init_pos, init_vel, init_acc_ca]), dtype=np.float64)
                filter_obj.P = np.eye(9) * 100
        
        true_positions = []
        estimated_positions = []
        
        for _, row in target_data.iterrows():
            ekf.predict()
            measurement = row[['position_x', 'position_y', 'position_z']].values.astype(np.float64)
            ekf.update(measurement)
            
            combined_state, _ = ekf._combine_estimates()
            state_estimate = combined_state[:3]
            
            true_positions.append(measurement)
            estimated_positions.append(state_estimate)
            
        return np.array(true_positions), np.array(estimated_positions)
    
    def get_target_ids_by_type_and_range(self, target_type, id_range):
        """根据目标类型和ID范围获取目标ID列表
        
        :param target_type: 目标类型 ('Ballistic_Missile', 'cruise_missile', 'Aircraft')
        :param id_range: ID范围，格式为(start_id, end_id)
        :return: 目标ID列表
        """
        start_id, end_id = id_range
        target_data = self.data[(self.data['target_type'] == target_type) & 
                              (self.data['id'] >= start_id) & 
                              (self.data['id'] <= end_id)]
        unique_ids = target_data['id'].unique()
        
        if len(unique_ids) == 0:
            print(f"Warning: No {target_type} targets found in ID range {start_id}-{end_id}")
        
        return sorted(unique_ids)
    
    def test_rmse_for_all_targets(self):
        """测试指定ID范围的目标的RMSE
        
        :return: RMSE结果字典
        """
        results = {
            'Ballistic_Missile': [],
            'cruise_missile': [],
            'Aircraft': []
        }
        
        # 测试弹道导弹目标（ID 1-50）
        print("Testing Ballistic Missile targets (ID 1-50)...")
        bm_ids = self.get_target_ids_by_type_and_range('Ballistic_Missile', (1, 50))
        for i, target_id in enumerate(bm_ids):
            print(f"Processing Ballistic Missile {i+1}/{len(bm_ids)} (ID: {target_id})")
            true_pos, est_pos = self.test_ballistic_missile(target_id)
            if true_pos is not None and est_pos is not None:
                rmse = self.calculate_rmse(true_pos, est_pos)
                results['Ballistic_Missile'].append((i+1, target_id, rmse))
            else:
                results['Ballistic_Missile'].append((i+1, target_id, np.nan))
        
        # 测试巡航导弹目标（ID 51-100）
        print("Testing Cruise Missile targets (ID 51-100)...")
        cm_ids = self.get_target_ids_by_type_and_range('cruise_missile', (51, 100))
        for i, target_id in enumerate(cm_ids):
            print(f"Processing Cruise Missile {i+1}/{len(cm_ids)} (ID: {target_id})")
            true_pos, est_pos = self.test_cruise_missile(target_id)
            if true_pos is not None and est_pos is not None:
                rmse = self.calculate_rmse(true_pos, est_pos)
                results['cruise_missile'].append((i+1, target_id, rmse))
            else:
                results['cruise_missile'].append((i+1, target_id, np.nan))
        
        # 测试飞机目标（ID 251-300）
        print("Testing Aircraft targets (ID 251-300)...")
        ac_ids = self.get_target_ids_by_type_and_range('Aircraft', (251, 300))
        for i, target_id in enumerate(ac_ids):
            print(f"Processing Aircraft {i+1}/{len(ac_ids)} (ID: {target_id})")
            true_pos, est_pos = self.test_aircraft(target_id)
            if true_pos is not None and est_pos is not None:
                rmse = self.calculate_rmse(true_pos, est_pos)
                results['Aircraft'].append((i+1, target_id, rmse))
            else:
                results['Aircraft'].append((i+1, target_id, np.nan))
        
        return results
    
    def plot_rmse_curves(self, results):
        """绘制RMSE曲线图
        
        :param results: RMSE结果字典
        """
        plt.figure(figsize=(12, 8))
        
        # 设置颜色
        colors = {
            'Ballistic_Missile': 'red',
            'cruise_missile': 'blue', 
            'Aircraft': 'green'
        }
        
        # 设置标签
        labels = {
            'Ballistic_Missile': 'Ballistic Missile',
            'cruise_missile': 'Cruise Missile', 
            'Aircraft': 'Aircraft'
        }
        
        # 绘制每种目标类型的RMSE曲线
        for target_type, target_data in results.items():
            # 过滤掉NaN值
            valid_data = [(idx, target_id, rmse) for idx, target_id, rmse in target_data if not np.isnan(rmse)]
            
            if valid_data:  # 如果有有效数据
                indices, target_ids, rmse_values = zip(*valid_data)
                plt.plot(indices, rmse_values, 
                        color=colors[target_type], 
                        label=labels[target_type],
                        marker='o', 
                        markersize=4,
                        linewidth=2)
                
                # 在部分点上标注目标ID（每10个点标注一次）
                for i in range(0, len(indices), 10):
                    plt.annotate(f"ID:{target_ids[i]}", 
                                (indices[i], rmse_values[i]),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha='center',
                                fontsize=8)
        
        plt.xlabel('Target sequence', fontsize=12)
        plt.ylabel('RMSE (m)', fontsize=12)
        plt.title('Extended Kalman filter RMSE curves for different types of targets', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 设置更好的布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig('/Users/reznovlee/Desktop/git/radar/output/filter_rmse_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print("\n=== RMSE统计信息 ===")
        for target_type, target_data in results.items():
            valid_rmse = [rmse for _, _, rmse in target_data if not np.isnan(rmse)]
            if valid_rmse:
                print(f"{labels[target_type]}:")
                print(f"  平均RMSE: {np.mean(valid_rmse):.2f} 米")
                print(f"  标准差: {np.std(valid_rmse):.2f} 米")
                print(f"  最小RMSE: {np.min(valid_rmse):.2f} 米")
                print(f"  最大RMSE: {np.max(valid_rmse):.2f} 米")
                print(f"  有效目标数: {len(valid_rmse)}/{len(target_data)}")
            else:
                print(f"{labels[target_type]}: 无有效数据")


def main():
    """主函数"""
    # 初始化测试器
    data_path = "/Users/reznovlee/Desktop/git/radar/output/scenario-2025-05-27/500-targets.csv"
    tester = FilterRMSETester(data_path)
    
    print("开始测试指定ID范围目标的RMSE...")
    print("这可能需要几分钟时间...")
    
    # 测试指定ID范围的目标的RMSE
    results = tester.test_rmse_for_all_targets()
    
    # 绘制RMSE曲线图
    tester.plot_rmse_curves(results)
    
    print("\nRMSE测试完成!")
    print("结果已保存至: /Users/reznovlee/Desktop/git/radar/output/filter_rmse_curves.png")


if __name__ == "__main__":
    main()