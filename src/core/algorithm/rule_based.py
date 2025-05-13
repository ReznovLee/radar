# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : rule_based.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:24
"""

import numpy as np
import pandas as pd
import os
import logging  # 引入日志记录

from src.core.models.radar_model import RadarNetwork


class RuleBasedScheduler:
    """
    基于规则（就近可用原则）的雷达目标分配模拟器。

    该类模拟随时间推移的雷达目标分配过程。它从CSV文件读取目标轨迹，
    在每个时间步应用测量噪声模拟雷达观测，然后根据“最近邻可用”规则
    将目标分配给雷达网络中的雷达。分配决策基于带有噪声的观测位置。

    Attributes:
        radar_network (RadarNetwork): 雷达网络对象，包含所有雷达及其属性和状态。
        data (pd.DataFrame): 从CSV加载的包含目标轨迹信息的Pandas DataFrame。
        measurement_noise_std (float): 用于模拟雷达观测位置的标准差。
        assignment_history (list): 存储每个时间步分配结果的列表。
                                   每个元素是一个字典，包含 'timestamp' 和 'assignments'。
                                   'assignments' 是一个字典 {target_id: assigned_radar_id or None}。
        timestamps (list): 从数据中提取的唯一且排序的时间戳列表。
    """

    def __init__(self, radar_network, data_dir, data_filename="10-targets.csv", measurement_noise_std=5.0):
        """
        初始化基于规则的调度器。

        Args:
            radar_network (RadarNetwork): 一个配置好的 RadarNetwork 实例。
            data_dir (str): 包含目标轨迹数据的CSV文件所在的目录路径。
                            例如：".\\output\\scenario-2025-04-28"
            data_filename (str): 目录中包含轨迹数据的CSV文件名。
                                 默认为 "target_trajectory.csv"。
                                 CSV应包含 'timestamp', 'id', 'position_x', 'position_y', 'position_z' 等列。
            measurement_noise_std (float): 模拟雷达位置测量误差的标准差（单位与位置坐标一致）。
        """
        if not isinstance(radar_network, RadarNetwork):
            raise TypeError("radar_network 必须是 RadarNetwork 的实例")

        self.radar_network = radar_network
        self.measurement_noise_std = measurement_noise_std
        self.assignment_history = []

        # 构造完整数据文件路径
        data_path = os.path.join(data_dir, data_filename)

        try:
            self.data = pd.read_csv(data_path)
            # 确保必要列存在
            required_cols = ['id', 'timestep', 'position_x', 'position_y', 'position_z']
            if not all(col in self.data.columns for col in required_cols):
                raise ValueError(f"CSV文件 '{data_path}' 必须包含以下列: {required_cols}")
            logging.info(f"成功从 '{data_path}' 加载数据。")
        except FileNotFoundError:
            logging.error(f"无法找到数据文件: {data_path}")
            raise FileNotFoundError(f"无法找到数据文件: {data_path}")
        except Exception as e:
            logging.error(f"加载或验证CSV数据时出错: {e}")
            raise Exception(f"加载或验证CSV数据时出错: {e}")

        # 提取并排序时间戳
        self.timestamps = sorted(self.data['timestep'].unique())
        if not self.timestamps:
            logging.warning(f"数据文件 '{data_path}' 中未找到有效的时间戳。")

    def _add_measurement_noise(self, true_position):
        """
        向真实位置添加高斯噪声以模拟雷达观测误差。

        Args:
            true_position (np.ndarray): 目标的真实三维位置。

        Returns:
            np.ndarray: 带有噪声的观测位置。
        """
        # 确保输入是 numpy 数组
        true_position = np.asarray(true_position, dtype=np.float64)
        # 为每个维度独立生成噪声
        noise = np.random.normal(0, self.measurement_noise_std, size=true_position.shape)
        observed_position = true_position + noise
        # logging.debug(f"True pos: {true_position}, Observed pos: {observed_position}") # 可选的调试日志
        return observed_position

    def _get_targets_at_timestamp(self, timestamp):
        """
        获取指定时间戳的所有目标及其真实位置。

        Args:
            timestamp (float or int): 需要获取目标数据的时间戳。

        Returns:
            list: 一个列表，其中每个元素是一个字典，包含 'id' 和 'true_position'。
                  如果该时间戳无目标，则返回空列表。
        """
        targets_now = self.data[self.data['timestep'] == timestamp]
        target_list = []
        for _, row in targets_now.iterrows():
            # 确保位置数据是数值类型
            try:
                position = np.array([
                    float(row['position_x']),
                    float(row['position_y']),
                    float(row['position_z'])
                ], dtype=np.float64)
                target_list.append({'id': row['id'], 'true_position': position})
            except (ValueError, TypeError) as e:
                logging.warning(
                    f"在时间戳 {timestamp} 处理目标 {row.get('id', 'N/A')} 的位置数据时出错: {e}。跳过此目标。")
                continue  # 跳过数据格式错误的目标
        return target_list

    def run_simulation(self):
        """
        执行整个时间跨度上的雷达目标分配模拟。

        遍历所有时间戳，在每个时间戳执行分配逻辑，并记录结果。

        Returns:
            list: 包含所有时间步分配结果的历史记录列表。
                  格式: [{'timestamp': t1, 'assignments': {tgt_id: rdr_id/None, ...}}, ...]
        """
        self.assignment_history = []  # 清空历史记录以防重复运行

        if not self.timestamps:
            logging.warning("没有时间戳可供模拟。")
            return self.assignment_history

        logging.info("开始规则化调度模拟...")
        for timestamp in self.timestamps:
            # 1. 重置雷达通道状态
            # 假设 RadarNetwork 有 reset_all_channels 方法，该方法会重置其包含的所有 Radar 对象的通道
            try:
                self.radar_network.reset_all_channels()
            except AttributeError:
                logging.error("RadarNetwork 对象缺少 'reset_all_channels' 方法。无法重置通道状态。")
                # 根据需要决定是否在此处停止模拟或继续（可能导致通道一直被占用）
                # raise NotImplementedError("RadarNetwork 需要实现 reset_all_channels 方法")
                pass  # 暂时允许继续，但会有警告

            # 2. 获取当前时间戳的目标及其真实位置
            current_targets = self._get_targets_at_timestamp(timestamp)
            if not current_targets:
                logging.info(f"时间戳 {timestamp}: 没有目标。")
                # 仍然记录一个空分配的时间步
                self.assignment_history.append({
                    'timestamp': timestamp,
                    'assignments': {}
                })
                continue

            assignments_this_step = {}  # 存储当前时间步的分配结果 {target_id: radar_id or None}

            # 3. 遍历目标进行分配
            for target in current_targets:
                target_id = target['id']
                true_position = target['true_position']

                # 3b. 添加观测噪声得到观测位置
                observed_position = self._add_measurement_noise(true_position)

                # 3c. 查找覆盖该 *观测位置* 的雷达
                try:
                    covering_radars = self.radar_network.find_covering_radars(observed_position)
                except AttributeError:
                    logging.error("RadarNetwork 对象缺少 'find_covering_radars' 方法。无法查找覆盖雷达。")
                    covering_radars = []  # 无法查找，假设没有覆盖
                    # raise NotImplementedError("RadarNetwork 需要实现 find_covering_radars 方法")

                assigned_to_radar_id = None  # 默认未分配
                if covering_radars:
                    # 3d. 按距离排序 (基于观测位置到雷达位置的距离)
                    try:
                        # --- 修改点 1：使用 radar_position ---
                        # covering_radars.sort(key=lambda r: np.linalg.norm(observed_position - r.position)) # 原代码
                        covering_radars.sort(key=lambda r: np.linalg.norm(observed_position - r.radar_position))  # 修改后
                        # --- 修改结束 ---
                    except AttributeError as e:
                        logging.error(f"排序覆盖雷达时出错: 雷达对象可能缺少 'radar_position' 属性。 {e}")  # 更新错误消息
                        pass

                    # 3e. 尝试分配给最近的可用雷达
                    for radar in covering_radars:
                        try:
                            # --- 修改点 2：使用 assign_channel ---
                            # if radar.allocate_channel(target_id): # 原代码
                            if radar.assign_channel(target_id):  # 修改后
                                # --- 修改结束 ---
                                assigned_to_radar_id = radar.radar_id  # radar_id 名称是正确的
                                logging.debug(f"时间戳 {timestamp}: 目标 {target_id} 分配给雷达 {assigned_to_radar_id}")
                                break  # 分配成功，处理下一个目标
                        except AttributeError as e:
                            # 更新错误消息以反映实际尝试的方法名
                            logging.error(
                                f"尝试分配通道时出错: 雷达对象可能缺少 'assign_channel' 或 'radar_id' 属性。 {e}")
                            continue
                        except Exception as e:
                            logging.error(
                                f"分配雷达 {getattr(radar, 'radar_id', 'N/A')} 给目标 {target_id} 时发生意外错误: {e}")
                            continue

                # 记录此目标的分配结果
                assignments_this_step[target_id] = assigned_to_radar_id
                if assigned_to_radar_id is None:
                    logging.debug(f"时间戳 {timestamp}: 目标 {target_id} 未能分配到雷达。")

            # 4. 记录当前时间步的完整分配结果
            self.assignment_history.append({
                'timestamp': timestamp,
                'assignments': assignments_this_step
            })
            logging.info(f"时间戳 {timestamp}: 完成分配。 {len(assignments_this_step)} 个目标处理完毕。")

        logging.info(f"规则化调度模拟完成。共处理 {len(self.timestamps)} 个时间步。")
        return self.assignment_history

    def get_assignment_history(self):
        """
        返回模拟过程中记录的完整分配历史。

        Returns:
            list: 包含所有时间步的分配结果的历史记录列表。
        """
        return self.assignment_history
