# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : metrics.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:26
"""

import numpy as np

# The 'targets' is a list of dictionaries, each with at least 'id' and 'priority'.
# The 'assignment_history' is a 2D numpy array or similar structure:
#   shape=(num_targets, num_time_steps)
#   assignment_history[i, t] = radar_id tracking target i at time t (or 0/None if not tracked)
# The 'in_range_history' is a 2D numpy array (boolean):
#   shape=(num_targets, num_time_steps)
#   in_range_history[i, t] = True if target i is in any radar's range at time t


class RadarPerformanceMetrics:
    """ Radar tracking performance evaluation system

    Used to calculate various performance indicators of radar networks in tracking tasks.
    Attribute:
        - targets: Target object list
        - assignment_history: 2D numpy array of radar assignments (target index x time step)
        - in_range_history: 
    """

    def __init__(self,
                 targets,
                 assignment_history: np.ndarray,
                 in_range_history=None,
                 time_step: float = 1.0):
        """
        初始化性能指标计算器。

        Args:
            targets: 目标信息列表，每个字典包含 'id' 和 'priority'。 # Removed type hint from docstring description for consistency, though not strictly necessary
            assignment_history (np.ndarray): 跟踪分配历史记录 (目标索引 x 时间步长)，
                                             值为跟踪雷达ID，0或负值表示未跟踪。
            in_range_history: 目标是否在覆盖范围内的历史记录 (目标索引 x 时间步长)，
                                                     值为布尔值。如果提供，则可以计算跟踪覆盖率。
            time_step (float): 每个时间步长代表的实际时间（秒）。
        """
        if not targets:
            raise ValueError("目标列表不能为空。")

        self.targets = targets
        self.target_map = {target['id']: i for i, target in enumerate(targets)} # Map target ID to index
        self.num_targets = len(targets)
        self.assignment_history = assignment_history
        self.in_range_history = in_range_history
        self.time_step = time_step

        # Basic validation
        if self.assignment_history.shape[0] != self.num_targets:
            raise ValueError(f"分配历史记录的目标数量 ({self.assignment_history.shape[0]}) "
                             f"与目标列表数量 ({self.num_targets}) 不匹配。")
        if self.in_range_history is not None and self.in_range_history.shape != self.assignment_history.shape:
             raise ValueError("在范围历史记录的形状必须与分配历史记录的形状匹配。")

        self.num_time_steps = self.assignment_history.shape[1]
        self.priorities = np.array([t.get('priority', 1.0) for t in self.targets]) # Default priority 1.0 if missing

    def calculate_weighted_tracking_time(self): # Removed: -> float
        """
        计算加权总跟踪时间。

        时间根据目标优先级加权。

        Returns:
            float: 加权总跟踪时间。
        """
        total_weighted_time = 0.0
        # 检查 assignment_history 是否为空或无效
        if self.assignment_history is None or self.assignment_history.size == 0:
            return 0.0

        is_tracked_matrix = self.assignment_history > 0 # Boolean matrix: True if tracked

        for i in range(self.num_targets):
            tracked_steps = np.sum(is_tracked_matrix[i, :])
            total_weighted_time += tracked_steps * self.priorities[i]

        return total_weighted_time * self.time_step

    def calculate_tracking_coverage_ratio(self): # Removed: -> Optional[float]
        """
        计算总体跟踪覆盖率。

        定义为：(所有目标被跟踪的总时间步数) / (所有目标在覆盖范围内的总时间步数)。
        只有在提供了 `in_range_history` 时才能计算。

        Returns:
            float or None: 总体跟踪覆盖率，如果无法计算则返回 None。 # Adjusted docstring return type description
        """
        if self.in_range_history is None:
            print("警告：未提供 'in_range_history'，无法计算跟踪覆盖率。")
            return None
        # 检查 in_range_history 是否为空或无效
        if self.in_range_history.size == 0:
             return 0.0 # Or perhaps None, depending on desired behavior for empty history

        total_tracked_steps = np.sum(self.assignment_history > 0)
        total_in_range_steps = np.sum(self.in_range_history)

        if total_in_range_steps == 0:
            # Handle case where targets are never in range
            return 0.0 if total_tracked_steps == 0 else None # Or raise an error?

        return total_tracked_steps / total_in_range_steps

    def calculate_tracking_handoffs(self): # Removed: -> Dict[int, int]
        """
        计算每个目标的雷达切换次数。

        当跟踪同一个目标的雷达ID在相邻时间步发生变化时，计为一次切换。

        Returns:
            dict: 一个字典，键是目标ID，值是该目标的切换次数。 # Adjusted docstring return type description
        """
        handoffs = {target['id']: 0 for target in self.targets}
        # 检查 assignment_history 是否为空或无效
        if self.assignment_history is None or self.assignment_history.size == 0 or self.num_time_steps < 2:
            return handoffs # No handoffs possible with less than 2 time steps

        for i in range(self.num_targets):
            target_id = self.targets[i]['id']
            target_assignments = self.assignment_history[i, :]
            count = 0
            for t in range(1, self.num_time_steps):
                current_radar = target_assignments[t]
                previous_radar = target_assignments[t-1]

                # Handoff occurs if:
                # 1. Previously tracked (previous_radar > 0)
                # 2. Currently tracked (current_radar > 0)
                # 3. The radar ID changed (current_radar != previous_radar)
                if previous_radar > 0 and current_radar > 0 and current_radar != previous_radar:
                    count += 1
            handoffs[target_id] = count

        return handoffs

    def generate_report(self): # Removed: -> Dict[str, Union[float, Dict, None]]
        """
        生成包含所有计算指标的综合报告。

        Returns:
            dict: 包含各项性能指标的字典。 # Adjusted docstring return type description
        """
        report = {
            "weighted_tracking_time": self.calculate_weighted_tracking_time(),
            "tracking_coverage_ratio": self.calculate_tracking_coverage_ratio(),
            "tracking_handoffs": self.calculate_tracking_handoffs(),
            # 可以根据需要添加来自原 metrics.py 的其他指标，例如：
            # "average_delay": self.calculate_average_delay(...) # 需要 target_entries 数据
            # "resource_utilization": self.calculate_resource_utilization(...) # 需要 radar_network 数据
        }
        return report
