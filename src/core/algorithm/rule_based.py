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
from scipy.sparse import csr_matrix
from typing import List, Dict
import logging

from core.models.radar_model import RadarNetwork
from core.utils.constraints import ConstraintChecker


class RuleBasedScheduler:
    """
    基于规则的雷达目标分配算法
    采用就近可用原则进行分配
    """

    def __init__(self, radar_network: RadarNetwork, measurement_noise_std: float = 5.0):
        """
        初始化规则调度器
        
        Args:
            radar_network (RadarNetwork): 雷达网络对象
            measurement_noise_std (float): 测量噪声标准差，用于模拟观测误差
        """
        self.radar_network = radar_network
        self.measurement_noise_std = measurement_noise_std
        self.constraint_checker = ConstraintChecker(radar_network)
        self.radar_ids = list(radar_network.radars.keys())

    def solve(self, targets: List[Dict], observed_targets: List[Dict], t: int) -> csr_matrix:
        """
        执行规则调度算法
        
        Args:
            targets: 真实目标信息（含id、priority、type等）
            observed_targets: 带噪声观测（含id、position、velocity等）
            t: 当前时间步
            
        Returns:
            assignment: 稀疏分配矩阵（行：目标，列：雷达）
        """
        num_targets = len(targets)
        num_radars = len(self.radar_ids)
        assignment = csr_matrix((num_targets, num_radars), dtype=np.int8)

        # 按优先级排序目标
        sorted_targets = sorted(enumerate(zip(targets, observed_targets)), 
                              key=lambda x: x[1][0].get('priority', 1))

        for idx, (target, obs) in sorted_targets:
            pos = np.array(obs['position'])
            # 添加观测噪声
            observed_pos = pos + np.random.normal(0, self.measurement_noise_std, size=pos.shape)
            
            # 找到所有可用且覆盖目标的雷达
            candidate_radars = []
            for j, radar_id in enumerate(self.radar_ids):
                radar = self.radar_network.radars[radar_id]
                if (radar.is_target_in_range(observed_pos) and 
                    self.radar_network.is_radar_available(radar_id)):
                    candidate_radars.append((j, radar))

            if not candidate_radars:
                continue

            # 选择最近的雷达
            best_j = None
            min_dist = float('inf')
            for j, radar in candidate_radars:
                dist = np.linalg.norm(observed_pos - radar.radar_position)
                if dist < min_dist:
                    min_dist = dist
                    best_j = j

            if best_j is not None:
                assignment[idx, best_j] = 1
                
                # 检查约束
                if not self.constraint_checker.verify_all_constraints(
                    assignment, 
                    [obs['position'] for obs in observed_targets]
                )["all_satisfied"]:
                    assignment[idx, best_j] = 0
                    logging.warning(f"目标 {target['id']} 的分配不满足约束，取消分配。")

        return assignment
