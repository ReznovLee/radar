# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : bfsa_rho.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:24
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict
from src.core.models.radar_model import RadarNetwork
from src.core.utils.filter import BallisticMissileEKF, CruiseMissileEKF, AircraftIMMEKF
from src.core.utils.metrics import RadarPerformanceMetrics
from src.core.utils.constraints import ConstraintChecker
import logging

class BFSARHO:
    """
    基于滚动时域优化的后向-前向-融合调度算法（BFSA-RHO）
    """

    def __init__(self, radar_network: RadarNetwork, window_size: int = 3, prediction_steps: int = 2):
        self.radar_network = radar_network
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        self.history = []  # 历史分配记录
        self.trackers = {}  # 目标ID -> 滤波器
        self.constraint_checker = ConstraintChecker(radar_network)
        self.radar_ids = list(radar_network.radars.keys())

    def solve(self, targets: List[Dict], observed_targets: List[Dict], t: int) -> csr_matrix:
        """
        输入：
            targets: 真实目标信息（含id、priority、type等）
            observed_targets: 带噪声观测（含id、position、velocity等）
            t: 当前时间步
        输出：
            assignment: 稀疏分配矩阵（行：目标，列：雷达）
        """
        num_targets = len(targets)
        num_radars = len(self.radar_ids)
        # 1. 初始化/第一个时刻：就近分配
        if t == 0 or not self.history:
            assignment = self._nearest_assignment(observed_targets, num_targets, num_radars)
            self.history.append(assignment.copy())
            return assignment

        # 2. Backward阶段：历史加权
        assignment = self._backward_stage(num_targets, num_radars)

        # 3. Forward阶段：滤波预测+动态领域启发式
        assignment = self._forward_stage(assignment, targets, observed_targets, num_targets, num_radars)

        # 4. Fusion阶段：优先级+朝向+动态领域补全
        assignment = self._fusion_stage(assignment, targets, observed_targets, num_targets, num_radars)

        # 5. 记录历史
        if len(self.history) >= self.window_size:
            self.history.pop(0)
        self.history.append(assignment.copy())
        return assignment

    def _nearest_assignment(self, observed_targets, num_targets, num_radars):
        """第一个时刻：就近分配，满足约束"""
        assignment = csr_matrix((num_targets, num_radars), dtype=np.int8)
        for i, obs in enumerate(observed_targets):
            pos = np.array(obs['position'])
            min_dist = float('inf')
            chosen_radar = None
            for j, radar_id in enumerate(self.radar_ids):
                radar = self.radar_network.radars[radar_id]
                dist = np.linalg.norm(pos - radar.radar_position)
                if dist < min_dist and radar.is_target_in_range(pos) and self.radar_network.is_radar_available(radar_id):
                    min_dist = dist
                    chosen_radar = j
            if chosen_radar is not None:
                assignment[i, chosen_radar] = 1
        # 检查约束
        if not self.constraint_checker.verify_all_constraints(assignment, [obs['position'] for obs in observed_targets])["all_satisfied"]:
            logging.warning("初始分配不满足约束，部分目标未分配。")
        return assignment

    def _backward_stage(self, num_targets, num_radars):
        """历史加权分配，缺省部分未分配"""
        window = self.history[-self.window_size:]
        weights = np.array([0.9 ** (len(window) - 1 - i) for i in range(len(window))])
        weights /= weights.sum()
        weighted_matrix = sum(A.multiply(w) for A, w in zip(window, weights))
        # 归一化
        row_sum = weighted_matrix.sum(axis=1).A.ravel()
        row_sum[row_sum == 0] = 1
        normalized = weighted_matrix.multiply(1 / row_sum[:, None])
        assignment = csr_matrix((num_targets, num_radars), dtype=np.int8)
        for i in range(normalized.shape[0]):
            row = normalized.getrow(i).toarray().ravel()
            if np.any(row > 0):
                j = np.argmax(row)
                assignment[i, j] = 1
        return assignment

    def _forward_stage(self, assignment, targets, observed_targets, num_targets, num_radars):
        """滤波预测+动态领域启发式优化"""
        assignment = assignment.copy()
        for i, obs in enumerate(observed_targets):
            target_id = obs['id']
            pos = np.array(obs['position'])
            # 选择滤波器类型
            ttype = next((t['type'] for t in targets if t['id'] == target_id), "aircraft")
            if target_id not in self.trackers:
                if ttype == "ballistic":
                    self.trackers[target_id] = BallisticMissileEKF(dt=1.0)
                elif ttype == "cruise":
                    self.trackers[target_id] = CruiseMissileEKF(dt=1.0)
                else:
                    self.trackers[target_id] = AircraftIMMEKF(dt=1.0)
                self.trackers[target_id].x[:3] = pos
            else:
                self.trackers[target_id].predict()
                self.trackers[target_id].update(pos)
            # 预测未来状态
            pred_pos = self.trackers[target_id].x[:3]
            # 动态领域生成：找覆盖预测位置的所有可用雷达
            candidate_radars = []
            for j, radar_id in enumerate(self.radar_ids):
                radar = self.radar_network.radars[radar_id]
                if radar.is_target_in_range(pred_pos) and self.radar_network.is_radar_available(radar_id):
                    candidate_radars.append(j)
            # 启发式：优先选择与当前分配不同且评价更优的雷达
            current_radar = assignment.getrow(i).nonzero()[1]
            best_j = None
            best_score = -np.inf
            for j in candidate_radars:
                score = self._evaluate_assignment(i, j, assignment, targets)
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is not None and (current_radar.size == 0 or best_j != current_radar[0]):
                assignment[i, :] = 0
                assignment[i, best_j] = 1
            # 检查约束
            if not self.constraint_checker.verify_all_constraints(assignment, [obs['position'] for obs in observed_targets])["all_satisfied"]:
                assignment[i, :] = 0  # 不满足约束则取消分配
        return assignment

    def _fusion_stage(self, assignment, targets, observed_targets, num_targets, num_radars):
        """融合优先级、朝向、动态领域补全未分配目标"""
        assignment = assignment.copy()
        # 先按优先级排序
        sorted_targets = sorted(enumerate(targets), key=lambda x: x[1].get('priority', 1))
        for idx, target in sorted_targets:
            if assignment.getrow(idx).nnz == 0:
                obs = observed_targets[idx]
                pos = np.array(obs['position'])
                # 动态领域：找所有可用且覆盖的雷达
                candidate_radars = []
                for j, radar_id in enumerate(self.radar_ids):
                    radar = self.radar_network.radars[radar_id]
                    if radar.is_target_in_range(pos) and self.radar_network.is_radar_available(radar_id):
                        candidate_radars.append(j)
                # 启发式：结合目标运动朝向（假设velocity可用）和优先级
                best_j = None
                best_score = -np.inf
                for j in candidate_radars:
                    radar = self.radar_network.radars[self.radar_ids[j]]
                    direction_score = self._direction_score(obs, radar)
                    priority_score = 1.0 / (target.get('priority', 1) + 1e-3)
                    score = direction_score + priority_score
                    if score > best_score:
                        best_score = score
                        best_j = j
                if best_j is not None:
                    assignment[idx, best_j] = 1
                # 检查约束
                if not self.constraint_checker.verify_all_constraints(assignment, [obs['position'] for obs in observed_targets])["all_satisfied"]:
                    assignment[idx, :] = 0
        return assignment

    def _evaluate_assignment(self, target_idx, radar_j, assignment, targets):
        """评价函数：可结合优先级、切换次数等"""
        # 这里简单用优先级，实际可结合metrics.py更复杂指标
        priority = targets[target_idx].get('priority', 1)
        return 1.0 / (priority + 1e-3)

    def _direction_score(self, obs, radar):
        """根据目标运动朝向与雷达方向的夹角打分"""
        if 'velocity' not in obs or np.linalg.norm(obs['velocity']) < 1e-3:
            return 0.0
        v = np.array(obs['velocity'])
        to_radar = radar.radar_position - np.array(obs['position'])
        cos_theta = np.dot(v, to_radar) / (np.linalg.norm(v) * np.linalg.norm(to_radar) + 1e-6)
        return cos_theta  # 越接近1越好
        