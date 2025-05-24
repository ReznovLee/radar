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
from scipy import sparse
from typing import List, Dict
from core.models.radar_model import RadarNetwork
from core.utils.filter import BallisticMissileEKF, CruiseMissileEKF, AircraftIMMEKF
from core.utils.constraints import ConstraintChecker
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
        
        # 多目标优化权重
        self.weights = {
            'tracking_continuity': 0.3,  # 跟踪连续性权重
            'switching_cost': 0.3,       # 切换代价权重
            'priority': 0.2,             # 目标优先级权重
            'coverage_quality': 0.1,      # 覆盖质量权重
            'load_balance': 0.1          # 负载均衡权重（新增）
        }
        
        # 约束处理相关参数
        self.constraint_relaxation_levels = [0.0, 0.1, 0.2]  # 约束松弛级别
        self.max_constraint_iterations = 3  # 最大约束处理迭代次数
        
        # 自适应参数
        self.adaptive_params = {
            'enabled': True,                # 是否启用自适应
            'learning_rate': 0.05,          # 学习率
            'min_weight': 0.05,             # 权重最小值
            'max_weight': 0.7,              # 权重最大值
            'performance_window': 5,        # 性能评估窗口大小
            'performance_history': [],      # 性能历史记录
            'weight_history': [],           # 权重历史记录
            'constraint_history': [],       # 约束违反历史
            'assignment_rate_history': [],  # 分配率历史
        }
        
        # 性能指标
        self.performance_metrics = {
            'tracking_rate': 0.0,           # 跟踪率
            'switching_rate': 0.0,          # 切换率
            'constraint_violation_rate': 0.0, # 约束违反率
            'assignment_rate': 0.0,         # 分配率
        }
        
        # 雷达负载相关参数（新增）
        self.radar_loads = {}  # 雷达ID -> 负载值
        self.load_history = []  # 历史负载记录
        self.load_window_size = 5  # 负载历史窗口大小

    def solve(self, targets: List[Dict], observed_targets: List[Dict], t: int) -> sparse.csr_matrix:
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
        
        # 缓存目标优先级
        self._target_priorities = {i: targets[i].get('priority', 1) for i in range(len(targets))}
        
        # 自适应调整参数（如果启用）
        if self.adaptive_params['enabled'] and t > 0 and len(self.history) > 0:
            self._adapt_parameters(targets, observed_targets, t)
        
        # 1. 初始化/第一个时刻：就近分配
        if t == 0 or not self.history:
            assignment = self._nearest_assignment(observed_targets, num_targets, num_radars)
            self.history.append(assignment.copy())
            self._update_radar_loads(assignment)  # 更新雷达负载（新增）
            self._update_performance_metrics(assignment, targets, observed_targets, t)
            return assignment

        # 2. Backward阶段：历史加权
        assignment = self._backward_stage(num_targets, num_radars)

        # 3. Forward阶段：滤波预测+动态领域启发式
        assignment = self._forward_stage(assignment, targets, observed_targets, num_targets, num_radars)

        # 4. Fusion阶段：多目标优化分配
        assignment = self._fusion_stage(assignment, targets, observed_targets, num_targets, num_radars)
        
        # 5. 约束处理：尝试解决约束冲突
        target_positions = [np.array(obs['position']) for obs in observed_targets]
        assignment = self._resolve_constraint_conflicts(assignment, target_positions)

        # 6. 记录历史
        if len(self.history) >= self.window_size:
            self.history.pop(0)
        self.history.append(assignment.copy())
        
        # 7. 更新雷达负载（新增）
        self._update_radar_loads(assignment)
        
        # 8. 更新性能指标
        self._update_performance_metrics(assignment, targets, observed_targets, t)
        
        return assignment

    def _nearest_assignment(self, observed_targets, num_targets, num_radars):
        """第一个时刻：就近分配，满足约束"""
        # 使用lil_matrix进行构建（更高效）
        assignment = sparse.lil_matrix((num_targets, num_radars), dtype=np.int8)
        
        # 按优先级排序处理目标
        target_indices = list(range(len(observed_targets)))
        target_indices.sort(key=lambda i: observed_targets[i].get('priority', 1), reverse=True)
        
        for i in target_indices:
            obs = observed_targets[i]
            pos = np.array(obs['position'])
            min_dist = float('inf')
            chosen_radar = None
            for j, radar_id in enumerate(self.radar_ids):
                radar = self.radar_network.radars[radar_id]
                dist = np.linalg.norm(pos - radar.radar_position)
                if dist < min_dist and radar.is_target_in_range(pos) and self.radar_network.is_radar_available(
                        radar_id):
                    min_dist = dist
                    chosen_radar = j
            if chosen_radar is not None:
                assignment[i, chosen_radar] = 1
                
                # 即时检查约束，如果违反则尝试其他雷达
                temp_assignment = assignment.tocsr()
                target_positions = [np.array(obs['position']) for obs in observed_targets]
                if not self.constraint_checker.verify_all_constraints(temp_assignment, target_positions)["all_satisfied"]:
                    # 尝试其他雷达
                    assignment[i, chosen_radar] = 0
                    for j, radar_id in enumerate(self.radar_ids):
                        if j == chosen_radar:
                            continue
                        
                        radar = self.radar_network.radars[radar_id]
                        if radar.is_target_in_range(pos) and self.radar_network.is_radar_available(radar_id):
                            assignment[i, j] = 1
                            temp_assignment = assignment.tocsr()
                            if self.constraint_checker.verify_all_constraints(temp_assignment, target_positions)["all_satisfied"]:
                                break
                            assignment[i, j] = 0
        
        # 应用渐进式约束松弛
        target_positions = [np.array(obs['position']) for obs in observed_targets]
        assignment = self._resolve_constraint_conflicts(assignment, target_positions)
        
        if assignment.getnnz() < num_targets:
            logging.warning(f"BFSA-RHO: The initial assignment does not satisfy the constraints, {num_targets - assignment.getnnz()} targets are unassigned.")
            
        return assignment.tocsr()

    def _backward_stage(self, num_targets, num_radars):
        """历史加权分配，缺省部分未分配"""
        window = self.history[-self.window_size:]
        # 只保留shape一致的历史分配
        valid_pairs = [(A, w) for A, w in
                       zip(window, np.array([0.9 ** (len(window) - 1 - i) for i in range(len(window))]))
                       if A.shape == (num_targets, num_radars)]
        if not valid_pairs:
            # 没有可用历史，返回全零分配
            return sparse.csr_matrix((num_targets, num_radars), dtype=np.int8)
        weights = np.array([w for _, w in valid_pairs])
        weights /= weights.sum()
        matrices = [A for A, _ in valid_pairs]
        weighted_matrix = sum(A.multiply(w) for A, w in zip(matrices, weights))
        # 归一化
        row_sum = weighted_matrix.sum(axis=1).A.ravel()
        row_sum[row_sum == 0] = 1
        normalized = weighted_matrix.multiply(1 / row_sum[:, None])
        # 使用lil_matrix进行构建（更高效）
        assignment = sparse.lil_matrix((num_targets, num_radars), dtype=np.int8)
        for i in range(num_targets):
            if i >= normalized.shape[0]:
                break
            row = normalized.getrow(i).toarray().ravel()
            if np.any(row > 0):
                j = np.argmax(row)
                assignment[i, j] = 1
        return assignment.tocsr()

    def _forward_stage(self, assignment, targets, observed_targets, num_targets, num_radars):
        """滤波预测+动态领域启发式优化"""
        # 转换为lil_matrix进行修改（更高效）
        assignment = assignment.tolil()
        for i, obs in enumerate(observed_targets):
            target_id = obs['id']
            pos = np.array(obs['position'])
            # 选择滤波器类型
            ttype = next((t['target_type'] for t in targets if t['id'] == target_id), "aircraft")
            if target_id not in self.trackers:
                if ttype == "ballistic":
                    self.trackers[target_id] = BallisticMissileEKF(dt=1.0)
                    self.trackers[target_id].x[:3] = pos
                elif ttype == "cruise":
                    self.trackers[target_id] = CruiseMissileEKF(dt=1.0)
                    self.trackers[target_id].x[:3] = pos
                else:
                    self.trackers[target_id] = AircraftIMMEKF(dt=1.0)
                    for filter_obj in self.trackers[target_id].filters.values():
                        filter_obj.x[:3] = pos
            else:
                self.trackers[target_id].predict()
                self.trackers[target_id].update(pos)
            # 预测未来状态
            if isinstance(self.trackers[target_id], AircraftIMMEKF):
                pred_pos = self.trackers[target_id]._combine_estimates()[0][:3]
            else:
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
                score = self._evaluate_assignment(i, j, assignment.tocsr(), targets)
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is not None and (current_radar.size == 0 or best_j != current_radar[0]):
                assignment[i, :] = 0
                assignment[i, best_j] = 1
            
            # 不立即检查约束，而是在所有目标处理完后统一处理
        
        # 在所有目标处理完后，统一应用约束解决
        target_positions = [np.array(obs['position']) for obs in observed_targets]
        assignment = self._resolve_constraint_conflicts(assignment, target_positions)
        
        return assignment.tocsr()

    def _fusion_stage(self, assignment, targets, observed_targets, num_targets, num_radars):
        """多目标优化融合阶段"""
        # 转换为lil_matrix进行修改（更高效）
        assignment = assignment.tolil()
        
        # 计算目标优先级权重
        priority_weights = np.ones(num_targets)
        for i, target in enumerate(targets):
            if i < num_targets:
                priority_weights[i] = target.get('priority', 1)
        priority_weights = priority_weights / np.sum(priority_weights) if np.sum(priority_weights) > 0 else priority_weights
        
        # 构建多目标优化问题
        # 1. 先按优先级排序处理未分配目标
        unassigned_targets = []
        for i in range(num_targets):
            if assignment.getrow(i).nnz == 0:
                unassigned_targets.append(i)
        
        # 按优先级排序
        unassigned_targets.sort(key=lambda i: targets[i].get('priority', 1), reverse=True)
        
        # 2. 为每个未分配目标寻找最佳雷达
        for target_idx in unassigned_targets:
            if target_idx >= len(observed_targets):
                continue
                
            obs = observed_targets[target_idx]
            pos = np.array(obs['position'])
            
            # 找出所有可用且覆盖的雷达
            candidate_radars = []
            for j, radar_id in enumerate(self.radar_ids):
                radar = self.radar_network.radars[radar_id]
                if radar.is_target_in_range(pos) and self.radar_network.is_radar_available(radar_id):
                    candidate_radars.append(j)
            
            if not candidate_radars:
                continue
                
            # 计算多目标评分
            best_j = None
            best_score = -np.inf
            
            for j in candidate_radars:
                # 临时分配以评估
                temp_assignment = assignment.copy()
                temp_assignment[target_idx, j] = 1
                
                # 多目标评分
                scores = {}
                
                # 跟踪连续性评分
                scores['tracking_continuity'] = self._tracking_continuity_score(target_idx, j)
                
                # 切换代价评分
                scores['switching_cost'] = self._switching_cost_score(target_idx, j)
                
                # 优先级评分
                scores['priority'] = priority_weights[target_idx] if target_idx < len(priority_weights) else 0
                
                # 覆盖质量评分
                scores['coverage_quality'] = self._coverage_quality_score(obs, j)
                
                # 加权总分
                total_score = sum(self.weights[k] * v for k, v in scores.items())
                
                if total_score > best_score:
                    best_score = total_score
                    best_j = j
            
            # 应用最佳分配
            if best_j is not None:
                assignment[target_idx, best_j] = 1
                
                # 检查约束
                assignment_csr = assignment.tocsr()
                if not self.constraint_checker.verify_all_constraints(
                    assignment_csr, [obs['position'] for obs in observed_targets]
                )["all_satisfied"]:
                    assignment[target_idx, :] = 0  # 不满足约束则取消分配
        
        # 3. 优化已分配目标的雷达分配
        assigned_targets = []
        for i in range(num_targets):
            if assignment.getrow(i).nnz > 0:
                assigned_targets.append(i)
        
        # 尝试改进已分配目标的雷达分配
        for target_idx in assigned_targets:
            if target_idx >= len(observed_targets):
                continue
                
            obs = observed_targets[target_idx]
            pos = np.array(obs['position'])
            current_j = assignment.getrow(target_idx).nonzero()[1][0]
            
            # 找出所有可用且覆盖的雷达
            candidate_radars = []
            for j, radar_id in enumerate(self.radar_ids):
                if j == current_j:
                    continue  # 跳过当前分配的雷达
                    
                radar = self.radar_network.radars[radar_id]
                if radar.is_target_in_range(pos) and self.radar_network.is_radar_available(radar_id):
                    candidate_radars.append(j)
            
            if not candidate_radars:
                continue
                
            # 计算当前分配的多目标评分
            current_scores = {}
            current_scores['tracking_continuity'] = self._tracking_continuity_score(target_idx, current_j)
            current_scores['switching_cost'] = self._switching_cost_score(target_idx, current_j)
            current_scores['priority'] = priority_weights[target_idx] if target_idx < len(priority_weights) else 0
            current_scores['coverage_quality'] = self._coverage_quality_score(obs, current_j)
            current_total_score = sum(self.weights[k] * v for k, v in current_scores.items())
            
            # 寻找更好的分配
            best_j = None
            best_score = current_total_score
            
            for j in candidate_radars:
                # 计算新分配的多目标评分
                new_scores = {}
                new_scores['tracking_continuity'] = self._tracking_continuity_score(target_idx, j)
                new_scores['switching_cost'] = self._switching_cost_score(target_idx, j)
                new_scores['priority'] = priority_weights[target_idx] if target_idx < len(priority_weights) else 0
                new_scores['coverage_quality'] = self._coverage_quality_score(obs, j)
                new_total_score = sum(self.weights[k] * v for k, v in new_scores.items())
                
                if new_total_score > best_score:
                    best_score = new_total_score
                    best_j = j
            
            # 应用更好的分配
            if best_j is not None:
                assignment[target_idx, current_j] = 0
                assignment[target_idx, best_j] = 1
                
                # 检查约束
                assignment_csr = assignment.tocsr()
                if not self.constraint_checker.verify_all_constraints(
                    assignment_csr, [obs['position'] for obs in observed_targets]
                )["all_satisfied"]:
                    # 恢复原分配
                    assignment[target_idx, best_j] = 0
                    assignment[target_idx, current_j] = 1
        
        return assignment.tocsr()

    def _resolve_constraint_conflicts(self, assignment, target_positions):
        """
        尝试解决约束冲突，使用渐进式约束松弛
        
        Args:
            assignment: 分配矩阵（lil_matrix或csr_matrix）
            target_positions: 目标位置列表
            
        Returns:
            处理后的分配矩阵（csr_matrix）
        """
        # 确保使用lil_matrix进行修改（更高效）
        if not sparse.isspmatrix_lil(assignment):
            assignment = assignment.tolil()
            
        # 检查约束
        csr_assignment = assignment.tocsr()
        constraints_result = self.constraint_checker.verify_all_constraints(csr_assignment, target_positions)
        
        # 如果已满足所有约束，直接返回
        if constraints_result["all_satisfied"]:
            return csr_assignment
            
        # 尝试不同的约束松弛级别
        for iteration in range(self.max_constraint_iterations):
            # 处理C14约束（雷达通道数限制）
            if constraints_result["C14"]:
                for radar_id in constraints_result["C14"]:
                    radar_idx = self.radar_ids.index(radar_id)
                    # 找出分配给该雷达的所有目标
                    target_indices = csr_assignment.getcol(radar_idx).nonzero()[0]
                    # 按优先级排序
                    target_indices = sorted(target_indices, 
                                          key=lambda i: self._target_priorities.get(i, 1) if i < len(self._target_priorities) else 1,
                                          reverse=True)
                    # 获取雷达通道数
                    radar = self.radar_network.radars[radar_id]
                    # 移除超出通道数的低优先级目标分配
                    for i in target_indices[radar.num_channels:]:
                        assignment[i, radar_idx] = 0
            
            # 处理C15约束（目标超出雷达覆盖范围）
            if constraints_result["C15"]:
                for target_idx in constraints_result["C15"]:
                    if target_idx < assignment.shape[0]:
                        # 找出分配给该目标的雷达
                        radar_indices = assignment.getrow(target_idx).nonzero()[1]
                        for j in radar_indices:
                            assignment[target_idx, j] = 0
            
            # 重新检查约束
            csr_assignment = assignment.tocsr()
            constraints_result = self.constraint_checker.verify_all_constraints(csr_assignment, target_positions)
            
            # 如果已满足所有约束，返回结果
            if constraints_result["all_satisfied"]:
                return csr_assignment
                
            # 应用约束松弛（如果有）
            relaxation_level = self.constraint_relaxation_levels[min(iteration, len(self.constraint_relaxation_levels)-1)]
            if relaxation_level > 0:
                logging.info(f"应用约束松弛级别 {relaxation_level}")
                # 这里可以实现具体的松弛策略，例如允许某些约束的轻微违反
                # 当前实现中暂不处理松弛
        
        # 如果经过多次迭代仍无法满足所有约束，返回最后一次尝试的结果
        logging.warning("无法满足所有约束，返回部分满足约束的分配结果")
        return csr_assignment

    def _update_radar_loads(self, assignment):
        """
        更新雷达负载信息
        
        Args:
            assignment: 当前分配矩阵
        """
        # 计算当前分配下每个雷达的负载
        current_loads = {}
        for j, radar_id in enumerate(self.radar_ids):
            radar = self.radar_network.radars[radar_id]
            # 计算分配给该雷达的目标数量
            assigned_targets = assignment.getcol(j).getnnz()
            # 计算负载比例（相对于通道数）
            load_ratio = assigned_targets / max(1, radar.num_channels)
            current_loads[radar_id] = load_ratio
        
        # 更新负载历史
        self.load_history.append(current_loads)
        if len(self.load_history) > self.load_window_size:
            self.load_history.pop(0)
        
        # 计算平均负载（考虑历史）
        self.radar_loads = {}
        for radar_id in self.radar_ids:
            loads = [h.get(radar_id, 0) for h in self.load_history]
            self.radar_loads[radar_id] = sum(loads) / len(loads) if loads else 0

    def _tracking_continuity_score(self, target_idx, radar_j):
        """
        计算跟踪连续性评分
        
        Args:
            target_idx: 目标索引
            radar_j: 雷达索引
            
        Returns:
            跟踪连续性评分 [0,1]
        """
        # 如果没有历史记录，返回中性评分
        if not self.history:
            return 0.5
            
        # 获取最近的历史分配
        last_assignment = self.history[-1]
        
        # 检查目标索引是否在历史分配范围内
        if target_idx >= last_assignment.shape[0]:
            return 0.5
            
        # 获取上一次分配的雷达
        last_radar = last_assignment.getrow(target_idx).nonzero()[1]
        
        # 如果上一次未分配，返回中性评分
        if last_radar.size == 0:
            return 0.5
            
        # 如果与上一次分配相同，返回高评分
        if last_radar[0] == radar_j:
            return 1.0
            
        # 否则返回低评分
        return 0.2

    def _switching_cost_score(self, target_idx, radar_j):
        """
        计算切换代价评分
        
        Args:
            target_idx: 目标索引
            radar_j: 雷达索引
            
        Returns:
            切换代价评分 [0,1]，值越高表示切换代价越低
        """
        # 如果没有历史记录，返回中性评分
        if not self.history:
            return 0.5
            
        # 获取最近的历史分配
        last_assignment = self.history[-1]
        
        # 检查目标索引是否在历史分配范围内
        if target_idx >= last_assignment.shape[0]:
            return 0.5
            
        # 获取上一次分配的雷达
        last_radar = last_assignment.getrow(target_idx).nonzero()[1]
        
        # 如果上一次未分配，返回中性评分
        if last_radar.size == 0:
            return 0.5
            
        # 如果与上一次分配相同，返回高评分（无切换代价）
        if last_radar[0] == radar_j:
            return 1.0
            
        # 否则返回低评分（有切换代价）
        return 0.3

    def _coverage_quality_score(self, target, radar_j):
        """
        计算覆盖质量评分
        
        Args:
            target: 目标信息字典
            radar_j: 雷达索引
            
        Returns:
            覆盖质量评分 [0,1]
        """
        if 'position' not in target:
            return 0.5
            
        pos = np.array(target['position'])
        radar_id = self.radar_ids[radar_j]
        radar = self.radar_network.radars[radar_id]
        
        # 计算目标到雷达的距离
        dist = np.linalg.norm(pos - radar.radar_position)
        
        # 如果目标不在雷达覆盖范围内，返回0
        if not radar.is_target_in_range(pos):
            return 0.0
            
        # 计算归一化距离（越近越好）
        # 尝试获取雷达的最大范围属性
        # 根据雷达模型的实际实现，可能是以下属性之一
        if hasattr(radar, 'detection_range'):
            max_range = radar.detection_range
        elif hasattr(radar, 'max_detection_range'):
            max_range = radar.max_detection_range
        elif hasattr(radar, 'range'):
            max_range = radar.range
        else:
            # 如果找不到合适的属性，使用默认值
            max_range = 100000  # 默认值，单位可能是米
            
        norm_dist = 1.0 - min(1.0, dist / max_range)
        
        return norm_dist

    def _load_balance_score(self, radar_j):
        """
        计算负载均衡评分，负载越低分数越高
        
        Args:
            radar_j: 雷达索引
            
        Returns:
            负载均衡评分 [0,1]
        """
        if not self.radar_loads:
            return 1.0
            
        radar_id = self.radar_ids[radar_j]
        current_load = self.radar_loads.get(radar_id, 0)
        
        # 计算所有雷达的平均负载
        avg_load = sum(self.radar_loads.values()) / max(1, len(self.radar_loads))
        
        # 如果当前雷达负载低于平均值，给予更高评分
        if current_load <= avg_load:
            return 1.0 - (current_load / max(1, avg_load * 2))
        else:
            # 负载高于平均值，评分降低
            return max(0.1, 1.0 - (current_load / max(1, avg_load)))

    def _evaluate_assignment(self, target_idx, radar_j, assignment, targets):
        """
        综合评价函数：最大化跟踪时间，最小化切换，最大化跟踪率
        """
        scores = {}
        
        # 跟踪连续性评分
        scores['tracking_continuity'] = self._tracking_continuity_score(target_idx, radar_j)
        
        # 切换代价评分
        scores['switching_cost'] = self._switching_cost_score(target_idx, radar_j)
        
        # 优先级评分
        target_priority = self._get_target_priority(target_idx)
        scores['priority'] = target_priority
        
        # 覆盖质量评分
        if target_idx < assignment.shape[0]:
            target_pos = None
            for i, t in enumerate(targets):
                if i == target_idx:
                    target_pos = t.get('position')
                    break
            # 修复：检查 target_pos 是否为 None，而不是直接将其用作条件
            if target_pos is not None:
                scores['coverage_quality'] = self._coverage_quality_score({'position': target_pos}, radar_j)
            else:
                scores['coverage_quality'] = 0.5
        else:
            scores['coverage_quality'] = 0.5
            
        # 负载均衡评分
        scores['load_balance'] = self._load_balance_score(radar_j)
        
        # 加权总分
        total_score = sum(self.weights[k] * v for k, v in scores.items())
        
        return total_score

    def _find_alternative_radar_with_load_balance(self, assignment, target_idx, target_position):
        """
        为被移除分配的目标寻找负载较低的替代雷达
        
        Args:
            assignment: 当前分配矩阵
            target_idx: 目标索引
            target_position: 目标位置
        
        Returns:
            是否成功找到替代雷达
        """
        # 修复：检查 target_position 是否为 None，而不是直接将其用作条件
        if target_position is None:
            return False
            
        # 找出所有可用且覆盖的雷达
        candidate_radars = []
        for j, radar_id in enumerate(self.radar_ids):
            radar = self.radar_network.radars[radar_id]
            if radar.is_target_in_range(target_position) and self.radar_network.is_radar_available(radar_id):
                # 计算负载评分
                load_score = self._load_balance_score(j)
                candidate_radars.append((j, load_score))
        
        if not candidate_radars:
            return False
            
        # 按负载评分降序排序（优先选择负载低的雷达）
        candidate_radars.sort(key=lambda x: x[1], reverse=True)
        
        # 尝试每个候选雷达
        for j, _ in candidate_radars:
            assignment[target_idx, j] = 1
            
            # 检查约束
            temp_assignment = assignment.tocsr()
            if self.constraint_checker.verify_radar_constraints(temp_assignment, target_position, self.radar_ids[j])["all_satisfied"]:
                return True
                
            assignment[target_idx, j] = 0
        
        return False

    def _adapt_parameters(self, targets, observed_targets, t):
        """
        自适应调整算法参数
        
        根据历史性能指标动态调整权重，以提高算法性能
        
        Args:
            targets: 目标列表
            observed_targets: 观测目标列表
            t: 当前时间步
        """
        # 如果历史记录不足，无法进行自适应调整
        if len(self.adaptive_params['performance_history']) < 2:
            return
            
        # 获取最近的性能指标
        recent_metrics = self.adaptive_params['performance_history'][-1]
        
        # 计算权重调整方向
        adjustments = {
            'tracking_continuity': 0.0,
            'switching_cost': 0.0,
            'priority': 0.0,
            'coverage_quality': 0.0,
            'load_balance': 0.0  # 新增负载均衡权重调整
        }
        
        # 根据跟踪率调整跟踪连续性权重
        if recent_metrics['tracking_rate'] < 0.7:
            adjustments['tracking_continuity'] = 0.02
        elif recent_metrics['tracking_rate'] > 0.9:
            adjustments['tracking_continuity'] = -0.01
            
        # 根据切换率调整切换代价权重
        if recent_metrics['switching_rate'] > 0.3:
            adjustments['switching_cost'] = 0.02
        elif recent_metrics['switching_rate'] < 0.1:
            adjustments['switching_cost'] = -0.01
            
        # 根据约束违反率调整优先级权重
        if recent_metrics['constraint_violation_rate'] > 0:
            adjustments['priority'] = -0.01
            adjustments['coverage_quality'] = 0.02
            
        # 根据分配率调整覆盖质量权重
        if recent_metrics['assignment_rate'] < 0.8:
            adjustments['coverage_quality'] = 0.02
            
        # 根据负载均衡情况调整负载均衡权重（新增）
        if self.radar_loads:
            # 计算负载标准差，衡量负载均衡程度
            loads = list(self.radar_loads.values())
            mean_load = sum(loads) / len(loads)
            load_std = (sum((l - mean_load) ** 2 for l in loads) / len(loads)) ** 0.5
            
            # 如果负载不均衡，增加负载均衡权重
            if load_std > 0.2:  # 负载标准差阈值
                adjustments['load_balance'] = 0.02
            elif load_std < 0.1:  # 负载已经比较均衡
                adjustments['load_balance'] = -0.01
            
        # 应用权重调整
        learning_rate = self.adaptive_params['learning_rate']
        for key in self.weights:
            if key in adjustments:
                self.weights[key] += adjustments[key] * learning_rate
                # 确保权重在有效范围内
                self.weights[key] = max(self.adaptive_params['min_weight'], 
                                      min(self.adaptive_params['max_weight'], 
                                          self.weights[key]))
            
        # 归一化权重
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for key in self.weights:
                self.weights[key] /= weight_sum

    def _update_performance_metrics(self, assignment, targets, observed_targets, t):
        """
        更新性能指标
        
        Args:
            assignment: 当前分配矩阵
            targets: 目标列表
            observed_targets: 观测目标列表
            t: 当前时间步
        """
        num_targets = len(targets)
        
        # 计算分配率
        assignment_rate = assignment.getnnz() / max(1, num_targets)
        
        # 计算跟踪率（连续跟踪的目标比例）
        tracking_rate = 0.0
        if len(self.history) > 1:
            prev_assignment = self.history[-1]
            tracked_targets = 0
            for i in range(min(num_targets, prev_assignment.shape[0])):
                if prev_assignment.getrow(i).nnz > 0 and assignment.getrow(i).nnz > 0:
                    tracked_targets += 1
            tracking_rate = tracked_targets / max(1, num_targets)
        
        # 计算切换率（改变雷达分配的目标比例）
        switching_rate = 0.0
        if len(self.history) > 1:
            prev_assignment = self.history[-1]
            switched_targets = 0
            for i in range(min(num_targets, prev_assignment.shape[0], assignment.shape[0])):
                prev_radar = prev_assignment.getrow(i).nonzero()[1]
                curr_radar = assignment.getrow(i).nonzero()[1]
                if prev_radar.size > 0 and curr_radar.size > 0 and prev_radar[0] != curr_radar[0]:
                    switched_targets += 1
            switching_rate = switched_targets / max(1, num_targets)
        
        # 计算约束违反率
        target_positions = [np.array(obs['position']) for obs in observed_targets]
        constraints_result = self.constraint_checker.verify_all_constraints(assignment, target_positions)
        constraint_violation_rate = 0.0 if constraints_result["all_satisfied"] else 1.0
        
        # 更新性能指标
        self.performance_metrics = {
            'tracking_rate': tracking_rate,
            'switching_rate': switching_rate,
            'constraint_violation_rate': constraint_violation_rate,
            'assignment_rate': assignment_rate,
        }
        
        # 记录历史性能
        self.adaptive_params['performance_history'].append(self.performance_metrics.copy())
        self.adaptive_params['weight_history'].append(self.weights.copy())
        self.adaptive_params['constraint_history'].append(constraint_violation_rate)
        self.adaptive_params['assignment_rate_history'].append(assignment_rate)
        
        # 保持历史记录在窗口大小内
        window_size = self.adaptive_params['performance_window']
        if len(self.adaptive_params['performance_history']) > window_size:
            self.adaptive_params['performance_history'] = self.adaptive_params['performance_history'][-window_size:]
        if len(self.adaptive_params['weight_history']) > window_size:
            self.adaptive_params['weight_history'] = self.adaptive_params['weight_history'][-window_size:]
        if len(self.adaptive_params['constraint_history']) > window_size:
            self.adaptive_params['constraint_history'] = self.adaptive_params['constraint_history'][-window_size:]
        if len(self.adaptive_params['assignment_rate_history']) > window_size:
            self.adaptive_params['assignment_rate_history'] = self.adaptive_params['assignment_rate_history'][-window_size:]

    def _get_target_priority(self, target_idx):
        """获取目标优先级，如果不存在则返回默认值"""
        if hasattr(self, '_target_priorities') and target_idx in self._target_priorities:
            return self._target_priorities[target_idx]
        return 1.0  # 默认优先级
