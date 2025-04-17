#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : constraints.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:25
"""
import numpy as np
from typing import Dict, List, Union, Tuple
from scipy.sparse import csr_matrix
from src.core.models.radar_model import RadarNetwork


class ConstraintChecker:
    """
    Constraint checker class for validating radar network scheduling constraints:
    1. Each target can only be tracked by one radar at a time (C13).
    2. The number of targets tracked by a radar must not exceed its channel count (C14).
    3. Targets must be within the radar's coverage range (C15).
    4. Assignment matrix entries must be binary (C16).
    """
    def __init__(self, radar_network: RadarNetwork):
        self.radar_network = radar_network
        self.radar_ids = list(radar_network.radars.keys())

    def check_single_assignment(self, assignment: csr_matrix) -> bool:
        """
        C13 Constraint: Ensure each target is tracked by at most one radar at any given time.
        """
        return np.all(assignment.sum(axis=1).A1 <= 1)

    def check_radar_channels(self, assignment: csr_matrix) -> Tuple[bool, List[int]]:
        """
        C14 Constraint: Ensure the number of targets tracked by each radar does not exceed its available channels.
        
        Returns:
            Tuple[bool, List[int]]: (是否满足约束, 超载雷达ID列表)
        """
        allocated_targets = assignment.sum(axis=0).A1
        overloaded_radars = []
        
        for i, radar_id in enumerate(self.radar_ids):
            if i < len(allocated_targets):
                radar = self.radar_network.radars[radar_id]
                if allocated_targets[i] > radar.num_channels:
                    overloaded_radars.append(radar_id)
        
        return len(overloaded_radars) == 0, overloaded_radars

    def check_radar_coverage(self, assignment: csr_matrix, target_positions: List[np.ndarray]) -> Tuple[bool, List[int]]:
        """
        C15 Constraint: Ensure each assigned target is within the radar's coverage range.
        
        Returns:
            Tuple[bool, List[int]]: (是否满足约束, 超出覆盖范围的目标ID列表)
        """
        rows, cols = assignment.nonzero()
        out_of_range_targets = []
        
        for target_idx, radar_idx in zip(rows, cols):
            if target_idx >= len(target_positions):
                out_of_range_targets.append(target_idx)
                continue
                
            radar_id = self.radar_ids[radar_idx] if radar_idx < len(self.radar_ids) else None
            radar = self.radar_network.radars.get(radar_id)
            
            if radar is None or not radar.is_target_in_range(target_positions[target_idx]):
                out_of_range_targets.append(target_idx)
        
        return len(out_of_range_targets) == 0, out_of_range_targets

    def check_binary_variables(self, assignment: csr_matrix) -> bool:
        """
        C16 Constraint: Ensure that assignment matrix entries are binary (0 or 1).
        """
        return np.all(np.isin(assignment.data, [0, 1])) if assignment.count_nonzero() > 0 else True

    def verify_all_constraints(self, assignment: csr_matrix, target_positions: List[np.ndarray]) -> Dict[str, Union[bool, List[int]]]:
        """
        Run all constraint checks and return the results.
        
        Returns:
            Dict with:
            - "C13": True/False - 每个目标最多被一个雷达跟踪
            - "C14": List of overloaded radar IDs - 超出通道数的雷达ID列表
            - "C15": List of target IDs outside radar coverage - 超出覆盖范围的目标ID列表
            - "C16": True/False - 分配矩阵是否为二进制
            - "all_satisfied": True/False - 是否满足所有约束
        """
        c13_satisfied = self.check_single_assignment(assignment)
        c14_satisfied, overloaded_radars = self.check_radar_channels(assignment)
        c15_satisfied, out_of_range_targets = self.check_radar_coverage(assignment, target_positions)
        c16_satisfied = self.check_binary_variables(assignment)
        
        results = {
            "C13": c13_satisfied,
            "C14": overloaded_radars,
            "C15": out_of_range_targets,
            "C16": c16_satisfied,
            "all_satisfied": c13_satisfied and c14_satisfied and c15_satisfied and c16_satisfied
        }
        
        return results
    
    def get_constraint_violations(self, assignment: csr_matrix, target_positions: List[np.ndarray]) -> Dict[str, str]:
        """
        获取约束违反的详细信息
        
        Returns:
            Dict[str, str]: 约束违反的详细描述
        """
        results = self.verify_all_constraints(assignment, target_positions)
        violations = {}
        
        if not results["C13"]:
            violations["C13"] = "有目标被多个雷达同时跟踪"
            
        if results["C14"]:
            violations["C14"] = f"雷达 {results['C14']} 超出了可用通道数"
            
        if results["C15"]:
            violations["C15"] = f"目标 {results['C15']} 超出了雷达覆盖范围"
            
        if not results["C16"]:
            violations["C16"] = "分配矩阵包含非二进制值"
            
        return violations