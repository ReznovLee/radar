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
from scipy import sparse
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
        """ Initialization method.

        Initialize the ConstraintChecker class.

        :param radar_network:
        """
        self.radar_network = radar_network
        self.radar_ids = list(radar_network.radars.keys())

    @staticmethod
    def validate_input(assignment):
        """ Validate the input params

        Used to verify that the parameters involved in this class meet the sparse matrix
        format requirements and provide support for sparse matrix calculations in other codes.

        :param assignment: Radar-target assignment matrix.
        :return: True if the assignment result is valid, False otherwise.
        """
        if not sparse.issparse(assignment):
            raise ValueError("Assignment matrix must be a sparse matrix")
        return True

    def _check_single_assignment(self, assignment):
        """ C13 constraint checker.

        C13 Constraint: Ensure each target is tracked by at most one radar at any given time.

        :param assignment: Radar-target assignment matrix.
        :return: True if the assignment result is valid, False otherwise.
        """
        if self.validate_input(assignment) is False:
            raise ValueError("Assignment matrix must be a sparse matrix, please check the assignment params.")
        return np.all(assignment.sum(axis=1).A1 <= 1)

    def check_radar_channels(self, assignment):
        """ C14 constraint checker.

        C14 Constraint: Ensure the number of targets tracked by each radar does not exceed its available channels.

        :param assignment: Radar-target assignment matrix.
        :return: True if the assignment result is valid, False otherwise.
        """
        if self.validate_input(assignment) is False:
            raise ValueError("Assignment matrix must be a sparse matrix, please check the assignment params.")

        allocated_targets = assignment.sum(axis=0).A1
        overloaded_radars = []

        for i, radar_id in enumerate(self.radar_ids):
            if i < len(allocated_targets):
                radar = self.radar_network.radars[radar_id]
                if allocated_targets[i] > radar.num_channels:
                    overloaded_radars.append(radar_id)

        return len(overloaded_radars) == 0, overloaded_radars

    def check_radar_coverage(self, assignment, target_positions):
        """ C15 constraint checker.

        C15 Constraint: Ensure each assigned target is within the radar's coverage range.

        :param assignment: Radar-target assignment matrix.
        :param target_positions: Radar-target positions.
        :return: True if the assignment result is valid, False otherwise.
        """
        if self.validate_input(assignment) is False:
            raise ValueError("Assignment matrix must be a sparse matrix, please check the assignment params.")

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

    @staticmethod
    def check_binary_variables(assignment):
        """ C16 constraint checker.

        C16 Constraint: Ensure that assignment matrix entries are binary (0 or 1).

        :param assignment: Radar-target assignment matrix.
        :return: True if the assignment result is valid, False otherwise.
        """
        return np.all(np.isin(assignment.data, [0, 1])) if assignment.count_nonzero() > 0 else True

    def verify_all_constraints(self, assignment, target_positions):
        """ All constraint checker.

        Run all constraint checks and return the results.

        Attribute:
            assignment: Radar-target assignment matrix.
            target_positions: Radar-target positions.

        Returns:
            Dict with:
            - "C13": True/False - each target is assigned to at most one radar
            - "C14": List of overloaded radar IDs - overloaded radar IDs
            - "C15": List of target IDs outside radar coverage - overloaded target IDs
            - "C16": True/False - assignment matrix entries are binary
            - "all_satisfied": True/False - all constraints are satisfied
        """
        c13_satisfied = self._check_single_assignment(assignment)
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

    def get_constraint_violations(self, assignment, target_positions):
        """
        Get the detailed description of constraint violations.
        
        Returns:
            Description of constraint violations.
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
