#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：radar 
@File    ：LNS.py
@IDE     ：PyCharm 
@Author  ：ReznovLee
@Date    ：2025/5/16 11:40 
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple, Set
import random
import logging
from core.models.radar_model import RadarNetwork
from core.utils.filter import BallisticMissileEKF, CruiseMissileEKF, AircraftIMMEKF
from core.utils.metrics import RadarPerformanceMetrics
from core.utils.constraints import ConstraintChecker


class LNS:
    """
    Large Neighborhood Search based radar target assignment algorithm
    
    Finds better radar-target assignments by iteratively destroying and repairing the current solution.
    """

    def __init__(self, radar_network: RadarNetwork, max_iterations: int = 3, 
                 destroy_percentage: float = 0.3, tabu_size: int = 5):
        """
        Initialize LNS algorithm
        
        Args:
            radar_network: Radar network object
            max_iterations: Maximum iterations per time step (reduced to improve performance)
            destroy_percentage: Percentage of assignments to destroy in each iteration
            tabu_size: Size of tabu list to avoid repeating recent assignments
        """
        self.radar_network = radar_network
        self.max_iterations = max_iterations  # Reduced from 10 to 3
        self.destroy_percentage = destroy_percentage
        self.radar_ids = list(radar_network.radars.keys())
        self.history = []  # Assignment history
        self.constraint_checker = ConstraintChecker(radar_network)
        self.tabu_list = []  # Tabu list to store recent assignments
        self.tabu_size = tabu_size
        self.best_score_history = []  # Record best score for each time step

    def solve(self, targets: List[Dict], observed_targets: List[Dict], t: int) -> csr_matrix:
        """
        Execute LNS algorithm to solve radar-target assignment problem
        
        Args:
            targets: Real target information (with id, priority, type, etc.)
            observed_targets: Noisy observations (with id, position, velocity, etc.)
            t: Current time step
            
        Returns:
            assignment: Sparse assignment matrix (rows: targets, columns: radars)
        """
        num_targets = len(targets)
        num_radars = len(self.radar_ids)
        
        # 1. Initialize solution using greedy algorithm
        current_solution = self._initialize_solution(targets, observed_targets, num_targets, num_radars)
        
        # Evaluate initial solution
        best_solution = current_solution.copy()
        best_score = self._evaluate_solution(best_solution, targets, observed_targets)
        
        # 2. Iterative optimization (limited iterations)
        for iteration in range(self.max_iterations):
            # Destroy current solution
            destroyed_solution, destroyed_indices = self._destroy(current_solution, num_targets)
            
            # Repair solution
            repaired_solution = self._repair(destroyed_solution, destroyed_indices, targets, observed_targets)
            
            # Evaluate new solution
            new_score = self._evaluate_solution(repaired_solution, targets, observed_targets)
            
            # Acceptance criteria: accept if better or with probability based on score difference
            if new_score > best_score or random.random() < np.exp((new_score - best_score) / max(1.0, iteration + 1)):
                current_solution = repaired_solution.copy()
                
                # Update best solution
                if new_score > best_score:
                    best_solution = repaired_solution.copy()
                    best_score = new_score
                    
                # Check if in tabu list
                solution_hash = self._hash_solution(repaired_solution)
                if solution_hash not in self.tabu_list:
                    # Add to tabu list
                    self.tabu_list.append(solution_hash)
                    if len(self.tabu_list) > self.tabu_size:
                        self.tabu_list.pop(0)
        
        # Record best score
        self.best_score_history.append(best_score)
        
        # Record history
        if len(self.history) > 10:  # Keep only recent 10 time steps
            self.history.pop(0)
        self.history.append(best_solution.copy())
        
        return best_solution

    def _initialize_solution(self, targets: List[Dict], observed_targets: List[Dict], 
                            num_targets: int, num_radars: int) -> csr_matrix:
        """
        Initialize solution: prioritize historical assignments, otherwise use greedy algorithm
        """
        # If there's history, prioritize it
        if self.history:
            # Check if history assignment shape matches
            last_assignment = self.history[-1]
            if last_assignment.shape == (num_targets, num_radars):
                # Verify if history assignment still satisfies constraints
                if self.constraint_checker.verify_all_constraints(
                    last_assignment, [obs['position'] for obs in observed_targets]
                )["all_satisfied"]:
                    return last_assignment.copy()
        
        # Otherwise use greedy algorithm
        assignment = csr_matrix((num_targets, num_radars), dtype=np.int8)
        
        # Sort targets by priority
        sorted_targets = sorted(enumerate(zip(targets, observed_targets)), 
                              key=lambda x: x[1][0].get('priority', 1), reverse=True)
        
        for idx, (target, obs) in sorted_targets:
            pos = np.array(obs['position'])
            
            # Find all available radars covering the target
            candidate_radars = []
            for j, radar_id in enumerate(self.radar_ids):
                radar = self.radar_network.radars[radar_id]
                if (radar.is_target_in_range(pos) and 
                    self.radar_network.is_radar_available(radar_id)):
                    candidate_radars.append(j)
            
            if not candidate_radars:
                continue
                
            # Choose closest radar
            best_j = None
            min_dist = float('inf')
            for j in candidate_radars:
                radar = self.radar_network.radars[self.radar_ids[j]]
                dist = np.linalg.norm(pos - radar.radar_position)
                if dist < min_dist:
                    min_dist = dist
                    best_j = j
            
            if best_j is not None:
                assignment[idx, best_j] = 1
                
                # Check constraints
                if not self.constraint_checker.verify_all_constraints(
                    assignment, [obs['position'] for obs in observed_targets]
                )["all_satisfied"]:
                    assignment[idx, best_j] = 0
        
        return assignment

    def _destroy(self, solution: csr_matrix, num_targets: int) -> Tuple[csr_matrix, List[int]]:
        """
        Destroy current solution: randomly select some targets and remove their assignments
        
        Returns:
            destroyed_solution: Solution after destruction
            destroyed_indices: List of target indices that were destroyed
        """
        destroyed_solution = solution.copy()
        
        # Determine number of targets to destroy
        num_to_destroy = max(1, int(num_targets * self.destroy_percentage))
        
        # Find assigned targets
        assigned_targets = []
        for i in range(num_targets):
            if destroyed_solution.getrow(i).nnz > 0:
                assigned_targets.append(i)
        
        # If no assigned targets, return original solution
        if not assigned_targets:
            return destroyed_solution, []
        
        # Randomly select targets to destroy
        num_to_destroy = min(num_to_destroy, len(assigned_targets))
        destroyed_indices = random.sample(assigned_targets, num_to_destroy)
        
        # Remove assignments for selected targets
        for idx in destroyed_indices:
            destroyed_solution[idx, :] = 0
        
        return destroyed_solution, destroyed_indices

    def _repair(self, solution: csr_matrix, destroyed_indices: List[int], 
               targets: List[Dict], observed_targets: List[Dict]) -> csr_matrix:
        """
        Repair solution: reassign radars to destroyed targets
        """
        repaired_solution = solution.copy()
        
        # Sort destroyed targets by priority
        sorted_indices = sorted(destroyed_indices, 
                              key=lambda idx: targets[idx].get('priority', 1), 
                              reverse=True)
        
        for idx in sorted_indices:
            obs = observed_targets[idx]
            pos = np.array(obs['position'])
            
            # Find all available radars covering the target
            candidate_radars = []
            for j, radar_id in enumerate(self.radar_ids):
                radar = self.radar_network.radars[radar_id]
                # Check if radar covers target and is available
                if (radar.is_target_in_range(pos) and 
                    self.radar_network.is_radar_available(radar_id)):
                    # Temporarily assign to check constraints
                    temp_solution = repaired_solution.copy()
                    temp_solution[idx, j] = 1
                    if self.constraint_checker.verify_all_constraints(
                        temp_solution, [obs['position'] for obs in observed_targets]
                    )["all_satisfied"]:
                        candidate_radars.append(j)
            
            if not candidate_radars:
                continue
            
            # Use heuristic to select best radar
            best_j = self._select_best_radar(idx, candidate_radars, obs, targets[idx])
            
            if best_j is not None:
                repaired_solution[idx, best_j] = 1
        
        return repaired_solution

    def _select_best_radar(self, target_idx: int, candidate_radars: List[int], 
                          obs: Dict, target: Dict) -> int:
        """
        Use heuristics to select best radar
        
        Factors considered:
        1. Radar-target distance
        2. Radar load balancing
        3. Target movement direction vs radar direction
        4. Historical assignment stability
        """
        if not candidate_radars:
            return None
        
        pos = np.array(obs['position'])
        scores = []
        
        for j in candidate_radars:
            radar = self.radar_network.radars[self.radar_ids[j]]
            
            # 1. Distance factor: closer is better
            dist = np.linalg.norm(pos - radar.radar_position)
            dist_score = 1.0 / (1.0 + dist / radar.radar_radius)
            
            # 2. Load balancing: fewer assigned channels is better
            load = sum(1 for ch in radar.radar_channels.values() if ch is not None)
            load_score = 1.0 - load / max(1, radar.num_channels)
            
            # 3. Movement direction vs radar direction: smaller angle is better
            direction_score = 0.0
            if 'velocity' in obs and np.linalg.norm(obs['velocity']) > 1e-3:
                v = np.array(obs['velocity'])
                to_radar = radar.radar_position - pos
                cos_theta = np.dot(v, to_radar) / (np.linalg.norm(v) * np.linalg.norm(to_radar) + 1e-6)
                direction_score = (cos_theta + 1) / 2  # Normalize to [0,1]
            
            # 4. Historical stability: previously assigned radar is preferred
            stability_score = 0.0
            if self.history:
                last_assignment = self.history[-1]
                if target_idx < last_assignment.shape[0]:
                    row = last_assignment.getrow(target_idx).toarray().ravel()
                    if j < len(row) and row[j] > 0:
                        stability_score = 1.0
            
            # Combined score with weights
            priority_weight = target.get('priority', 1) / 5.0  # Normalize priority
            final_score = (
                0.3 * dist_score + 
                0.2 * load_score + 
                0.2 * direction_score + 
                0.3 * stability_score
            ) * (1 + priority_weight)  # Priority as multiplier
            
            scores.append((j, final_score))
        
        # Choose radar with highest score
        if scores:
            return max(scores, key=lambda x: x[1])[0]
        return None

    def _evaluate_solution(self, solution: csr_matrix, targets: List[Dict], 
                          observed_targets: List[Dict]) -> float:
        """
        Evaluate solution quality
        
        Factors considered:
        1. Assignment rate: assigned targets / total targets
        2. Priority weighting: higher priority targets are more important
        3. Distance efficiency: closer radars are better
        4. Load balancing: even distribution across radars
        """
        if solution is None or solution.nnz == 0:
            return 0.0
        
        # 1. Assignment rate with priority weighting
        assignment_score = 0.0
        total_priority = sum(target.get('priority', 1) for target in targets)
        
        for i, target in enumerate(targets):
            if solution.getrow(i).nnz > 0:
                priority = target.get('priority', 1)
                assignment_score += priority
        
        if total_priority > 0:
            assignment_score /= total_priority
        
        # 2. Distance efficiency
        distance_score = 0.0
        assigned_count = 0
        
        for i, obs in enumerate(observed_targets):
            radar_idx = solution.getrow(i).nonzero()[1]
            if len(radar_idx) > 0:
                j = radar_idx[0]
                radar = self.radar_network.radars[self.radar_ids[j]]
                pos = np.array(obs['position'])
                
                # Normalize distance to [0,1] where 1 is best (closest)
                dist = np.linalg.norm(pos - radar.radar_position)
                normalized_dist = 1.0 - min(1.0, dist / radar.radar_radius)
                distance_score += normalized_dist
                assigned_count += 1
        
        if assigned_count > 0:
            distance_score /= assigned_count
        
        # 3. Load balancing
        load_score = 0.0
        radar_loads = np.zeros(len(self.radar_ids))
        
        for j in range(len(self.radar_ids)):
            radar_loads[j] = solution[:, j].sum()
        
        if solution.nnz > 0 and len(self.radar_ids) > 1:
            # Calculate coefficient of variation (lower is better)
            cv = np.std(radar_loads) / (np.mean(radar_loads) + 1e-6)
            load_score = 1.0 / (1.0 + cv)  # Transform to [0,1] where 1 is best
        
        # Combined score with weights
        final_score = 0.5 * assignment_score + 0.3 * distance_score + 0.2 * load_score
        
        return final_score

    def _hash_solution(self, solution: csr_matrix) -> str:
        """
        Create a hash of the solution for tabu list
        """
        # Simple hash: concatenate row indices and column indices
        rows, cols = solution.nonzero()
        return ','.join(f"{r}:{c}" for r, c in zip(rows, cols))


