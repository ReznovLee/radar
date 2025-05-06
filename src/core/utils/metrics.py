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
        - in_range_history: History of whether the target is in coverage
        - time_step: Time step in seconds
    """

    def __init__(self,
                 targets,
                 assignment_history: np.ndarray,
                 in_range_history=None,
                 time_step: float = 1.0):
        """
        Initialize the performance indicator calculator.

        Args:
            targets: A list of target information, each dictionary contains 'id' and 'priority'.
            assignment_history (np.ndarray): Tracking assignment history (target index x time step),
                                             value is tracking radar ID, 0 or negative value means not tracked.
            in_range_history: Whether the target is in the history of coverage (target index x timesteps), as a boolean. 
                                If provided, the tracking coverage can be calculated.
            time_step (float): The actual time (in seconds) that each time step represents.
        """
        if not targets:
            raise ValueError("The target list cannot be empty.")

        self.targets = targets
        self.target_map = {target['id']: i for i, target in enumerate(targets)} 
        self.num_targets = len(targets)
        self.assignment_history = assignment_history
        self.in_range_history = in_range_history
        self.time_step = time_step

        # Basic validation
        if self.assignment_history.shape[0] != self.num_targets:
            raise ValueError(f"Target quantity for allocation history ({self.assignment_history.shape[0]}) "
                             f"Does not match the target list number ({self.num_targets}).")
        if self.in_range_history is not None and self.in_range_history.shape != self.assignment_history.shape:
             raise ValueError("The shape of the scope history must match the shape of the allocation history.")

        self.num_time_steps = self.assignment_history.shape[1]
        self.priorities = np.array([t.get('priority', 1.0) for t in self.targets]) # Default priority 1.0 if missing

    def calculate_weighted_tracking_time(self):
        """ Calculate the weighted total tracking time.

        Time is weighted according to goal priority.

        Returns:
            float: Weighted total tracking time.
        """
        total_weighted_time = 0.0
        # Check if assignment_history is empty or invalid
        if self.assignment_history is None or self.assignment_history.size == 0:
            return 0.0

        is_tracked_matrix = self.assignment_history > 0 # Boolean matrix: True if tracked

        for i in range(self.num_targets):
            tracked_steps = np.sum(is_tracked_matrix[i, :])
            total_weighted_time += tracked_steps * self.priorities[i]

        return total_weighted_time * self.time_step

    def calculate_tracking_coverage_ratio(self): 
        """ Calculate the overall trace coverage.
        
        Defined as: (total number of time steps all targets were tracked) / 
                    (total number of time steps all targets were within the coverage range).
        Can only be computed if `in_range_history` is provided.

        Returns:
            float or None: The overall trace coverage, or None if it cannot be computed.
        """
        if self.in_range_history is None:
            print("Warning: 'in_range_history' not provided, unable to compute trace coverage.")
            return None
        # Check if in_range_history is empty or invalid
        if self.in_range_history.size == 0:
             return 0.0

        total_tracked_steps = np.sum(self.assignment_history > 0)
        total_in_range_steps = np.sum(self.in_range_history)

        if total_in_range_steps == 0:
            # Handle case where targets are never in range
            return 0.0 if total_tracked_steps == 0 else None

        return total_tracked_steps / total_in_range_steps

    def calculate_tracking_handoffs(self):
        """ Count the number of radar switches for each target.
        
        When the radar IDs tracking the same target change in adjacent time steps, 
        it is counted as a switch.   

        Returns:
            dict:  A dictionary whose keys are target IDs and values are the number 
                    of times the target has been switched.
        """
        handoffs = {target['id']: 0 for target in self.targets}
        # Check if assignment_history is empty or invalid
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

    def generate_report(self):
        """ Debug function.
        
        Generates a comprehensive report containing all calculated metrics for debugging purposes.
        

        Returns:
            dict: A dictionary containing various performance indicators.
        """
        report = {
            "weighted_tracking_time": self.calculate_weighted_tracking_time(),
            "tracking_coverage_ratio": self.calculate_tracking_coverage_ratio(),
            "tracking_handoffs": self.calculate_tracking_handoffs(),
        }
        return report
