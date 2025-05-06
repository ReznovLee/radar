# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : target_test.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/05/06 16:04
"""
import numpy as np
from src.core.models.target_model import (
    BallisticMissileTargetModel,
    CruiseMissileTargetModel,
    AircraftTargetModel
)


def print_target_state(target, timestamp):
    """Helper function to print target state."""
    state = target.get_state(timestamp)
    print(f"Timestamp: {state[1]:.2f}s")
    print(f"  ID: {state[0]}, Type: {state[4]}, Priority: {state[5]}")
    print(f"  Position: {state[2].round(2)}")
    print(f"  Velocity: {state[3].round(2)}")
    if hasattr(target, 'acceleration'):
        print(f"  Acceleration: {target.acceleration.round(2)}")
    if hasattr(target, 'current_phase'):  # For CruiseMissile
        print(f"  Phase: {target.current_phase}")
    print("-" * 30)


def test_target_simulation(target, num_steps=10, delta_time=1.0):
    """Simulates a target for a number of steps and prints its state."""
    print(f"\n--- Simulating {target.target_type} (ID: {target.target_id}) ---")
    print_target_state(target, 0.0)  # Initial state
    for i in range(num_steps):
        timestamp = (i + 1) * delta_time
        target.update_state(delta_time)
        print_target_state(target, timestamp)


if __name__ == "__main__":
    # --- Test BallisticMissileTargetModel ---
    ballistic_missile = BallisticMissileTargetModel(
        target_id=1,
        target_position=np.array([0, 0, 20000], dtype=float),  # Initial position (x, y, z) in meters
        velocity_ms=np.array([1000, 500, -100], dtype=float)  # Initial velocity (vx, vy, vz) in m/s
    )
    test_target_simulation(ballistic_missile, num_steps=5, delta_time=1.0)

    # --- Test CruiseMissileTargetModel ---
    cruise_missile = CruiseMissileTargetModel(
        target_id=2,
        target_position=np.array([0, 0, 8000], dtype=float),
        velocity=np.array([250, 0, 0], dtype=float),  # Mach 0.7-0.8 approx 240-270 m/s
        cruise_end_point=np.array([10000, 0, 1000], dtype=float),  # Target point for dive phase
        dive_time=60.0,  # Estimated dive time
        cruise_time=120.0,  # Total cruise time
        rocket_acceleration=np.array([0, 0, -5], dtype=float)  # Acceleration during dive
    )
    test_target_simulation(cruise_missile, num_steps=15, delta_time=2.0)  # Longer simulation to see phase change

    # --- Test AircraftTargetModel ---
    aircraft = AircraftTargetModel(
        target_id=3,
        target_position=np.array([0, 0, 7000], dtype=float),
        velocity_ms=np.array([200, 50, 0], dtype=float)  # Initial velocity
    )
    # Aircraft model has more complex maneuvering, simulate for more steps
    test_target_simulation(aircraft, num_steps=20, delta_time=1.0)

    # --- Test Basic TargetModel (if needed, though usually you'd test subclasses) ---
    # basic_target = TargetModel(
    #     target_id=0,
    #     target_position=np.array([0, 0, 100], dtype=float),
    #     velocity=np.array([10, 5, 1], dtype=float),
    #     target_type="Generic",
    #     priority=4
    # )
    # # For basic TargetModel, you might need to set acceleration manually if not done in its update_state
    # # basic_target.acceleration = np.array([0, 0, -1.0]) # Example constant acceleration
    # # test_target_simulation(basic_target, num_steps=5, delta_time=0.5)

    print("\n--- All Target Model Tests Completed ---")
