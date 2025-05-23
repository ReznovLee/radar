# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : scenario_visualization.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:27
"""
import csv
import matplotlib.pyplot as plt
import numpy as np


def load_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


def process_radar_data(radar_data):
    radars = []
    for row in radar_data:
        radars.append({
            "id": int(row["id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "radius": float(row["radius"]),
            "channels": int(row["number_channel"]),
        })
    return radars


def process_target_data(target_data):
    targets = {}
    for row in target_data:
        target_id = int(row["id"])
        if target_id not in targets:
            targets[target_id] = []
        targets[target_id].append({
            "timestep": int(float(row["timestep"])),
            "position_x": float(row["position_x"]),
            "position_y": float(row["position_y"]),
            "position_z": float(row["position_z"]),
            "velocity_x": float(row["velocity_x"]),
            "velocity_y": float(row["velocity_y"]),
            "velocity_z": float(row["velocity_z"]),
            "target_type": row["target_type"],
            "priority": int(row["priority"]),
        })
    return targets


def plot_scenario(radar_file, target_file, save_path):
    radar_data = process_radar_data(load_csv(radar_file))
    target_data = process_target_data(load_csv(target_file))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for radar in radar_data:
        ax.scatter(radar["x"], radar["y"], radar["z"], c='r', marker='^', label=f'Radar {radar["id"]}')
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = radar["radius"] * np.cos(u) * np.sin(v) + radar["x"]
        y = radar["radius"] * np.sin(u) * np.sin(v) + radar["y"]
        z = radar["radius"] * np.cos(v) + radar["z"]
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    colors = plt.cm.jet(np.linspace(0, 1, len(target_data)))  # 给每个目标分配不同颜色
    for (target_id, trajectory), color in zip(target_data.items(), colors):
        x = [p["position_x"] for p in trajectory]
        y = [p["position_y"] for p in trajectory]
        z = [p["position_z"] for p in trajectory]
        ax.plot(x, y, z, label=f'Target {target_id}', color=color)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Scenario Visualization: Target Trajectories & Radar Coverage")
    ax.legend()

    # 保存或显示图像
    if save_path:
        plt.show()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # 示例文件路径（需替换为实际路径）
    radar_csv = "output/scenario-2025-05-18/5-radar.csv"
    target_csv = "output/scenario-2025-05-18/50-targets.csv"
    save_image = "output/scenario-2025-05-18/scenario_visualization.png"

    plot_scenario(radar_csv, target_csv, save_image)
