#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：radar
@File    ：plot.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/2/5 15:30
"""

import csv
import sys

import numpy as np
import matplotlib
if sys.platform == "win32":
    matplotlib.use("TkAgg")
elif sys.platform == "darwin":
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict


# 读取 CSV 文件
def load_csv(file_path: str) -> List[Dict[str, str]]:
    """ 读取 CSV 文件并转换为字典列表 """
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


# 解析雷达数据
def process_radar_data(radar_data: List[Dict[str, str]]) -> List[Dict]:
    """ 解析雷达数据并返回雷达字典列表 """
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


# 解析目标数据
def process_target_data(target_data: List[Dict[str, str]]) -> Dict[int, List[Dict]]:
    """ 解析目标数据，并将相同 ID 的目标轨迹合并 """
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


# 可视化目标轨迹和雷达探测范围
def plot_scenario(radar_file: str, target_file: str, save_path: str = None):
    """ 绘制目标轨迹和雷达探测范围 """
    radar_data = process_radar_data(load_csv(radar_file))
    target_data = process_target_data(load_csv(target_file))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制雷达探测范围
    for radar in radar_data:
        ax.scatter(radar["x"], radar["y"], radar["z"], c='r', marker='^', label=f'Radar {radar["id"]}')
        # 以 (radar["x"], radar["y"], radar["z"]) 为球心的半球
        u, v = np.mgrid[0:2 * np.pi:360j, 0:0.5 * np.pi:60j]
        x = radar["radius"] * np.cos(u) * np.sin(v) + radar["x"]
        y = radar["radius"] * np.sin(u) * np.sin(v) + radar["y"]
        z = radar["radius"] * np.cos(v) + radar["z"]
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)

    # 绘制目标轨迹
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

    # 设置三维坐标轴等比例
    def set_axes_equal(ax):
        '''使3D坐标轴等比例'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    set_axes_equal(ax)

    # 保存或显示图像
    if save_path:
        plt.show()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    radar_csv = "output/scenario-2025-05-12/5-radar.csv"
    target_csv = "output/scenario-2025-05-12/10-targets.csv"
    save_image = "output/scenario-2025-05-12/scenario_visualization.png"

    plot_scenario(radar_csv, target_csv, save_image)
