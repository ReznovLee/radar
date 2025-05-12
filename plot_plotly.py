#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：radar
@File    ：plot_plotly.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/5/12 11:30
"""

import csv
import numpy as np
import plotly.graph_objs as go
# from typing import List, Dict   # 删除此行


# 读取 CSV 文件
def load_csv(file_path):   # 删除类型声明
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


# 解析雷达数据
def process_radar_data(radar_data):   # 删除类型声明
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
def process_target_data(target_data):   # 删除类型声明
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


# 绘制三维场景
def plot_scenario_plotly(radar_file: str, target_file: str, save_path: str = None):
    radar_data = process_radar_data(load_csv(radar_file))
    target_data = process_target_data(load_csv(target_file))

    fig = go.Figure()

    # 绘制雷达探测半球
    for radar in radar_data:
        # 雷达中心
        fig.add_trace(go.Scatter3d(
            x=[radar["x"]], y=[radar["y"]], z=[radar["z"]],
            mode='markers',
            marker=dict(size=1, color='red', symbol='diamond'),
            name=f'Radar {radar["id"]}'
        ))
        # 半球面
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, 0.5 * np.pi, 30)
        u, v = np.meshgrid(u, v)
        x = radar["radius"] * np.cos(u) * np.sin(v) + radar["x"]
        y = radar["radius"] * np.sin(u) * np.sin(v) + radar["y"]
        z = radar["radius"] * np.cos(v) + radar["z"]
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            showscale=False,
            opacity=0.15,
            surfacecolor=np.ones_like(z) * radar["id"],
            colorscale='ice', # Greys
            name=f'Radar {radar["id"]} Coverage',
            hoverinfo='skip'
        ))

    # 绘制目标轨迹
    color_list = px.colors.qualitative.Dark24 if hasattr(go, 'px') else [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    for idx, (target_id, trajectory) in enumerate(target_data.items()):
        x = [p["position_x"] for p in trajectory]
        y = [p["position_y"] for p in trajectory]
        z = [p["position_z"] for p in trajectory]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(color=color_list[idx % len(color_list)], width=1),
            marker=dict(size=0.5), # Points size
            name=f'Target {target_id}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data',
            xaxis=dict(
                tickformat='.0f',
                exponentformat='none'
            ),
            yaxis=dict(
                tickformat='.0f',
                exponentformat='none'
            ),
            zaxis=dict(
                tickformat='.0f',
                exponentformat='none'
            ),
        ),
        title="Scenario Visualization: Target Trajectories & Radar Coverage",
        legend=dict(itemsizing='constant')
    )

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


if __name__ == "__main__":
    radar_csv = "output/scenario-2025-05-12/5-radar.csv"
    target_csv = "output/scenario-2025-05-12/10-targets.csv"
    save_html = "output/scenario-2025-05-12/scenario_visualization_plotly.html"

    plot_scenario_plotly(radar_csv, target_csv, save_html)
