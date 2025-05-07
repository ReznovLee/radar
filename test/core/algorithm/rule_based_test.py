#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：radar
@File    ：rule_based_test.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/5/6 10:40
"""
import numpy as np
import os # 导入 os 用于路径操作
import json # 导入 json 用于保存结果
import matplotlib.pyplot as plt # 导入 matplotlib 用于绘图

# 示例 (在其他文件中)
from src.core.models.radar_model import RadarNetwork, Radar # 假设 Radar 也在这里
from src.core.algorithm.rule_based import RuleBasedScheduler
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO) # 或 logging.DEBUG 获取更详细信息

# 1. 创建 RadarNetwork 实例 (你需要根据你的配置创建)
radars_list = [
    Radar(radar_id=1, radar_position=np.array([0, 0, 0]), radar_radius=100000, num_channels=5),
    Radar(radar_id=2, radar_position=np.array([50000, 0, 0]), radar_radius=120000, num_channels=8)
    # ... 添加更多雷达
]
# 假设 RadarNetwork 接受雷达列表
radar_network = RadarNetwork(radars_list) # 你可能需要调整 RadarNetwork 的初始化方式

# 2. 指定数据目录
data_directory = r"c:\Users\Reznov Lee\PycharmProjects\radar\output\scenario-2025-04-28"
# data_filename = "your_data.csv" # 如果不是默认的 target_trajectory.csv
# --- 修改点 1：指定要读取的CSV文件名 ---
data_filename = "10-targets.csv" # 明确指定文件名，与 rule_based.py 中的默认值一致或根据需要修改

# 3. 创建调度器实例
# scheduler = RuleBasedScheduler(radar_network, data_directory, data_filename="your_data.csv", measurement_noise_std=10.0)
# --- 修改点 2：在创建调度器时传入文件名 ---
scheduler = RuleBasedScheduler(radar_network, data_directory, data_filename=data_filename, measurement_noise_std=10.0) # 使用指定的文件名和噪声

# 4. 运行模拟
assignment_history = scheduler.run_simulation()

# 5. 处理结果 (例如，传递给评估模块)
if assignment_history:
    print(f"模拟完成，生成了 {len(assignment_history)} 个时间步的分配记录。")
    # print(assignment_history[-1]) # 打印最后一个时间步的结果看看

    # --- 新增代码：保存分配结果到 JSON 文件 ---
    results_save_path = os.path.join(data_directory, "rule_based_assignment_history.json")
    try:
        with open(results_save_path, 'w') as f:
            # 将 numpy 数组转换为列表以便 JSON 序列化 (如果 history 中包含 numpy 对象)
            # 这里 assignment_history 是 list of dicts，可以直接 dump
            json.dump(assignment_history, f, indent=4)
        print(f"分配结果已保存到: {results_save_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")
    # --- 保存代码结束 ---

    # --- 新增代码：计算并绘制分配比例 ---
    timestamps = []
    assigned_ratios = []
    for step_data in assignment_history:
        timestamp = step_data['timestamp']
        assignments = step_data['assignments']
        total_targets = len(assignments)
        if total_targets > 0:
            assigned_count = sum(1 for radar_id in assignments.values() if radar_id is not None)
            ratio = assigned_count / total_targets
        else:
            ratio = 0 # 或者 np.nan，取决于你希望如何表示没有目标的时刻

        timestamps.append(timestamp)
        assigned_ratios.append(ratio)

    # 开始绘图
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, assigned_ratios, marker='.', linestyle='-')
    plt.title('Ratio of Assigned Targets Over Time (Rule-Based)')
    plt.xlabel('Timestamp')
    plt.ylabel('Assigned Ratio (Assigned / Total)')
    plt.ylim(0, 1.1) # 设置 Y 轴范围从 0 到 1.1
    plt.grid(True)
    plt.tight_layout() # 调整布局防止标签重叠

    # 保存图形
    plot_save_path = os.path.join(data_directory, "rule_based_assigned_ratio.png")
    try:
        plt.savefig(plot_save_path)
        print(f"分配比例图已保存到: {plot_save_path}")
    except Exception as e:
        print(f"保存图形时出错: {e}")

    plt.show() # 显示图形
    # --- 绘图代码结束 ---

else:
    print("模拟未产生任何分配记录。")
