#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：radar
@File    ：radar_test.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/4/14 10:45
"""
import numpy as np # 导入 numpy
import logging # 导入 logging

from src.core.models.radar_model import Radar, RadarNetwork
from src.core.models.target_model import TargetModel

# 配置日志记录，以便能看到 radar_model.py 中的日志输出
logging.basicConfig(level=logging.INFO)


radar1 = Radar(radar_id=0, radar_position=np.array([0,0,0]), radar_radius=1000, num_channels=1) # 使用 np.array
radar2 = Radar(radar_id=1, radar_position=np.array([10,10,0]), radar_radius=1500, num_channels=1) # 使用 np.array

# --- 修改点 1: RadarNetwork 初始化使用字典 ---
# radar_network_1 = RadarNetwork({radar1, radar2}) # 原代码
radar_network_1 = RadarNetwork({radar1.radar_id: radar1, radar2.radar_id: radar2}) # 修改后：使用字典
# --- 修改结束 ---

target1 = TargetModel(target_id=0, target_position=np.array([100,100,100]), velocity=np.array([10,0,0]), target_type="aircraft", priority=3) # 使用 np.array

print("--- Testing Radar Class ---")
available_sign = radar1.is_radar_available()
print("Is radar1 available:", available_sign)

# 测试分配通道
channel_test_radar2 = radar2.assign_channel(target1.target_id)
if channel_test_radar2 is not None:
    print(f"Radar2 assigned channel {channel_test_radar2} to target {target1.target_id}")
else:
    print(f"Radar2 failed to assign channel to target {target1.target_id}")

# 测试释放通道
if channel_test_radar2 is not None:
    release_success = radar2.release_channel(channel_test_radar2)
    print(f"Radar2 release channel {channel_test_radar2} status: {release_success}")
else:
    print("Skipping release_channel test for radar2 as no channel was assigned.")


# 测试目标是否在范围内
range_test = radar1.is_target_in_range(target1.target_position)
print("Is target1 in range of radar1:", range_test)

# 测试一个不在范围内的目标
target_far = TargetModel(target_id=1, target_position=np.array([2000,0,0]), velocity=np.array([0,0,0]), target_type="missile", priority=1)
range_test_far = radar1.is_target_in_range(target_far.target_position)
print(f"Is target_far in range of radar1: {range_test_far}")


print("\n" + "-"*50 + "\n")
print("--- Testing RadarNetwork Class ---")

# 测试查找覆盖雷达
# 注意：find_covering_radars 在 radar_model.py 中期望 self.radars 是可迭代的（列表或字典值）
# 如果 RadarNetwork 初始化时 self.radars 是字典，find_covering_radars 内部需要正确迭代
# 假设 radar_model.py 中 find_covering_radars 已能正确处理字典形式的 self.radars (例如迭代 self.radars.values())
covering_radars_list = radar_network_1.find_covering_radars(target1.target_position)
print(f"Radars in network covering target1: {[r.radar_id for r in covering_radars_list] if covering_radars_list else 'None'}")

# 测试雷达网络分配雷达
# 假设 radar_model.py 中 RadarNetwork.assign_radar 已将 radar.allocate_channel 改为 radar.assign_channel
rn_test_assign = radar_network_1.assign_radar(target1.target_id, target1.target_position)
if rn_test_assign and rn_test_assign[0] is not None and rn_test_assign[1] is not None:
    assigned_radar_id, assigned_channel_id = rn_test_assign
    print(f"Radar network assign result: radar {assigned_radar_id} assigned channel {assigned_channel_id} to target {target1.target_id}")

    # 测试雷达网络释放通道
    radar_network_1.release_radar_channel(assigned_radar_id, assigned_channel_id)
    print(f"Radar network released channel {assigned_channel_id} for radar {assigned_radar_id}")
else:
    print(f"Radar network failed to assign radar to target {target1.target_id}")


# 测试雷达网络检查雷达是否可用
# 假设 radar_model.py 中 RadarNetwork.is_radar_available 已将 radar.is_available() 改为 radar.is_radar_available()
rn_available_test1 = radar_network_1.is_radar_available(radar1.radar_id)
rn_available_test2 = radar_network_1.is_radar_available(radar2.radar_id) # radar2 的通道可能已被 target1 占用后释放
print(f"Is radar1 available in network (after potential assignment/release): {rn_available_test1}")
print(f"Is radar2 available in network (after potential assignment/release): {rn_available_test2}")

# 测试重置所有通道
print("Resetting all channels in the network...")
radar_network_1.reset_all_channels()
rn_available_test1_after_reset = radar_network_1.is_radar_available(radar1.radar_id)
rn_available_test2_after_reset = radar_network_1.is_radar_available(radar2.radar_id)
print(f"Is radar1 available in network after reset: {rn_available_test1_after_reset}")
print(f"Is radar2 available in network after reset: {rn_available_test2_after_reset}")

# 验证 radar1 的通道是否真的被重置了
print(f"Radar1 channels after network reset: {radar1.radar_channels}")


