#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：radar
@File    ：radar_test.py
@IDE     ：PyCharm
@Author  ：ReznovLee
@Date    ：2025/4/14 10:45
"""
from src.core.models.radar_model import Radar, RadarNetwork
from src.core.models.target_model import TargetModel


radar1 = Radar(radar_id=0, radar_position=[0,0,0], radar_radius=1000, num_channels=5)
radar2 = Radar(radar_id=1, radar_position=[10,10,0], radar_radius=1500, num_channels=6)
radar_network_1 = RadarNetwork([radar1, radar2])
target1 = TargetModel(target_id=0, target_position=[100,100,100], velocity=[10,0,0], target_type="aircraft", priority=3)

available_sign = radar1.is_radar_available()
print("available_sign:", available_sign, "\n")

channel_test = radar2.assign_channel(target1.target_id)
print("channel_test:", channel_test, "\n")

print("is release:", radar2.release_channel(radar2.radar_channels[0]), "\n")

range_test = radar1.is_target_in_range(target1.target_position)
print("range_test:", range_test, "\n")

print("\n" + "-"*50 + "\n")

covering_test = radar_network_1.find_covering_radars(target1.target_position)
print("covering_test:", covering_test, "\n")

rn_test = radar_network_1.assign_radar(target1.target_id, target1.target_position)
print("rn_test:", rn_test, "\n")

release_test = radar_network_1.release_radar_channel(target1.target_id, target1.target_position)
print("release_test", release_test, "\n")

rn_available_test = radar_network_1.is_radar_available(radar1.radar_id)
print("rn_available_test", rn_available_test, "\n")

