# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : radar_model.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:25
"""
import logging
import numpy as np


class Radar:
    """ Basic class for single radar model

    The main role of this class is to provide basic properties of radar, including basic parameters based on radar
    equations.

    Attributes:
        radar_id:       Each radar has a basic ID (int), which is used to distinguish between different radars.
        radar_position: The 3D coordinates of the radar center, which is used to locate the radar.
        radar_radius:   The maximum radar tracking range is derived from the radar equation in an ideal situation.
                            In this project, the range of each radar is a constant.
                            For details, please refer to the paper.
        num_channels:   To simplify the calculation, each radar will assume the existence of a number of channels,
                            and each channel can perform only one task at any time,
                            and no other tasks can be performed unless the channel is released.
    """

    def __init__(self, radar_id, radar_position, radar_radius, num_channels):
        """ Initializing Radar class

        The radar class is initialized, and the radar basic parameter information is provided according to the ideal
        radar equation.

        :param radar_id: Radar ID
        :param 3-dimensional coordinates (x, y, z) of the radar center (It is assumed that the radar is located on the
               ground at 0 elevation at the time of simulation, so the z coordinate is 0)
        :param radar_radius: Maximum radiation radius of radar
        :param num_channels: Maximum number of channels available for the radar
        """
        assert num_channels > 0, "Number of channels must be greater than 0"
        assert radar_radius > 0, "Radar radius must be greater than 0"

        self.radar_id = radar_id
        self.radar_position = radar_position
        self.radar_radius = radar_radius
        self.num_channels = num_channels
        self.radar_channels = {i: None for i in range(num_channels)}

    def is_radar_available(self):
        """ Checks if the radar is available

        Check if the radar is available, which is determined by checking if the current radar has an available channel.

        :return: Check if the current radar is available -> bool
        """
        return any(channel is None for channel in self.radar_channels.values())

    def allocate_channel(self, target_id):
        """Assign the current destination to one of the channels of the current ID"""
        for channel_id in range(self.num_channels):
            if self.radar_channels[channel_id] is None:
                self.radar_channels[channel_id] = target_id
                print(f"Radar {self.radar_id} assign channel {channel_id} to target {target_id} \n")
                return channel_id
        return None

    def release_channel(self, channel_id):
        """ Release the current channel

        When the target is no longer tracked (including being intercepted, actively disappearing, out of range),
        the channel of the corresponding target needs to be released.

        :param channel_id: Channel ID
        :return: Whether to release the channel -> bool
        """
        if channel_id not in self.radar_channels:
            logging.error(f"The channel ID {channel_id} does not exist. \n")
            return False

        if self.radar_channels[channel_id] is not None:
            self.radar_channels[channel_id] = None
            print(f"Radar {self.radar_id} released channel {channel_id}. \n")
            return True
        else:
            print(f"The channel {channel_id} of the radar {self.radar_id} is already idle. \n")
            return False

    def is_target_in_range(self, target_position):
        """ Checks if the target position is within the radar range

        The detection of whether the target is within the radiation range of the corresponding id radar is used for
        the initial screening of available radars.

        :param target_position: Target position
        :return: Whether the target position is within the radar range -> bool
        """

        distance = np.linalg.norm(self.radar_position - target_position)

        if distance <= self.radar_radius:
            return True
        else:
            # print(f"The target position is outside the range")
            return False


class RadarNetwork:
    """
    The RadarNetwork class is used to define the network of radars.
    """

    def __init__(self, radars):
        """ Initializing RadarNetwork class

        The radar network is initialized, and the radar basic parameter information is provided.

        :param radars: List of Radar objects
        """
        self.radars = radars
        if isinstance(self.radars, dict):
            self.radar_ids = list(self.radars.keys())
        elif isinstance(self.radars, list):
            self.radar_ids = [r.radar_id for r in self.radars]
        else:
            self.radar_ids = []

    def find_covering_radars(self, target_position):
        """ Find all radars covering the target position

        Find the target that is within the total range of radar network and return to the available radar list

        :param target_position: Target position
        :return: list of Radar objects -> list  # <--- 修改了文档字符串的返回值说明
        """
        covering_radars = []
        for radar in self.radars:
            # 假设 self.radars 是一个包含 Radar 对象的列表或字典值
            if isinstance(radar, Radar) and radar.is_target_in_range(target_position): # 增加类型检查确保是Radar对象
                covering_radars.append(radar)
            # 如果 self.radars 是字典 {radar_id: Radar_object}，则迭代方式可能需要调整
            # 例如: for radar_id, radar_obj in self.radars.items():
            #          if radar_obj.is_target_in_range(target_position):
            #              covering_radars.append(radar_obj)

        # 可选：按可用通道数排序（如果需要）
        # covering_radars.sort(
        #     key=lambda r: sum(1 for c in r.radar_channels.values() if c is None),
        #     reverse=True
        # )

        # --- 修改点：直接返回包含 Radar 对象的列表 ---
        # radar_ids = [radar.radar_id for radar in covering_radars] # 原代码：返回ID列表
        logging.debug(f"Find {len(covering_radars)} radar(s) covering target position {target_position}")
        # return radar_ids # 原代码
        return covering_radars # 修改后：返回Radar对象列表
        # --- 修改结束 ---

    def assign_radar(self, target_id, target_position):
        """The radar is assigned to the target

        The corresponding radar with channel is assigned to the target.

        :param target_id: Target ID
        :param target_position: The 3D coordinates of the target
        :return: Tuple (Radar ID, Channel ID) or None -> tuple or None # <--- 修改了返回值说明
        """
        # --- 修改点：find_covering_radars 现在返回 Radar 对象列表 ---
        # covering_radars_ids = self.find_covering_radars(target_position) # 原代码变量名
        covering_radars_objects = self.find_covering_radars(target_position) # 修改后变量名，更清晰
        # --- 修改结束 ---

        # --- 修改点：直接迭代 Radar 对象列表 ---
        # for radar_id in covering_radars_ids: # 原代码
        #     radar = next((r for r in self.radars if r.radar_id == radar_id), None) # 原代码需要查找对象
        for radar in covering_radars_objects: # 修改后：直接使用对象
            if radar: # 确保对象有效 (虽然从 find_covering_radars 返回的应该都是有效的)
                # --- 修改点：Radar 类中没有 assign_channel 方法，应使用 allocate_channel ---
                # channel_id = radar.assign_channel(target_id) # 原代码，方法名不匹配 Radar 类
                channel_id = radar.allocate_channel(target_id) # 修改后：使用 Radar 类中定义的 allocate_channel
                # --- 修改结束 ---
                if channel_id is not None:
                    # 返回分配成功的雷达ID和通道ID
                    return radar.radar_id, channel_id
        # --- 修改结束 ---

        logging.warning(f"Target {target_id} has no radar channel available at position {target_position}")
        return None, None # 返回两个 None，保持一致性

    def release_radar_channel(self, radar_id, channel_id):
        """ Release the radar and its channel

        Release the radar and its channel.

        :param radar_id: Radar ID
        :param channel_id: The corresponding channel number of the radar
        """
        if radar_id in self.radars:
            self.radars[radar_id].release_channel(channel_id)
            print(f"The channel {channel_id} of radar {radar_id} has been released. \n")
        else:
            logging.error(f"Error: Trying to free a channel {channel_id} for a nonexistent radar {radar_id}")

    def is_radar_available(self, radar_id):
        """ Checks if the radar is available

        Check the enactment radar for the existence of idle channels

        :param radar_id: Radar ID
        :return: Whether the radar is available -> bool
        """
        return radar_id in self.radars and self.radars[radar_id].is_radar_available()

    def reset_all_channels(self):
        """ Resets the channel status for all radars in the network. """
        if isinstance(self.radars, list):
            for radar in self.radars:
                if isinstance(radar, Radar):
                    # 假设 Radar 类有一个 reset_channels 方法或直接操作 radar_channels
                    radar.radar_channels = {i: None for i in range(radar.num_channels)}
                    logging.debug(f"Reset channels for radar {radar.radar_id}")
        elif isinstance(self.radars, dict):
             for radar_id, radar in self.radars.items():
                 if isinstance(radar, Radar):
                    radar.radar_channels = {i: None for i in range(radar.num_channels)}
                    logging.debug(f"Reset channels for radar {radar.radar_id}")
        else:
            logging.error("RadarNetwork.radars is not a list or dict, cannot reset channels.")
