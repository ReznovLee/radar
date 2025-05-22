# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project: radar
@File   : plotter.py
@IDE    : PyCharm
@Author : Reznov Lee
@Date   : 2025/02/15 14:28
"""

import matplotlib.pyplot as plt
"""import matplotlib
matplotlib.rc("font",family='TimeNewRoman')"""
import numpy as np
from typing import List, Dict, Tuple, Any, Optional  # 删除typing
from scipy.sparse import csr_matrix
import os
import logging


class RadarPlotter:
    """
    雷达资源分配算法可视化工具
    用于生成各种图表来评估和比较不同算法的性能
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        初始化绘图器
        
        Args:
            figsize: 图表尺寸
            dpi: 图表分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        # 设置默认颜色方案
        self.target_colors = plt.cm.tab20.colors
        self.radar_colors = plt.cm.Set2.colors
        # 设置默认样式
        plt.style.use('ggplot')
        
    def _create_figure(self, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        创建新的图表
        
        Args:
            title: 图表标题
            
        Returns:
            fig, ax: 图表对象和坐标轴对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        if title:
            fig.suptitle(title, fontsize=16)
        return fig, ax
    
    def plot_radar_gantt(self, 
                     allocation_data, 
                     time_range,
                     radar_info,
                     target_info,
                     save_path=None):
        fig, ax = self._create_figure("Gantt chart of radar resource allocation")
    
        # 准备Y轴标签和位置
        y_labels = []
        y_ticks = []
        current_y = 0
        radar_y_positions = {}
    
        # 为每个雷达和通道分配Y轴位置
        for radar_id, channel_count in radar_info.items():
            radar_y_positions[radar_id] = {}
            for channel_id in range(channel_count):
                y_labels.append(f"R{radar_id}-C{channel_id}")  # 简化标签
                y_ticks.append(current_y)
                radar_y_positions[radar_id][channel_id] = current_y
                current_y += 1
    
        # 设置Y轴
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
    
        # 设置X轴
        ax.set_xlim(time_range)
        ax.set_xlabel("时间")
    
        # 绘制分配结果
        target_handles = {}
        for target_id in target_info.keys():
            target_color_idx = int(target_id) % len(self.target_colors)
            target_color = self.target_colors[target_color_idx]
            bar = ax.barh(-1, 0, color=target_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            target_handles[target_id] = bar
    
        drawn_targets = set()
        for alloc in allocation_data:
            timestamp = alloc['timestamp']
            assignments = alloc['assignments']
            
            for target_id, assignment in assignments.items():
                if assignment['radar_id'] is not None:
                    radar_id = assignment['radar_id']
                    channel_id = assignment['channel_id']
                    target_color_idx = int(target_id) % len(self.target_colors)
                    target_color = self.target_colors[target_color_idx]
                    
                    y_pos = radar_y_positions[radar_id][channel_id]
                    ax.barh(y_pos, 
                           width=1,  # 每个时间步长度为1
                           left=timestamp,
                           height=0.8, 
                           color=target_color, 
                           alpha=0.8,
                           edgecolor='black',
                           linewidth=0.5)
                    drawn_targets.add(target_id)
    
        # 添加简化的图例
        if len(target_handles) > 20:  # 如果目标数量过多，只显示部分图例
            selected_targets = sorted(target_handles.keys())[:20]  # 选择前20个目标
            legend_elements = [target_handles[tid][0] for tid in selected_targets]
            legend_labels = [f"T{tid}" for tid in selected_targets]
            legend_labels.append("...")  # 添加省略号表示还有更多目标
        else:
            legend_elements = [target_handles[tid][0] for tid in target_handles.keys()]
            legend_labels = [f"T{tid}" for tid in target_handles.keys()]
    
        ax.legend(legend_elements, legend_labels, loc='center left', bbox_to_anchor=(1.01, 0.5),
                 ncol=max(1, len(legend_labels)//20))  # 根据数量调整图例列数
    
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_radar_channel_occupancy_over_time(self,
                                           assignment_histories,
                                           radar_info,
                                           time_range,
                                           save_path=None):
        """
        绘制随时间变化的雷达信道占用率曲线。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            time_range: 时间范围元组 (start, end)
            save_path: 保存路径
        """
        fig, ax = self._create_figure("雷达信道占用率随时间变化")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算每个时间步的信道占用率
        for algo_name, history in assignment_histories.items():
            timestamps = []
            occupancy_rates = []
            
            for entry in history:
                timestamp = entry['timestamp']
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                
                timestamps.append(timestamp)
                occupancy_rates.append(occupancy_rate)
            
            # 绘制曲线
            ax.plot(timestamps, occupancy_rates, label=algo_name, marker='.', alpha=0.7)
        
        ax.set_xlim(time_range)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("时间")
        ax.set_ylabel("信道占用率")
        ax.set_title("雷达信道占用率随时间变化")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_average_radar_channel_occupancy(self,
                                         assignment_histories,
                                         radar_info,
                                         save_path=None):
        """
        绘制各算法的平均雷达信道占用率条形图。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            save_path: 保存路径
        """
        fig, ax = self._create_figure("各算法平均雷达信道占用率")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算平均信道占用率
        algo_names = []
        avg_occupancy_rates = []
        
        for algo_name, history in assignment_histories.items():
            occupancy_rates = []
            
            for entry in history:
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                occupancy_rates.append(occupancy_rate)
            
            # 计算平均占用率
            avg_occupancy_rate = sum(occupancy_rates) / len(occupancy_rates) if occupancy_rates else 0
            
            algo_names.append(algo_name)
            avg_occupancy_rates.append(avg_occupancy_rate)
        
        # 绘制条形图
        bars = ax.bar(algo_names, avg_occupancy_rates, 
                     color=[self.radar_colors[i % len(self.radar_colors)] for i in range(len(algo_names))],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(avg_occupancy_rates) * 1.2 if avg_occupancy_rates else 1.0)
        ax.set_ylabel("平均信道占用率")
        ax.set_title("各算法平均雷达信道占用率")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_target_gantt(self, 
                     allocation_data, 
                     time_range,
                     target_info,
                     radar_info,
                     save_path=None):
        fig, ax = self._create_figure("目标跟踪分配甘特图")

        # 准备Y轴标签和位置
        target_ids = sorted([str(tid) for tid in target_info.keys()])
        y_labels = [f"Target{target_id}" for target_id in target_ids]
        y_ticks = list(range(len(target_ids)))
        target_y_positions = {target_id: i for i, target_id in enumerate(target_ids)}

        # 设置Y轴
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        # 设置X轴
        ax.set_xlim(time_range)
        ax.set_xlabel("Time")

        # 绘制分配结果
        radar_handles = {}  # 用于图例，确保所有雷达都显示
        for radar_id in radar_info.keys():
            radar_color_idx = int(radar_id) % len(self.radar_colors)
            radar_color = self.radar_colors[radar_color_idx]
            # 创建一个不可见的bar用于图例
            bar = ax.barh(-1, 0, color=radar_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            radar_handles[radar_id] = bar

        drawn_radars = set()
        current_assignments = {}  # 用于跟踪当前分配状态
        
        # 遍历时间步骤
        for t_idx, alloc in enumerate(allocation_data):
            timestamp = alloc['timestamp']
            assignments = alloc['assignments']
            
            # 更新当前分配状态
            for target_id, assignment in assignments.items():
                if assignment is not None and assignment['radar_id'] is not None:
                    radar_id = assignment['radar_id']
                    if target_id not in current_assignments or current_assignments[target_id]['radar_id'] != radar_id:
                        # 如果是新分配或分配发生改变
                        if target_id in current_assignments:
                            # 结束上一个分配
                            prev_assignment = current_assignments[target_id]
                            y_pos = target_y_positions[target_id]
                            radar_color_idx = int(prev_assignment['radar_id']) % len(self.radar_colors)
                            radar_color = self.radar_colors[radar_color_idx]
                            
                            width = timestamp - prev_assignment['start_time']
                            if width > 0:
                                rect = ax.barh(y_pos,
                                             width=width,
                                             left=prev_assignment['start_time'],
                                             height=0.8,
                                             color=radar_color,
                                             alpha=0.8,
                                             edgecolor='black',
                                             linewidth=0.5)
                                
                                # 添加雷达ID标签
                                if width > (time_range[1] - time_range[0]) / 30:
                                    ax.text(prev_assignment['start_time'] + width/2,
                                          y_pos,
                                          f"R{prev_assignment['radar_id']}",
                                          ha='center',
                                          va='center',
                                          fontsize=8,
                                          color='black')
                        
                        # 记录新分配
                        current_assignments[target_id] = {
                            'radar_id': radar_id,
                            'start_time': timestamp
                        }
                elif target_id in current_assignments:
                    # 目标不再被分配，结束当前分配
                    prev_assignment = current_assignments[target_id]
                    y_pos = target_y_positions[target_id]
                    radar_color_idx = int(prev_assignment['radar_id']) % len(self.radar_colors)
                    radar_color = self.radar_colors[radar_color_idx]
                    
                    width = timestamp - prev_assignment['start_time']
                    if width > 0:
                        rect = ax.barh(y_pos,
                                     width=width,
                                     left=prev_assignment['start_time'],
                                     height=0.8,
                                     color=radar_color,
                                     alpha=0.8,
                                     edgecolor='black',
                                     linewidth=0.5)
                        
                        # 添加雷达ID标签
                        if width > (time_range[1] - time_range[0]) / 30:
                            ax.text(prev_assignment['start_time'] + width/2,
                                  y_pos,
                                  f"R{prev_assignment['radar_id']}",
                                  ha='center',
                                  va='center',
                                  fontsize=8,
                                  color='black')
                    del current_assignments[target_id]

        # 处理最后一个时间步的分配
        for target_id, assignment in current_assignments.items():
            y_pos = target_y_positions[target_id]
            radar_color_idx = int(assignment['radar_id']) % len(self.radar_colors)
            radar_color = self.radar_colors[radar_color_idx]
            
            width = time_range[1] - assignment['start_time']
            if width > 0:
                rect = ax.barh(y_pos,
                             width=width,
                             left=assignment['start_time'],
                             height=0.8,
                             color=radar_color,
                             alpha=0.8,
                             edgecolor='black',
                             linewidth=0.5)
                
                # 添加雷达ID标签
                if width > (time_range[1] - time_range[0]) / 30:
                    ax.text(assignment['start_time'] + width/2,
                          y_pos,
                          f"R{assignment['radar_id']}",
                          ha='center',
                          va='center',
                          fontsize=8,
                          color='black')

        # 添加图例，确保所有雷达都显示
        legend_elements = [radar_handles[rid][0] for rid in radar_info.keys()]
        legend_labels = [f"Radar{rid}" for rid in radar_info.keys()]
        ax.legend(legend_elements, legend_labels, loc='upper right', bbox_to_anchor=(1.15, 1))

        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_radar_channel_occupancy_over_time(self,
                                           assignment_histories,
                                           radar_info,
                                           time_range,
                                           save_path=None):
        """
        绘制随时间变化的雷达信道占用率曲线。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            time_range: 时间范围元组 (start, end)
            save_path: 保存路径
        """
        fig, ax = self._create_figure("雷达信道占用率随时间变化")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算每个时间步的信道占用率
        for algo_name, history in assignment_histories.items():
            timestamps = []
            occupancy_rates = []
            
            for entry in history:
                timestamp = entry['timestamp']
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                
                timestamps.append(timestamp)
                occupancy_rates.append(occupancy_rate)
            
            # 绘制曲线
            ax.plot(timestamps, occupancy_rates, label=algo_name, marker='.', alpha=0.7)
        
        ax.set_xlim(time_range)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("时间")
        ax.set_ylabel("信道占用率")
        ax.set_title("雷达信道占用率随时间变化")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_average_radar_channel_occupancy(self,
                                         assignment_histories,
                                         radar_info,
                                         save_path=None):
        """
        绘制各算法的平均雷达信道占用率条形图。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            save_path: 保存路径
        """
        fig, ax = self._create_figure("各算法平均雷达信道占用率")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算平均信道占用率
        algo_names = []
        avg_occupancy_rates = []
        
        for algo_name, history in assignment_histories.items():
            occupancy_rates = []
            
            for entry in history:
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                occupancy_rates.append(occupancy_rate)
            
            # 计算平均占用率
            avg_occupancy_rate = sum(occupancy_rates) / len(occupancy_rates) if occupancy_rates else 0
            
            algo_names.append(algo_name)
            avg_occupancy_rates.append(avg_occupancy_rate)
        
        # 绘制条形图
        bars = ax.bar(algo_names, avg_occupancy_rates, 
                     color=[self.radar_colors[i % len(self.radar_colors)] for i in range(len(algo_names))],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(avg_occupancy_rates) * 1.2 if avg_occupancy_rates else 1.0)
        ax.set_ylabel("平均信道占用率")
        ax.set_title("各算法平均雷达信道占用率")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_target_radar_switching_frequency(self,
                                              switching_data: Dict[str, int],
                                              save_path: Optional[str] = None):
        """
        绘制各算法的总目标雷达切换频次对比图。

        Args:
            switching_data: 字典，键为算法名称，值为该算法的总切换次数。
                            Example: {"BFSA-RHO": 15, "Rule-Based": 25, "LNS": 10}
            save_path: 保存路径。
        """
        algo_names = list(switching_data.keys())
        switch_counts = list(switching_data.values())

        fig, ax = self._create_figure("Target Radar Switching Frequency Comparison")

        bars = ax.bar(algo_names, switch_counts, color=[self.radar_colors[i % len(self.radar_colors)] for i in range(len(algo_names))])

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Total Radar Switches")
        ax.set_title("Comparison of Total Target Radar Switching Frequency by Algorithm")
        
        # 在条形图上添加数值标签
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05 * max(switch_counts) if max(switch_counts) > 0 else 0.5, 
                    int(yval), ha='center', va='bottom')

        plt.xticks(rotation=15, ha="right") # 如果标签重叠，旋转标签
        plt.tight_layout()

        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logging.info(f"Target radar switching frequency plot saved to {save_path}")
            plt.close(fig) # 关闭图形，防止在某些环境中意外显示
        else:
            plt.show()

    def plot_radar_channel_occupancy_over_time(self,
                                           assignment_histories,
                                           radar_info,
                                           time_range,
                                           save_path=None):
        """
        绘制随时间变化的雷达信道占用率曲线。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            time_range: 时间范围元组 (start, end)
            save_path: 保存路径
        """
        fig, ax = self._create_figure("雷达信道占用率随时间变化")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算每个时间步的信道占用率
        for algo_name, history in assignment_histories.items():
            timestamps = []
            occupancy_rates = []
            
            for entry in history:
                timestamp = entry['timestamp']
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                
                timestamps.append(timestamp)
                occupancy_rates.append(occupancy_rate)
            
            # 绘制曲线
            ax.plot(timestamps, occupancy_rates, label=algo_name, marker='.', alpha=0.7)
        
        ax.set_xlim(time_range)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("时间")
        ax.set_ylabel("信道占用率")
        ax.set_title("雷达信道占用率随时间变化")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_average_radar_channel_occupancy(self,
                                         assignment_histories,
                                         radar_info,
                                         save_path=None):
        """
        绘制各算法的平均雷达信道占用率条形图。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            save_path: 保存路径
        """
        fig, ax = self._create_figure("各算法平均雷达信道占用率")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算平均信道占用率
        algo_names = []
        avg_occupancy_rates = []
        
        for algo_name, history in assignment_histories.items():
            occupancy_rates = []
            
            for entry in history:
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                occupancy_rates.append(occupancy_rate)
            
            # 计算平均占用率
            avg_occupancy_rate = sum(occupancy_rates) / len(occupancy_rates) if occupancy_rates else 0
            
            algo_names.append(algo_name)
            avg_occupancy_rates.append(avg_occupancy_rate)
        
        # 绘制条形图
        bars = ax.bar(algo_names, avg_occupancy_rates, 
                     color=[self.radar_colors[i % len(self.radar_colors)] for i in range(len(algo_names))],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(avg_occupancy_rates) * 1.2 if avg_occupancy_rates else 1.0)
        ax.set_ylabel("平均信道占用率")
        ax.set_title("各算法平均雷达信道占用率")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_gantt_chart(self, 
                        assignments: List[csr_matrix], 
                        time_steps: List[int], 
                        mode: str, 
                        save_path: Optional[str] = None) -> None:
        """
        绘制调度甘特图：
        
        Args:
            assignments: 分配矩阵列表，每个元素是一个时间步的稀疏分配矩阵
            time_steps: 时间步列表
            mode: 显示模式，"target" 表示目标 vs 时间，"radar" 表示雷达 vs 时间
            save_path: 保存路径，如果为None则显示图表
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        assignment_matrix = np.array([a.toarray() for a in assignments])

        if mode == "target":
            plt.imshow(assignment_matrix.argmax(axis=2), cmap='tab10', aspect='auto')
            plt.ylabel('Target ID')
            plt.title('Target Scheduling Gantt Chart')
        elif mode == "radar":
            plt.imshow(assignment_matrix.argmax(axis=1).T, cmap='tab10', aspect='auto')
            plt.ylabel('Radar ID')
            plt.title('Radar Scheduling Gantt Chart')

        plt.xlabel('Timestep')
        plt.colorbar(label="Assignment")
        
        # 保存或显示图表
        self._save_or_show(save_path)
    
    def _save_or_show(self, save_path: Optional[str], fig: Optional[plt.Figure] = None) -> None:
        """
        Helper function to save or show the plot.
        If fig is None, it uses plt.gcf().
        """
        current_fig = fig if fig else plt.gcf()
        if save_path:
            # 确保目录存在
            if os.path.dirname(save_path): # Check if dirname is not empty
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            current_fig.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
            plt.close(current_fig)
        else:
            plt.show()

    def plot_radar_channel_occupancy_over_time(self,
                                           assignment_histories,
                                           radar_info,
                                           time_range,
                                           save_path=None):
        """
        绘制随时间变化的雷达信道占用率曲线。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            time_range: 时间范围元组 (start, end)
            save_path: 保存路径
        """
        fig, ax = self._create_figure("雷达信道占用率随时间变化")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算每个时间步的信道占用率
        for algo_name, history in assignment_histories.items():
            timestamps = []
            occupancy_rates = []
            
            for entry in history:
                timestamp = entry['timestamp']
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                
                timestamps.append(timestamp)
                occupancy_rates.append(occupancy_rate)
            
            # 绘制曲线
            ax.plot(timestamps, occupancy_rates, label=algo_name, marker='.', alpha=0.7)
        
        ax.set_xlim(time_range)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("时间")
        ax.set_ylabel("信道占用率")
        ax.set_title("雷达信道占用率随时间变化")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_average_radar_channel_occupancy(self,
                                         assignment_histories,
                                         radar_info,
                                         save_path=None):
        """
        绘制各算法的平均雷达信道占用率条形图。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            save_path: 保存路径
        """
        fig, ax = self._create_figure("各算法平均雷达信道占用率")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算平均信道占用率
        algo_names = []
        avg_occupancy_rates = []
        
        for algo_name, history in assignment_histories.items():
            occupancy_rates = []
            
            for entry in history:
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                occupancy_rates.append(occupancy_rate)
            
            # 计算平均占用率
            avg_occupancy_rate = sum(occupancy_rates) / len(occupancy_rates) if occupancy_rates else 0
            
            algo_names.append(algo_name)
            avg_occupancy_rates.append(avg_occupancy_rate)
        
        # 绘制条形图
        bars = ax.bar(algo_names, avg_occupancy_rates, 
                     color=[self.radar_colors[i % len(self.radar_colors)] for i in range(len(algo_names))],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(avg_occupancy_rates) * 1.2 if avg_occupancy_rates else 1.0)
        ax.set_ylabel("平均信道占用率")
        ax.set_title("各算法平均雷达信道占用率")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_target_switching(self,
                              algorithms_switch_data: Dict[str, List[Dict[str, Any]]], # Key: algo_name, Value: list of switch events
                              target_info: Dict[str, Any], # target_info is {target_id: target_details}
                              save_path: Optional[str] = None) -> None:
        """
        绘制目标切换频次图 (多算法对比)
        
        Args:
            algorithms_switch_data: 字典，键为算法名称，值为该算法的切换事件列表。
                                    每个切换事件是一个字典，例如 {'target_id': tid, 'radar_id': rid, 'start_time': ts}
            target_info: 目标信息字典，键为目标ID。
            save_path: 保存路径，如果为None则显示图表
        """
        fig, ax = self._create_figure("Target Radar Switching Frequency Comparison") # 通用标题
        
        all_target_ids = sorted([str(tid) for tid in target_info.keys()]) # 获取所有唯一的目标ID
        if not all_target_ids:
            logging.warning("No target information provided for switching plot.")
            if save_path:
                # Ensure directory exists even for an empty plot
                if os.path.dirname(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                else: # Handle case where save_path is just a filename in current dir
                    pass 
                ax.text(0.5, 0.5, "No target data", ha='center', va='center')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show() # 或直接返回
            return

        num_algorithms = len(algorithms_switch_data)
        if num_algorithms == 0:
            logging.warning("No algorithm data provided for switching plot.")
            if save_path:
                if os.path.dirname(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                ax.text(0.5, 0.5, "No algorithm data", ha='center', va='center')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show() # 或直接返回
            return

        algorithm_names = list(algorithms_switch_data.keys())
        
        # 计算每个算法和每个目标的切换次数
        switches_by_algo_target = {algo: {} for algo in algorithm_names}

        for algo_name, switch_data_list in algorithms_switch_data.items():
            target_radar_history_for_algo = {} # {target_id: [{'timestamp': ts, 'radar_id': rid}]}
            for record in switch_data_list:
                target_id = str(record['target_id']) 
                if target_id not in target_radar_history_for_algo:
                    target_radar_history_for_algo[target_id] = []
                target_radar_history_for_algo[target_id].append({
                    'timestamp': record['start_time'], # Using 'start_time' as the timestamp for sorting
                    'radar_id': record['radar_id']
                })
                
            current_algo_switches = {} # {target_id: count}
            for target_id, history in target_radar_history_for_algo.items():
                sorted_history = sorted(history, key=lambda x: x['timestamp'])
                switches = 0
                prev_radar = None
                for entry in sorted_history:
                    current_radar = entry['radar_id']
                    if prev_radar is not None and current_radar != prev_radar:
                        switches += 1
                    prev_radar = current_radar
                current_algo_switches[target_id] = switches
            switches_by_algo_target[algo_name] = current_algo_switches

        # 绘制分组条形图
        num_targets = len(all_target_ids)
        bar_width = 0.8 / num_algorithms 
        index = np.arange(num_targets)

        for i, algo_name in enumerate(algorithm_names):
            algo_switch_counts = [switches_by_algo_target[algo_name].get(tid, 0) for tid in all_target_ids]
            
            bars = ax.bar(index + i * bar_width, algo_switch_counts, bar_width, 
                          label=algo_name, 
                          color=plt.cm.tab10(i % 10), 
                          edgecolor='black')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                        f'{height}',
                        ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Targets')
        ax.set_ylabel('Radar Switch Counts')
        ax.set_xticks(index + bar_width * (num_algorithms - 1) / 2)
        ax.set_xticklabels([f"Target {tid}" for tid in all_target_ids], rotation=45, ha="right")
        ax.legend(title="Algorithms")
        
        plt.tight_layout()
        self._save_or_show(save_path)
    
    def plot_convergence_curve(self,
                              convergence_data: Dict[str, List[float]],
                              save_path: Optional[str] = None) -> None:
        """
        绘制算法收敛曲线
        
        Args:
            convergence_data: 收敛数据，格式为 {算法名称: [每时间步的覆盖率值, ...]}
            save_path: 保存路径，如果为None则显示图表
        """
        fig, ax = self._create_figure("Comparison of algorithm convergence curves")
        
        if not convergence_data or all(not v for v in convergence_data.values()):
            logging.warning("No convergence data provided or all data lists are empty. Skipping plot.")
            ax.text(0.5, 0.5, "No data available for convergence plot",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            self._save_or_show(save_path)
            if not save_path: # 如果仅显示，确保关闭图形
                 plt.close(fig)
            return

        plotted_something = False
        all_max_values_from_plots = []
        
        # 绘制每个算法的收敛曲线
        for i, (algorithm, values) in enumerate(convergence_data.items()):
            if not values:
                logging.info(f"Skipping convergence plot for {algorithm} as its data list is empty.")
                continue
            
            # iterations 实际上代表时间步
            timesteps_for_plot = list(range(1, len(values) + 1)) # X轴从1开始
            ax.plot(timesteps_for_plot, values, marker='o', markersize=4, label=algorithm,
                   color=plt.cm.tab10.colors[i % 10], linewidth=2)
            all_max_values_from_plots.append(max(values))
            plotted_something = True
        
        if not plotted_something:
            logging.warning("No valid algorithm data was plotted for convergence.")
            ax.text(0.5, 0.5, "No valid data to plot after filtering",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            self._save_or_show(save_path)
            if not save_path: # 如果仅显示，确保关闭图形
                 plt.close(fig)
            return

        # 设置坐标轴标签
        ax.set_xlabel('Timesteps') # 标签改为 Timesteps，更准确
        ax.set_ylabel('Tracking Coverage Ratio') # 标签简化，更清晰
        
        # 设置Y轴范围，从0到1或略高于最大值
        max_y_val = 0.0
        if all_max_values_from_plots: # 确保列表非空
            max_y_val = max(all_max_values_from_plots)
        
        ax.set_ylim(0, max(1.0, max_y_val * 1.05)) # 确保Y轴至少为0到1
        
        # 添加网格和图例
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best') # 使用 'best' 位置
        
        plt.tight_layout()
        self._save_or_show(save_path)
        if not save_path: # 如果仅显示，确保关闭图形
            plt.close(fig)
    def plot_target_assignment_rate_over_time(self,
                                              assignment_rate_data: Dict[str, List[float]],
                                              save_path: Optional[str] = None) -> None:
        """
        绘制不同算法的目标分配率随时间变化的曲线 (仅使用线条)
        
        Args:
            assignment_rate_data: 分配率数据，格式为 {算法名称: [每时间步的分配率值, ...]}
            save_path: 保存路径，如果为None则显示图表
        """
        # Construct title dynamically
        algo_names = list(assignment_rate_data.keys())
        if len(algo_names) > 0:
            title_str = "Assigned Ratio Comparison: " + " vs ".join(algo_names)
        else:
            title_str = "Assigned Ratio Comparison"
        fig, ax = self._create_figure(title_str) # _create_figure uses fig.suptitle
        
        if not assignment_rate_data or all(not v for v in assignment_rate_data.values()):
            logging.warning("No assignment rate data provided or all data lists are empty. Skipping plot.")
            ax.text(0.5, 0.5, "No data available for assignment rate plot",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            self._save_or_show(save_path, fig)
            return

        plotted_something = False

        # 为特定算法定义线条样式
        algorithm_specific_linestyles = {
            "BFSA-RHO": '-',   # 实线
            "Rule-Based": '--', # 虚线
            "LNS": ':'         # 点线
        }
        default_linestyle = '-.' # 其他算法的默认线型 (点划线)
        
        # 绘制每个算法的分配率曲线
        for i, (algorithm, values) in enumerate(assignment_rate_data.items()):
            if not values:
                logging.info(f"Skipping assignment rate plot for {algorithm} as its data list is empty.")
                continue
            
            timesteps_for_plot = list(range(len(values))) # X轴从0到T

            linestyle_to_use = algorithm_specific_linestyles.get(algorithm, default_linestyle)

            ax.plot(timesteps_for_plot, values,
                    linestyle=linestyle_to_use,  # 应用线条样式
                    label=algorithm,
                    color=plt.cm.tab10.colors[i % 10], # 保持颜色循环
                    linewidth=2) # 线条宽度
            plotted_something = True
        
        if not plotted_something:
            logging.warning("No valid algorithm data was plotted for assignment rate.")
            ax.text(0.5, 0.5, "No valid data to plot after filtering",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            self._save_or_show(save_path, fig)
            return

        # 设置坐标轴标签
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Assigned Ratio (Assigned / Total)')
        
        # 设置Y轴范围，固定为 0 到 1.1
        ax.set_ylim(0, 1.1)
        
        # 添加网格和图例
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best') 
        
        # 调整布局，在图表顶部留出4%的边距以容纳标题
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        self._save_or_show(save_path, fig)

    def plot_average_target_assignment_rate(self,
                                            assignment_rate_data: Dict[str, List[float]],
                                            save_path: Optional[str] = None) -> None:
        """
        绘制不同算法的平均目标分配率对比柱状图。

        Args:
            assignment_rate_data: 字典，键为算法名称，值为每个时间步的分配率列表。
                                  例如: {"BFSA-RHO": [0.8, 0.85], "Rule-Based": [0.7, 0.75]}
            save_path: 保存路径，如果为None则显示图表。
        """
        fig, ax = self._create_figure("Average Target Assignment Rate Comparison")

        if not assignment_rate_data or all(not v for v in assignment_rate_data.values()):
            logging.warning("No data for average target assignment rate plot.")
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            self._save_or_show(save_path, fig=fig)
            if not save_path:
                plt.close(fig)
            return

        algorithms = []
        average_rates = []

        for algo_name, rates in assignment_rate_data.items():
            if rates: # 确保列表不为空
                algorithms.append(algo_name)
                average_rates.append(np.mean(rates))
            else:
                logging.info(f"Skipping average calculation for {algo_name} due to empty rate list.")
        
        if not algorithms: # 如果所有算法的数据都为空或处理后没有有效数据
            logging.warning("No valid data to plot for average target assignment rate.")
            ax.text(0.5, 0.5, "No valid data to plot", ha='center', va='center')
            self._save_or_show(save_path, fig=fig)
            if not save_path:
                plt.close(fig)
            return

        bars = ax.bar(algorithms, average_rates, color=[plt.cm.tab10(i % 10) for i in range(len(algorithms))], edgecolor='black')

        ax.set_xlabel("Algorithms")
        ax.set_ylabel("Average Target Assignment Rate")
        ax.set_ylim(0, 1.1) # Y轴范围0到1.1，以便显示顶部的文本
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # 在柱状图顶部显示具体数值
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        self._save_or_show(save_path, fig=fig)
        if not save_path: # 如果仅显示，确保关闭图形
            plt.close(fig)

    def plot_assignment_rates_comparison(self,
                                         assignment_rate_data: Dict[str, List[float]],
                                         time_steps: List[int],
                                         title: str = "Comparison of Target Assignment Rates", # 更新默认标题为英文
                                         save_path: Optional[str] = None):
        """
        绘制多种算法的目标分配率对比图。
        风格与参考图片 /c:/Users/reznovlee/Desktop/radar_code/radar/output/scenario-2025-05-18/visualization/assignment_rates_comparison.png 一致：
        使用 'ggplot' 样式，线条带有 'o' 标记，英文标题和轴标签。

        Args:
            assignment_rate_data: 字典，键为算法名称，值为分配率列表.
            time_steps: 时间步列表.
            title: 图表标题.
            save_path: 保存路径，如果为None则显示图表.
        """
        fig, ax = self._create_figure(title)

        line_style = '-'
        marker_style = 'o'
        line_width = 1.5  # 可以根据实际效果调整线条宽度

        # 'ggplot' 样式会自动处理颜色循环
        for algo_name, rates in assignment_rate_data.items():
            ax.plot(time_steps, rates, label=algo_name,
                    marker=marker_style,
                    linestyle=line_style,
                    linewidth=line_width)

        ax.set_xlabel("Time Step")  # 更新X轴标签为英文
        ax.set_ylabel("Assignment Rate")  # 更新Y轴标签为英文
        
        ax.legend(loc='best')  # 显示图例，'best' 会自动选择最佳位置
        
        # 'ggplot' 样式通常包含网格, 这里明确设置以确保风格一致
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置Y轴范围，确保至少覆盖0.0到1.0
        min_display_rate = 0.0
        max_display_rate = 1.0
        if assignment_rate_data:
            all_rates_flat = [rate for rates_list in assignment_rate_data.values() for rate in rates_list]
            if all_rates_flat: # 确保有数据
                # 实际数据中的最小值和最大值
                actual_min_rate = min(all_rates_flat)
                actual_max_rate = max(all_rates_flat)
                # 更新显示范围以包含所有数据点，同时保持0-1的基本范围
                min_display_rate = min(min_display_rate, actual_min_rate)
                max_display_rate = max(max_display_rate, actual_max_rate)

        # 给Y轴顶部留一些空间
        ax.set_ylim(min_display_rate, max_display_rate * 1.05 if max_display_rate > 0 else 0.1)

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        if save_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(save_path)
            if output_dir: # 检查目录名是否为空
                 os.makedirs(output_dir, exist_ok=True)
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)  # 保存后关闭图形，释放内存
            logging.info(f"Saved assignment rates comparison chart to {save_path}")
        else:
            plt.show()

    def plot_radar_channel_occupancy_over_time(self,
                                           assignment_histories,
                                           radar_info,
                                           time_range,
                                           save_path=None):
        """
        绘制随时间变化的雷达信道占用率曲线。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            time_range: 时间范围元组 (start, end)
            save_path: 保存路径
        """
        fig, ax = self._create_figure("雷达信道占用率随时间变化")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算每个时间步的信道占用率
        for algo_name, history in assignment_histories.items():
            timestamps = []
            occupancy_rates = []
            
            for entry in history:
                timestamp = entry['timestamp']
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                
                timestamps.append(timestamp)
                occupancy_rates.append(occupancy_rate)
            
            # 绘制曲线
            ax.plot(timestamps, occupancy_rates, label=algo_name, marker='.', alpha=0.7)
        
        ax.set_xlim(time_range)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("时间")
        ax.set_ylabel("信道占用率")
        ax.set_title("雷达信道占用率随时间变化")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_average_radar_channel_occupancy(self,
                                         assignment_histories,
                                         radar_info,
                                         save_path=None):
        """
        绘制各算法的平均雷达信道占用率条形图。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            save_path: 保存路径
        """
        fig, ax = self._create_figure("各算法平均雷达信道占用率")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算平均信道占用率
        algo_names = []
        avg_occupancy_rates = []
        
        for algo_name, history in assignment_histories.items():
            occupancy_rates = []
            
            for entry in history:
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                occupancy_rates.append(occupancy_rate)
            
            # 计算平均占用率
            avg_occupancy_rate = sum(occupancy_rates) / len(occupancy_rates) if occupancy_rates else 0
            
            algo_names.append(algo_name)
            avg_occupancy_rates.append(avg_occupancy_rate)
        
        # 绘制条形图
        bars = ax.bar(algo_names, avg_occupancy_rates, 
                     color=[self.radar_colors[i % len(self.radar_colors)] for i in range(len(algo_names))],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(avg_occupancy_rates) * 1.2 if avg_occupancy_rates else 1.0)
        ax.set_ylabel("平均信道占用率")
        ax.set_title("各算法平均雷达信道占用率")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    
    def plot_boxplot(self, 
                    performance_data: Dict[str, List[float]],
                    metric_name: str = "Tracking coverage",
                    save_path: Optional[str] = None) -> None:
        """
        绘制箱线图比较不同算法的性能
        
        Args:
            performance_data: 性能数据，格式为 {算法名称: [运行1的性能值, 运行2的性能值, ...]}
            metric_name: 性能指标名称
            save_path: 保存路径，如果为None则显示图表
        """
        fig, ax = self._create_figure(f"Algorithm performance box plot - {metric_name}")
        
        # 准备数据
        data = []
        labels = []
        
        for algorithm, values in performance_data.items():
            data.append(values)
            labels.append(algorithm)
        
        # 绘制箱线图
        box_plot = ax.boxplot(data, patch_artist=True, labels=labels)
        
        # 设置箱体颜色
        for i, box in enumerate(box_plot['boxes']):
            box.set(facecolor=plt.cm.tab10.colors[i % 10], alpha=0.7)
        
        # 设置坐标轴标签
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(metric_name)
        
        # 添加网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 添加均值点
        for i, d in enumerate(data):
            mean_val = np.mean(d)
            ax.scatter(i+1, mean_val, marker='*', color='red', s=100, zorder=10)
            ax.text(i+1, mean_val, f'{mean_val:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_radar_channel_occupancy_over_time(self,
                                           assignment_histories,
                                           radar_info,
                                           time_range,
                                           save_path=None):
        """
        绘制随时间变化的雷达信道占用率曲线。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            time_range: 时间范围元组 (start, end)
            save_path: 保存路径
        """
        fig, ax = self._create_figure("雷达信道占用率随时间变化")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算每个时间步的信道占用率
        for algo_name, history in assignment_histories.items():
            timestamps = []
            occupancy_rates = []
            
            for entry in history:
                timestamp = entry['timestamp']
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                
                timestamps.append(timestamp)
                occupancy_rates.append(occupancy_rate)
            
            # 绘制曲线
            ax.plot(timestamps, occupancy_rates, label=algo_name, marker='.', alpha=0.7)
        
        ax.set_xlim(time_range)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("时间")
        ax.set_ylabel("信道占用率")
        ax.set_title("雷达信道占用率随时间变化")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_average_radar_channel_occupancy(self,
                                         assignment_histories,
                                         radar_info,
                                         save_path=None):
        """
        绘制各算法的平均雷达信道占用率条形图。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            save_path: 保存路径
        """
        fig, ax = self._create_figure("各算法平均雷达信道占用率")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算平均信道占用率
        algo_names = []
        avg_occupancy_rates = []
        
        for algo_name, history in assignment_histories.items():
            occupancy_rates = []
            
            for entry in history:
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                occupancy_rates.append(occupancy_rate)
            
            # 计算平均占用率
            avg_occupancy_rate = sum(occupancy_rates) / len(occupancy_rates) if occupancy_rates else 0
            
            algo_names.append(algo_name)
            avg_occupancy_rates.append(avg_occupancy_rate)
        
        # 绘制条形图
        bars = ax.bar(algo_names, avg_occupancy_rates, 
                     color=[self.radar_colors[i % len(self.radar_colors)] for i in range(len(algo_names))],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(avg_occupancy_rates) * 1.2 if avg_occupancy_rates else 1.0)
        ax.set_ylabel("平均信道占用率")
        ax.set_title("各算法平均雷达信道占用率")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_algorithm_comparison(self, 
                                 algorithms: List[str], 
                                 performance_metrics: Dict[str, Dict[str, float]],
                                 save_path: Optional[str] = None) -> None:
        """
        绘制雷达图比较多个算法的综合性能
        
        Args:
            algorithms: 算法名称列表
            performance_metrics: 性能指标数据，格式为 {指标名称: {算法名称: 值}}
            save_path: 保存路径，如果为None则显示图表
        """
        if not algorithms or not performance_metrics:
            logging.warning("No algorithms or performance metrics provided for radar chart.")
            if save_path:
                # Create an empty plot or a plot with a warning message
                fig, ax = self._create_figure("Algorithm Performance Comparison (No Data)")
                ax.text(0.5, 0.5, "No data to display", horizontalalignment='center', verticalalignment='center')
                if os.path.dirname(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            return

        metric_names = list(performance_metrics.keys())
        num_metrics = len(metric_names)
        
        if num_metrics == 0:
            logging.warning("No metric names found in performance_metrics for radar chart.")
            if save_path:
                fig, ax = self._create_figure("Algorithm Performance Comparison (No Metrics)")
                ax.text(0.5, 0.5, "No metrics to display", horizontalalignment='center', verticalalignment='center')
                if os.path.dirname(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
            return

        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  #闭合

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), dpi=self.dpi)
        fig.suptitle("Algorithm Performance Comparison", fontsize=16)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_yticks(np.arange(0, 1.1, 0.2)) # Y轴刻度从0到1
        ax.set_ylim(0, 1.05) # Y轴范围

        for i, algorithm_name in enumerate(algorithms):
            values = []
            for metric in metric_names:
                # Ensure the algorithm exists for the metric, otherwise default to 0 or handle as needed
                values.append(performance_metrics.get(metric, {}).get(algorithm_name, 0.0))
            
            values += values[:1] # 闭合
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=algorithm_name, color=plt.cm.tab10(i % 10))
            ax.fill(angles, values, alpha=0.25, color=plt.cm.tab10(i % 10))

        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        
        self._save_or_show(save_path)
    
    def plot_all_metrics(self, 
                        results: Dict[str, Any],
                        output_dir: str = "./results") -> None:
        """
        生成所有评价指标的图表
        
        Args:
            results: 包含所有算法结果的数据
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 为每个算法绘制雷达甘特图
        for algorithm, data in results['allocations'].items():
            self.plot_radar_gantt(
                data,
                results['time_range'],
                results['radar_info'],
                results['target_info'],
                save_path=f"{output_dir}/{algorithm}_radar_gantt.png"
            )
        
        # 2. 为每个算法绘制目标甘特图
        for algorithm, data in results['allocations'].items():
            self.plot_target_gantt(
                data,
                results['time_range'],
                results['target_info'],
                results['radar_info'],
                save_path=f"{output_dir}/{algorithm}_target_gantt.png"
            )
        
        # 3. 为每个算法绘制目标切换频次图
        for algorithm, data in results['allocations'].items():
            self.plot_target_switching(
                data,
                results['target_info'],
                algorithm_name=algorithm,
                save_path=f"{output_dir}/{algorithm}_target_switching.png"
            )
        
        # 4. 绘制所有算法的收敛曲线
        self.plot_convergence_curve(
            results['convergence_data'],
            save_path=f"{output_dir}/convergence_curves.png"
        )
        
        # 5. 绘制箱线图比较不同算法的性能
        for metric, data in results['performance_metrics'].items():
            self.plot_boxplot(
                data,
                metric_name=metric,
                save_path=f"{output_dir}/boxplot_{metric.replace(' ', '_')}.png"
            )
        
        # 6. 绘制算法综合性能对比图
        self.plot_algorithm_comparison(
            list(results['allocations'].keys()),
            results['average_metrics'],
            save_path=f"{output_dir}/algorithm_comparison.png"
        )

    def plot_radar_utilization_heatmap(
        self,
        allocation_history: List[Dict[str, Any]],
        radar_info: Dict[int, int],  # radar_id -> num_channels
        time_range,
        algorithm_name="",
        save_path=None
    ):
        """
        绘制雷达利用率热力图
        
        Args:
            allocation_history: 分配历史数据 (原为 assignment_history，已修正)
            radar_info: 雷达信息字典
            time_range: 时间范围元组 (start, end)
            algorithm_name: 算法名称
            save_path: 保存路径
        """
        # 创建雷达利用率矩阵
        # 检查radar_info是否为字典，如果不是，则假设它是时间范围
        if isinstance(radar_info, dict):
            radar_ids = sorted(radar_info.keys())
        else:
            # 如果radar_info不是字典，那么它可能是时间范围
            # 此时需要从allocation_history中提取雷达ID
            radar_ids = []
            for step in allocation_history: # MODIFIED: assignment_history -> allocation_history
                for _, assignment in step['assignments'].items():
                    if assignment is not None and assignment['radar_id'] is not None:
                        if assignment['radar_id'] not in radar_ids:
                            radar_ids.append(assignment['radar_id'])
            radar_ids = sorted(radar_ids)
            
            # 如果time_range是None，则使用radar_info作为time_range
            if time_range is None:
                time_range = radar_info

        
        time_steps = int(time_range[1] - time_range[0]) + 1
        utilization_matrix = np.zeros((len(radar_ids), time_steps))
        
        # 计算每个时间步每个雷达的利用率
        for step in allocation_history: # MODIFIED: assignment_history -> allocation_history
            t = int(step['timestamp'] - time_range[0])
            if t < 0 or t >= time_steps:
                continue
                
            # 统计每个雷达分配的通道数
            radar_channel_count = {rid: 0 for rid in radar_ids}
            # radar_total_channels = {rid: 0 for rid in radar_ids} # This line seems unused, can be kept or removed
            
            for _, assignment in step['assignments'].items():
                if assignment is not None and assignment['radar_id'] is not None:
                    radar_id = assignment['radar_id']
                    if radar_id in radar_channel_count:
                        radar_channel_count[radar_id] += 1
            
            # 计算利用率
            for i, rid in enumerate(radar_ids):
                # 如果radar_info是字典，使用其中的通道数
                if isinstance(radar_info, dict) and rid in radar_info:
                    total_channels = radar_info[rid]
                else:
                    # 否则假设每个雷达有4个通道（默认值）
                    # This default might need review based on actual radar_info structure when it's not a dict
                    total_channels = 4 # Default, consider if this logic is robust
                    
                if total_channels > 0: # Avoid division by zero
                    utilization_matrix[i, t] = radar_channel_count[rid] / total_channels
                else:
                    utilization_matrix[i, t] = 0.0
        
        # 绘制热力图

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        title = f"雷达利用率热力图 - {algorithm_name}" if algorithm_name else "雷达利用率热力图"
        ax.set_title(title)
        
        im = ax.imshow(utilization_matrix, cmap='hot', aspect='auto', vmin=0, vmax=1)
        
        # 设置坐标轴
        ax.set_xlabel('时间')
        ax.set_ylabel('雷达ID')
        ax.set_yticks(np.arange(len(radar_ids)))
        ax.set_yticklabels([f'R{rid}' for rid in radar_ids])
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('利用率 (已分配通道/总通道)')
        
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_radar_channel_occupancy_over_time(self,
                                           assignment_histories,
                                           radar_info,
                                           time_range,
                                           save_path=None):
        """
        绘制随时间变化的雷达信道占用率曲线。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            time_range: 时间范围元组 (start, end)
            save_path: 保存路径
        """
        fig, ax = self._create_figure("雷达信道占用率随时间变化")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算每个时间步的信道占用率
        for algo_name, history in assignment_histories.items():
            timestamps = []
            occupancy_rates = []
            
            for entry in history:
                timestamp = entry['timestamp']
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                
                timestamps.append(timestamp)
                occupancy_rates.append(occupancy_rate)
            
            # 绘制曲线
            ax.plot(timestamps, occupancy_rates, label=algo_name, marker='.', alpha=0.7)
        
        ax.set_xlim(time_range)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("时间")
        ax.set_ylabel("信道占用率")
        ax.set_title("雷达信道占用率随时间变化")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_average_radar_channel_occupancy(self,
                                         assignment_histories,
                                         radar_info,
                                         save_path=None):
        """
        绘制各算法的平均雷达信道占用率条形图。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            save_path: 保存路径
        """
        fig, ax = self._create_figure("各算法平均雷达信道占用率")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算平均信道占用率
        algo_names = []
        avg_occupancy_rates = []
        
        for algo_name, history in assignment_histories.items():
            occupancy_rates = []
            
            for entry in history:
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                occupancy_rates.append(occupancy_rate)
            
            # 计算平均占用率
            avg_occupancy_rate = sum(occupancy_rates) / len(occupancy_rates) if occupancy_rates else 0
            
            algo_names.append(algo_name)
            avg_occupancy_rates.append(avg_occupancy_rate)
        
        # 绘制条形图
        bars = ax.bar(algo_names, avg_occupancy_rates, 
                     color=[self.radar_colors[i % len(self.radar_colors)] for i in range(len(algo_names))],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(avg_occupancy_rates) * 1.2 if avg_occupancy_rates else 1.0)
        ax.set_ylabel("平均信道占用率")
        ax.set_title("各算法平均雷达信道占用率")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_priority_satisfaction(self,
                              allocation_histories,
                              time_range,
                              target_priority_info,
                              save_path=None):
        """
        绘制不同算法的优先级满足度比较图
        
        Args:
            allocation_histories: 不同算法的分配历史数据，格式为 {算法名称: 分配历史}
            time_range: 时间范围
            target_priority_info: 目标优先级信息，格式为 {目标ID: {'priority': 优先级值, 'type': 目标类型}}
            save_path: 保存路径
        """
        fig, ax = self._create_figure("目标优先级满足度比较")
        
        # 计算每个算法的优先级满足度
        algorithm_satisfaction = {}
        
        for algorithm_name, history in allocation_histories.items():
            # 确保history是列表类型
            if not isinstance(history, list):
                logging.error(f"算法 {algorithm_name} 的分配历史数据不是列表类型")
                continue
                
            # 初始化优先级统计
            priority_counts = {}  # {优先级: [分配次数, 总次数]}
            
            # 遍历每个时间步
            for alloc in history:
                # 确保alloc是字典类型
                if not isinstance(alloc, dict):
                    logging.error(f"算法 {algorithm_name} 的分配历史中包含非字典类型数据")
                    continue
                    
                # 确保alloc包含必要的键
                if 'timestamp' not in alloc or 'assignments' not in alloc:
                    logging.error(f"算法 {algorithm_name} 的分配历史数据格式不正确，缺少必要的键")
                    continue
                    
                timestamp = alloc['timestamp']
                assignments = alloc['assignments']
                
                # 遍历每个目标的分配情况
                for target_id, assignment in assignments.items():
                    # 获取目标优先级
                    if target_id not in target_priority_info:
                        continue
                        
                    priority = target_priority_info[target_id]['priority']
                    
                    if priority not in priority_counts:
                        priority_counts[priority] = [0, 0]
                    
                    priority_counts[priority][1] += 1  # 总次数加1
                    
                    # 如果目标被分配了雷达，则分配次数加1
                    if assignment is not None and assignment['radar_id'] is not None:
                        priority_counts[priority][0] += 1
            
            # 计算每个优先级的满足率
            satisfaction_rates = {}
            for priority, counts in priority_counts.items():
                if counts[1] > 0:
                    satisfaction_rates[priority] = counts[0] / counts[1]
                else:
                    satisfaction_rates[priority] = 0
            
            algorithm_satisfaction[algorithm_name] = satisfaction_rates
        
        # 绘制柱状图
        priorities = sorted(set().union(*[list(rates.keys()) for rates in algorithm_satisfaction.values()]))
        x = np.arange(len(priorities))
        width = 0.8 / len(algorithm_satisfaction)
        
        for i, (algorithm, rates) in enumerate(algorithm_satisfaction.items()):
            values = [rates.get(p, 0) for p in priorities]
            offset = (i - len(algorithm_satisfaction) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=algorithm)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 设置坐标轴
        ax.set_xlabel('目标优先级')
        ax.set_ylabel('满足率 (分配次数/总次数)')
        ax.set_title('不同算法的优先级满足度比较')
        ax.set_xticks(x)
        ax.set_xticklabels([f'P{p}' for p in priorities])
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_radar_channel_occupancy_over_time(self,
                                           assignment_histories,
                                           radar_info,
                                           time_range,
                                           save_path=None):
        """
        绘制随时间变化的雷达信道占用率曲线。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            time_range: 时间范围元组 (start, end)
            save_path: 保存路径
        """
        fig, ax = self._create_figure("雷达信道占用率随时间变化")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算每个时间步的信道占用率
        for algo_name, history in assignment_histories.items():
            timestamps = []
            occupancy_rates = []
            
            for entry in history:
                timestamp = entry['timestamp']
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                
                timestamps.append(timestamp)
                occupancy_rates.append(occupancy_rate)
            
            # 绘制曲线
            ax.plot(timestamps, occupancy_rates, label=algo_name, marker='.', alpha=0.7)
        
        ax.set_xlim(time_range)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("时间")
        ax.set_ylabel("信道占用率")
        ax.set_title("雷达信道占用率随时间变化")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_average_radar_channel_occupancy(self,
                                         assignment_histories,
                                         radar_info,
                                         save_path=None):
        """
        绘制各算法的平均雷达信道占用率条形图。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            save_path: 保存路径
        """
        fig, ax = self._create_figure("各算法平均雷达信道占用率")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算平均信道占用率
        algo_names = []
        avg_occupancy_rates = []
        
        for algo_name, history in assignment_histories.items():
            occupancy_rates = []
            
            for entry in history:
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                occupancy_rates.append(occupancy_rate)
            
            # 计算平均占用率
            avg_occupancy_rate = sum(occupancy_rates) / len(occupancy_rates) if occupancy_rates else 0
            
            algo_names.append(algo_name)
            avg_occupancy_rates.append(avg_occupancy_rate)
        
        # 绘制条形图
        bars = ax.bar(algo_names, avg_occupancy_rates, 
                     color=[self.radar_colors[i % len(self.radar_colors)] for i in range(len(algo_names))],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(avg_occupancy_rates) * 1.2 if avg_occupancy_rates else 1.0)
        ax.set_ylabel("平均信道占用率")
        ax.set_title("各算法平均雷达信道占用率")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_overall_performance(self, assignment_histories, targets_by_timestep, radar_info, save_path=None):
        """绘制算法综合性能评分图"""
        algorithms = list(assignment_histories.keys())
        
        # 定义评估指标和权重
        metrics = {
            '分配率': 0.25,
            '优先级满足率': 0.25,
            '跟踪连续性': 0.2,
            '负载均衡度': 0.15,
            '切换次数': 0.15
        }
        
        # 计算各指标得分
        scores = {algo: {} for algo in algorithms}
        
        # 1. 计算分配率
        for algo, history in assignment_histories.items():
            total_targets = 0
            assigned_targets = 0
            
            for record in history:
                assignments = record['assignments']
                total_targets += len(assignments)
                assigned_targets += sum(1 for a in assignments.values() if a is not None)
            
            scores[algo]['分配率'] = assigned_targets / total_targets if total_targets > 0 else 0
        
        # 2. 计算优先级满足率（加权）
        for algo, history in assignment_histories.items():
            total_priority_weight = 0
            assigned_priority_weight = 0
            
            for t_idx, record in enumerate(history):
                timestep = record['timestamp']
                if timestep not in targets_by_timestep:
                    continue
                    
                targets = targets_by_timestep[timestep]
                assignments = record['assignments']
                
                for target in targets:
                    if 'priority' not in target or 'id' not in target:
                        continue
                        
                    target_id = str(target['id'])
                    priority = target['priority']
                    total_priority_weight += priority
                    
                    if target_id in assignments and assignments[target_id] is not None:
                        assigned_priority_weight += priority
            
            scores[algo]['优先级满足率'] = assigned_priority_weight / total_priority_weight if total_priority_weight > 0 else 0
        
        # 3. 计算跟踪连续性
        for algo, history in assignment_histories.items():
            if len(history) <= 1:
                scores[algo]['跟踪连续性'] = 0
                continue
                
            continuity_count = 0
            total_possible = 0
            
            for t_idx in range(1, len(history)):
                prev_assignments = history[t_idx-1]['assignments']
                curr_assignments = history[t_idx]['assignments']
                
                # 找出两个时间步都存在的目标
                common_targets = set(prev_assignments.keys()) & set(curr_assignments.keys())
                total_possible += len(common_targets)
                
                # 计算连续跟踪的目标数
                for target_id in common_targets:
                    prev_radar = prev_assignments[target_id]['radar_id'] if prev_assignments[target_id] else None
                    curr_radar = curr_assignments[target_id]['radar_id'] if curr_assignments[target_id] else None
                    
                    if prev_radar is not None and prev_radar == curr_radar:
                        continuity_count += 1
            
            scores[algo]['跟踪连续性'] = continuity_count / total_possible if total_possible > 0 else 0
        
        # 4. 计算负载均衡度
        for algo, history in assignment_histories.items():
            load_variance_sum = 0
            
            for record in history:
                radar_loads = {r_id: 0 for r_id in radar_info}
                assignments = record['assignments']
                
                for assignment in assignments.values():
                    if assignment is not None and assignment['radar_id'] in radar_loads:
                        radar_loads[assignment['radar_id']] += 1
                
                # 计算标准化负载
                normalized_loads = [radar_loads[r_id] / radar_info[r_id] for r_id in radar_info]
                
                # 计算方差（越小越均衡）
                if normalized_loads:
                    variance = np.var(normalized_loads)
                    load_variance_sum += variance
            
            # 转换为均衡度得分（1-归一化方差）
            avg_variance = load_variance_sum / len(history) if history else 0
            scores[algo]['负载均衡度'] = max(0, 1 - avg_variance * 4)  # 缩放以使得分在0-1之间
        
        # 5. 计算切换次数（归一化为得分）
        max_switches = 0
        algo_switches = {}
        
        for algo, history in assignment_histories.items():
            if len(history) <= 1:
                algo_switches[algo] = 0
                continue
                
            switch_count = 0
            for t_idx in range(1, len(history)):
                prev_assignments = history[t_idx-1]['assignments']
                curr_assignments = history[t_idx]['assignments']
                
                common_targets = set(prev_assignments.keys()) & set(curr_assignments.keys())
                
                for target_id in common_targets:
                    prev_radar = prev_assignments[target_id]['radar_id'] if prev_assignments[target_id] else None
                    curr_radar = curr_assignments[target_id]['radar_id'] if curr_assignments[target_id] else None
                    
                    if prev_radar is not None and curr_radar is not None and prev_radar != curr_radar:
                        switch_count += 1
            
            algo_switches[algo] = switch_count
            max_switches = max(max_switches, switch_count)
        
        # 归一化切换次数为得分（越少越好）
        for algo in algorithms:
            scores[algo]['切换次数'] = 1 - (algo_switches[algo] / max_switches if max_switches > 0 else 0)
        
        # 计算综合得分
        overall_scores = {}
        for algo in algorithms:
            overall_scores[algo] = sum(scores[algo][metric] * weight for metric, weight in metrics.items())
        
        # 绘制综合得分柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(algorithms, [overall_scores[algo] for algo in algorithms], color='skyblue')
        
        # 在柱状图上添加具体得分
        for bar, algo in zip(bars, algorithms):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{overall_scores[algo]:.3f}', ha='center', va='bottom')
        
        # 设置标签
        ax.set_xlabel('算法')
        ax.set_ylabel('综合性能得分')
        ax.set_title('算法综合性能评分')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加各指标得分的堆叠柱状图
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        bottom = np.zeros(len(algorithms))
        for metric, weight in metrics.items():
            metric_scores = [scores[algo][metric] * weight for algo in algorithms]
            ax2.bar(algorithms, metric_scores, bottom=bottom, label=f'{metric} ({weight:.2f})')
            bottom += metric_scores
        
        ax2.set_xlabel('算法')
        ax2.set_ylabel('加权得分')
        ax2.set_title('算法各指标加权得分')
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(f"{save_path}_overall.png", dpi=300, bbox_inches='tight')
            fig2.savefig(f"{save_path}_breakdown.png", dpi=300, bbox_inches='tight')
            plt.close('all')
        else:
            plt.show()

    def plot_radar_channel_occupancy_over_time(self,
                                           assignment_histories,
                                           radar_info,
                                           time_range,
                                           save_path=None):
        """
        绘制随时间变化的雷达信道占用率曲线。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            time_range: 时间范围元组 (start, end)
            save_path: 保存路径
        """
        fig, ax = self._create_figure("雷达信道占用率随时间变化")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算每个时间步的信道占用率
        for algo_name, history in assignment_histories.items():
            timestamps = []
            occupancy_rates = []
            
            for entry in history:
                timestamp = entry['timestamp']
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                
                timestamps.append(timestamp)
                occupancy_rates.append(occupancy_rate)
            
            # 绘制曲线
            ax.plot(timestamps, occupancy_rates, label=algo_name, marker='.', alpha=0.7)
        
        ax.set_xlim(time_range)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("时间")
        ax.set_ylabel("信道占用率")
        ax.set_title("雷达信道占用率随时间变化")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_average_radar_channel_occupancy(self,
                                         assignment_histories,
                                         radar_info,
                                         save_path=None):
        """
        绘制各算法的平均雷达信道占用率条形图。
        
        Args:
            assignment_histories: 字典，键为算法名称，值为该算法的分配历史数据
            radar_info: 字典，键为雷达ID，值为该雷达的信道数量
            save_path: 保存路径
        """
        fig, ax = self._create_figure("各算法平均雷达信道占用率")
        
        # 计算网络中总的可用信道数
        total_channels = sum(radar_info.values())
        
        # 为每个算法计算平均信道占用率
        algo_names = []
        avg_occupancy_rates = []
        
        for algo_name, history in assignment_histories.items():
            occupancy_rates = []
            
            for entry in history:
                assignments = entry['assignments']
                
                # 统计当前时间步中被占用的信道数
                occupied_channels = 0
                for target_id, assignment in assignments.items():
                    if assignment['radar_id'] is not None and assignment['channel_id'] is not None:
                        occupied_channels += 1
                
                # 计算占用率
                occupancy_rate = occupied_channels / total_channels if total_channels > 0 else 0
                occupancy_rates.append(occupancy_rate)
            
            # 计算平均占用率
            avg_occupancy_rate = sum(occupancy_rates) / len(occupancy_rates) if occupancy_rates else 0
            
            algo_names.append(algo_name)
            avg_occupancy_rates.append(avg_occupancy_rate)
        
        # 绘制条形图
        bars = ax.bar(algo_names, avg_occupancy_rates, 
                     color=[self.radar_colors[i % len(self.radar_colors)] for i in range(len(algo_names))],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2%}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, max(avg_occupancy_rates) * 1.2 if avg_occupancy_rates else 1.0)
        ax.set_ylabel("平均信道占用率")
        ax.set_title("各算法平均雷达信道占用率")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    