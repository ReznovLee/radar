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

    def plot_target_gantt(self, 
                     allocation_data, 
                     time_range,
                     target_info,
                     radar_info,
                     save_path=None):
        fig, ax = self._create_figure("Goal Tracking Assignment Gantt Chart")

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

    def plot_target_assignment_rate_over_time(self,
                                              assignment_rate_data: Dict[str, List[float]],
                                              save_path: Optional[str] = None) -> None:
        """
        绘制不同算法的目标分配率随时间变化的曲线 (仅使用线条)
        
        Args:
            assignment_rate_data: 分配率数据，格式为 {算法名称: [每时间步的分配率值, ...]}
                                    这些值应根据 (已跟踪目标数 / 覆盖范围内目标数) * (优先级因子) 计算得出。
            save_path: 保存路径，如果为None则显示图表
        """
        # Construct title dynamically
        algo_names = list(assignment_rate_data.keys())
        if len(algo_names) > 0:
            title_str = "Target Tracking Rate Comparison: " + " vs ".join(algo_names) # Updated title
        else:
            title_str = "Target Tracking Rate Comparison" # Updated title
        fig, ax = self._create_figure(title_str) # _create_figure uses fig.suptitle
        
        if not assignment_rate_data or all(not v for v in assignment_rate_data.values()):
            logging.warning("No assignment rate data provided or all data lists are empty. Skipping plot.")
            ax.text(0.5, 0.5, "No data available for tracking rate plot",
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
                logging.info(f"Skipping tracking rate plot for {algorithm} as its data list is empty.")
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
            logging.warning("No valid algorithm data was plotted for tracking rate.")
            ax.text(0.5, 0.5, "No valid data to plot after filtering",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            self._save_or_show(save_path, fig)
            return

        # 设置坐标轴标签
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Target Tracking Rate') # Updated Y-axis label
        
        # 设置Y轴范围，固定为 0 到 1.1 (如果您的新速率定义可能超出此范围，请酌情调整)
        ax.set_ylim(0, 1.1) 
        
        # 添加网格和图例
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best') 
        
        # 调整布局，在图表顶部留出4%的边距以容纳标题
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        self._save_or_show(save_path, fig)

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
        
        title = f"Radar Utilization Heat Map - {algorithm_name}" if algorithm_name else "雷达利用率热力图"
        ax.set_title(title)
        
        im = ax.imshow(utilization_matrix, cmap='hot', aspect='auto', vmin=0, vmax=1)
        
        # 设置坐标轴
        ax.set_xlabel('Time')
        ax.set_ylabel('Radar ID')
        ax.set_yticks(np.arange(len(radar_ids)))
        ax.set_yticklabels([f'R{rid}' for rid in radar_ids])
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Utilization (assignment channels/total channels)')
        
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
        fig, ax = self._create_figure("Comparison of target priority satisfaction")
        
        # 计算每个算法的优先级满足度
        algorithm_satisfaction = {}
        
        for algorithm_name, history in allocation_histories.items():
            # 确保history是列表类型
            if not isinstance(history, list):
                logging.error(f"Assignment history data for algorithm {algorithm_name} is not a list type")
                continue
                
            # 初始化优先级统计
            priority_counts = {}  # {优先级: [分配次数, 总次数]}
            
            # 遍历每个时间步
            for alloc in history:
                # 确保alloc是字典类型
                if not isinstance(alloc, dict):
                    logging.error(f"The assignment history of algorithm {algorithm_name} contains non-dictionary data")
                    continue
                    
                # 确保alloc包含必要的键
                if 'timestamp' not in alloc or 'assignments' not in alloc:
                    logging.error(f"The assignment history data for algorithm {algorithm_name} is malformed, missing a required key")
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
        ax.set_xlabel('Target Priority')
        ax.set_ylabel('Satisfaction rate (number of assignments/total number of assignments)')
        ax.set_title('Comparison of priority satisfaction of different algorithms')
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
        fig, ax = self._create_figure("Radar channel occupancy rate changes over time")
        
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
        ax.set_xlabel("Time")
        ax.set_ylabel("Channel occupancy")
        ax.set_title("Radar channel occupancy rate changes over time")
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
        fig, ax = self._create_figure("Average radar channel occupancy of each algorithm")
        
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
        ax.set_ylabel("Average channel occupancy")
        ax.set_title("Average radar channel occupancy of each algorithm")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    