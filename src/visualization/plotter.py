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
    
    def _save_or_show(self, save_path: str) -> None:
        """保存或显示图表"""
        if save_path:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logging.info(f"Result have been saved to {save_path}")
            except Exception as e:
                logging.error(f"Result saved to {save_path} falied: {e}")
            finally:
                plt.close()
        else:
            plt.show()
    
    def plot_target_switching(self, 
                             allocation_data: List[Dict[str, Any]],
                             target_info: Dict[str, Any],
                             algorithm_name: str = "BFSA-Rho",
                             save_path: Optional[str] = None) -> None:
        """
        绘制目标切换频次图
        
        Args:
            allocation_data: 分配结果数据
            target_info: 目标信息
            algorithm_name: 算法名称
            save_path: 保存路径，如果为None则显示图表
        """
        fig, ax = self._create_figure(f"Target radar switching frequency - {algorithm_name}")
        
        # 计算每个目标的雷达切换次数
        target_switches = {}
        
        # 从分配历史中提取目标切换数据
        target_radar_history = {}
        
        # 遍历时间步骤
        for step_data in allocation_data:
            # 确保 step_data 是字典类型
            if isinstance(step_data, dict) and 'timestamp' in step_data and 'assignments' in step_data:
                timestamp = step_data['timestamp']
                assignments = step_data['assignments']
                
                # 更新每个目标的雷达分配历史
                for target_id, assignment in assignments.items():
                    if target_id not in target_radar_history:
                        target_radar_history[target_id] = []
                    
                    if assignment is not None and assignment['radar_id'] is not None:
                        target_radar_history[target_id].append({
                            'timestamp': timestamp,
                            'radar_id': assignment['radar_id']
                        })
        
        # 计算每个目标的雷达切换次数
        for target_id, history in target_radar_history.items():
            # 按时间戳排序
            sorted_history = sorted(history, key=lambda x: x['timestamp'])
            switches = 0
            prev_radar = None
            
            for entry in sorted_history:
                current_radar = entry['radar_id']
                if prev_radar is not None and current_radar != prev_radar:
                    switches += 1
                prev_radar = current_radar
            
            target_switches[target_id] = switches
        
        # 绘制条形图
        target_ids = sorted(target_switches.keys())
        switch_counts = [target_switches[tid] for tid in target_ids]
        
        bars = ax.bar([f"Target {tid}" for tid in target_ids], switch_counts, color='skyblue', edgecolor='navy')
        
        # 添加数值标签
        for bar, count in zip(bars, switch_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}',
                   ha='center', va='bottom')
        
        # 设置坐标轴标签
        ax.set_xlabel('Targets')
        ax.set_ylabel('Radar switch counts')
        
        # 添加平均切换次数线
        if target_switches:  # 确保有数据再计算平均值
            avg_switches = np.mean(list(target_switches.values()))
            ax.axhline(y=avg_switches, color='red', linestyle='--', alpha=0.7)
            ax.text(len(target_ids) - 0.5, avg_switches + 0.2, f'avarage: {avg_switches:.2f}', color='red')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_convergence_curve(self, 
                              convergence_data: Dict[str, List[float]],
                              save_path: Optional[str] = None) -> None:
        """
        绘制算法收敛曲线
        
        Args:
            convergence_data: 收敛数据，格式为 {算法名称: [迭代1的目标函数值, 迭代2的目标函数值, ...]}
            save_path: 保存路径，如果为None则显示图表
        """
        fig, ax = self._create_figure("Comparison of algorithm convergence curves")
        
        # 绘制每个算法的收敛曲线
        for i, (algorithm, values) in enumerate(convergence_data.items()):
            iterations = list(range(1, len(values) + 1))
            ax.plot(iterations, values, marker='o', markersize=4, label=algorithm, 
                   color=plt.cm.tab10.colors[i % 10], linewidth=2)
        
        # 设置坐标轴标签
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Tracking coverage (tracking time/total time the target is in the radiation range)')
        
        # 设置Y轴范围，从0到1或略高于最大值
        max_value = max([max(values) for values in convergence_data.values()])
        ax.set_ylim(0, max(1.0, max_value * 1.05))
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例
        ax.legend(loc='lower right')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
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
    
    def plot_algorithm_comparison(self, 
                                 algorithms: List[str],
                                 metrics: Dict[str, Dict[str, float]],
                                 save_path: Optional[str] = None) -> None:
        """
        绘制算法综合性能对比图
        
        Args:
            algorithms: 算法名称列表
            metrics: 性能指标，格式为 {指标名称: {算法名称: 性能值}}
            save_path: 保存路径，如果为None则显示图表
        """
        # 设置雷达图的角度
        num_metrics = len(metrics)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi, subplot_kw=dict(polar=True))
        
        # 准备标签
        labels = list(metrics.keys())
        labels += labels[:1]  # 闭合标签
        
        # 绘制每个算法的雷达图
        for i, algorithm in enumerate(algorithms):
            values = [metrics[metric][algorithm] for metric in metrics.keys()]
            values += values[:1]  # 闭合数值
            
            ax.plot(angles, values, linewidth=2, label=algorithm, color=plt.cm.tab10.colors[i % 10])
            ax.fill(angles, values, alpha=0.1, color=plt.cm.tab10.colors[i % 10])
        
        # 设置标签
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 设置标题
        plt.title("Comparison of comprehensive performance of algorithms", size=15, y=1.1)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
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

    def plot_radar_utilization_heatmap(self, 
                                  assignment_history, 
                                  radar_info,
                                  time_range,
                                  algorithm_name="",
                                  save_path=None):
        """
        绘制雷达利用率热力图
        
        Args:
            assignment_history: 分配历史数据
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
            # 此时需要从assignment_history中提取雷达ID
            radar_ids = []
            for step in assignment_history:
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
        for step in assignment_history:
            t = int(step['timestamp'] - time_range[0])
            if t < 0 or t >= time_steps:
                continue
                
            # 统计每个雷达分配的通道数
            radar_channel_count = {rid: 0 for rid in radar_ids}
            radar_total_channels = {rid: 0 for rid in radar_ids}
            
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
                    total_channels = 4
                    
                if radar_channel_count[rid] > 0:
                    utilization_matrix[i, t] = radar_channel_count[rid] / total_channels
        
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

    def plot_priority_satisfaction(self,
                                 allocation_data,
                                 time_range,
                                 target_info,
                                 save_path=None):
        """
        绘制优先级满足度图表
        
        Args:
            allocation_data: 分配历史数据
            time_range: 时间范围
            target_info: 目标信息
            save_path: 保存路径
        """
        fig, ax = self._create_figure("目标优先级满足度")
        
        # 按优先级对目标分组
        priority_groups = {}
        for target_id, info in target_info.items():
            priority = info.get('priority', 1)  # 默认优先级为1
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(target_id)
            
        # 计算每个时间步的优先级满足度
        timestamps = []
        satisfaction_rates = []
        
        for alloc in allocation_data:
            timestamp = alloc['timestamp']
            assignments = alloc['assignments']
            
            # 统计每个优先级的分配情况
            total_by_priority = {p: len(targets) for p, targets in priority_groups.items()}
            assigned_by_priority = {p: 0 for p in priority_groups.keys()}
            
            for target_id, assignment in assignments.items():
                if assignment is not None and assignment['radar_id'] is not None:
                    target_priority = target_info[target_id].get('priority', 1)
                    assigned_by_priority[target_priority] += 1
            
            # 计算总体满足率
            total_weighted = sum(p * total for p, total in total_by_priority.items())
            assigned_weighted = sum(p * assigned for p, assigned in assigned_by_priority.items())
            
            satisfaction_rate = assigned_weighted / total_weighted if total_weighted > 0 else 0
            
            timestamps.append(timestamp)
            satisfaction_rates.append(satisfaction_rate)
        
        # 绘制满足度曲线
        ax.plot(timestamps, satisfaction_rates, '-o', linewidth=2, markersize=4)
        ax.set_xlim(time_range)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("时间")
        ax.set_ylabel("优先级满足度")
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

        def plot_spatiotemporal_assignment(assignment_history, radars_dict, targets_by_timestep, 
                                    time_range, save_path=None, interval=200):
            """创建时空分布动态可视化"""
            from matplotlib.animation import FuncAnimation
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制雷达位置和覆盖范围（静态部分）
            for radar_id, info in radars_dict.items():
                ax.scatter(info['x'], info['y'], info['z'], c='r', marker='^', s=100, label=f'雷达 {radar_id}')
                
                # 绘制雷达覆盖范围（简化为半透明球体）
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = info['radius'] * np.cos(u) * np.sin(v) + info['x']
                y = info['radius'] * np.sin(u) * np.sin(v) + info['y']
                z = info['radius'] * np.cos(v) + info['z']
                ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)
            
            # 初始化目标散点和连线
            target_scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=50, alpha=0.8)
            assignment_lines = []
            
            # 设置坐标轴
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('目标-雷达分配的时空分布')
            
            # 添加时间标签
            time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes)
            
            # 初始化函数
            def init():
                target_scatter._offsets3d = ([], [], [])
                for line in assignment_lines:
                    if line:
                        line.set_data([], [])
                        line.set_3d_properties([])
                time_text.set_text('')
                return [target_scatter, time_text] + assignment_lines
            
            # 更新函数
            def update(frame):
                # 清除之前的连线
                for line in assignment_lines:
                    if line:
                        line.remove()
                assignment_lines.clear()
                
                if frame >= len(assignment_history):
                    return [target_scatter, time_text]
                
                timestep = assignment_history[frame]['timestamp']
                assignments = assignment_history[frame]['assignments']
                
                if timestep not in targets_by_timestep:
                    return [target_scatter, time_text]
                    
                targets = targets_by_timestep[timestep]
                
                # 更新目标位置
                x, y, z = [], [], []
                colors = []  # 用颜色表示优先级
                
                for target in targets:
                    target_id = str(target['id'])
                    x.append(target['position'][0])
                    y.append(target['position'][1])
                    z.append(target['position'][2])
                    colors.append(target['priority'])  # 颜色基于优先级
                    
                    # 如果目标被分配了雷达，绘制连线
                    if target_id in assignments and assignments[target_id] is not None:
                        radar_id = assignments[target_id]['radar_id']
                        if radar_id in radars_dict:
                            radar = radars_dict[radar_id]
                            line = ax.plot([target['position'][0], radar['x']], 
                                        [target['position'][1], radar['y']], 
                                        [target['position'][2], radar['z']], 
                                        'k-', alpha=0.3)[0]
                            assignment_lines.append(line)
                
                # 更新散点图
                target_scatter._offsets3d = (x, y, z)
                target_scatter.set_array(np.array(colors))
                
                # 更新时间标签
                time_text.set_text(f'时间步: {timestep}')
                
                return [target_scatter, time_text] + assignment_lines
            
            # 创建动画
            ani = FuncAnimation(fig, update, frames=range(time_range[0], time_range[1]),
                            init_func=init, blit=False, interval=interval)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                ani.save(save_path, writer='ffmpeg', dpi=200)
                plt.close()
            else:
                plt.show()
            
            return ani

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

    