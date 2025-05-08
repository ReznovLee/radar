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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Any, Optional
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
                         allocation_data: List[Dict[str, Any]], 
                         time_range: Tuple[int, int],
                         radar_info: Dict[str, int],
                         target_info: Dict[str, Any],
                         save_path: Optional[str] = None) -> None:
        """
        绘制雷达甘特图
        
        Args:
            allocation_data: 分配结果数据，包含时间、雷达ID、通道ID、目标ID等信息
            time_range: 时间范围 (开始时间, 结束时间)
            radar_info: 雷达信息，包含雷达ID和通道数量
            target_info: 目标信息，包含目标ID和其他属性
            save_path: 保存路径，如果为None则显示图表
        """
        fig, ax = self._create_figure("雷达资源分配甘特图")
        
        # 准备Y轴标签和位置
        y_labels = []
        y_ticks = []
        current_y = 0
        radar_y_positions = {}
        
        # 为每个雷达和通道分配Y轴位置
        for radar_id, channel_count in radar_info.items():
            radar_y_positions[radar_id] = {}
            for channel_id in range(channel_count):
                y_labels.append(f"雷达{radar_id}-通道{channel_id}")
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
        target_handles = []  # 用于图例
        drawn_targets = set()
        
        for alloc in allocation_data:
            radar_id = alloc['radar_id']
            channel_id = alloc['channel_id']
            target_id = alloc['target_id']
            start_time = alloc['start_time']
            end_time = alloc['end_time']
            
            # 获取目标颜色
            target_color_idx = int(target_id) % len(self.target_colors)
            target_color = self.target_colors[target_color_idx]
            
            # 绘制分配块
            y_pos = radar_y_positions[radar_id][channel_id]
            rect = ax.barh(y_pos, 
                          width=end_time-start_time, 
                          left=start_time, 
                          height=0.8, 
                          color=target_color, 
                          alpha=0.8,
                          edgecolor='black',
                          linewidth=0.5)
            
            # 添加目标ID标签
            if (end_time - start_time) > (time_range[1] - time_range[0]) / 30:  # 只在足够宽的块上添加标签
                ax.text(start_time + (end_time - start_time) / 2, 
                       y_pos, 
                       f"T{target_id}", 
                       ha='center', 
                       va='center',
                       fontsize=8,
                       color='black')
            
            # 为图例准备句柄
            if target_id not in drawn_targets:
                target_handles.append((rect, f"目标{target_id}"))
                drawn_targets.add(target_id)
        
        # 添加图例
        legend_elements = [h[0] for h in target_handles]
        legend_labels = [h[1] for h in target_handles]
        ax.legend(legend_elements, legend_labels, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # 添加网格线
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_target_gantt(self, 
                         allocation_data: List[Dict[str, Any]], 
                         time_range: Tuple[int, int],
                         target_info: Dict[str, Any],
                         radar_info: Dict[str, int],
                         save_path: Optional[str] = None) -> None:
        """
        绘制目标甘特图
        
        Args:
            allocation_data: 分配结果数据，包含时间、雷达ID、通道ID、目标ID等信息
            time_range: 时间范围 (开始时间, 结束时间)
            target_info: 目标信息，包含目标ID和其他属性
            radar_info: 雷达信息，包含雷达ID和通道数量
            save_path: 保存路径，如果为None则显示图表
        """
        fig, ax = self._create_figure("目标跟踪分配甘特图")
        
        # 准备Y轴标签和位置
        target_ids = sorted(list(set([t['target_id'] for t in allocation_data])))
        y_labels = [f"目标{target_id}" for target_id in target_ids]
        y_ticks = list(range(len(target_ids)))
        
        # 创建目标ID到Y轴位置的映射
        target_y_positions = {target_id: i for i, target_id in enumerate(target_ids)}
        
        # 设置Y轴
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        
        # 设置X轴
        ax.set_xlim(time_range)
        ax.set_xlabel("时间")
        
        # 绘制分配结果
        radar_handles = []  # 用于图例
        drawn_radars = set()
        
        for alloc in allocation_data:
            radar_id = alloc['radar_id']
            target_id = alloc['target_id']
            start_time = alloc['start_time']
            end_time = alloc['end_time']
            
            # 获取雷达颜色
            radar_color_idx = int(radar_id) % len(self.radar_colors)
            radar_color = self.radar_colors[radar_color_idx]
            
            # 绘制分配块
            y_pos = target_y_positions[target_id]
            rect = ax.barh(y_pos, 
                          width=end_time-start_time, 
                          left=start_time, 
                          height=0.8, 
                          color=radar_color, 
                          alpha=0.8,
                          edgecolor='black',
                          linewidth=0.5)
            
            # 添加雷达ID标签
            if (end_time - start_time) > (time_range[1] - time_range[0]) / 30:  # 只在足够宽的块上添加标签
                ax.text(start_time + (end_time - start_time) / 2, 
                       y_pos, 
                       f"R{radar_id}", 
                       ha='center', 
                       va='center',
                       fontsize=8,
                       color='black')
            
            # 为图例准备句柄
            if radar_id not in drawn_radars:
                radar_handles.append((rect, f"雷达{radar_id}"))
                drawn_radars.add(radar_id)
        
        # 添加图例
        legend_elements = [h[0] for h in radar_handles]
        legend_labels = [h[1] for h in radar_handles]
        ax.legend(legend_elements, legend_labels, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # 添加网格线
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
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
            plt.ylabel('目标ID')
            plt.title('目标调度甘特图')
        elif mode == "radar":
            plt.imshow(assignment_matrix.argmax(axis=1).T, cmap='tab10', aspect='auto')
            plt.ylabel('雷达ID')
            plt.title('雷达调度甘特图')

        plt.xlabel('时间步')
        plt.colorbar(label="分配")
        
        # 保存或显示图表
        self._save_or_show(save_path)
    
    def _save_or_show(self, save_path: str) -> None:
        """保存或显示图表"""
        if save_path:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logging.info(f"已保存图表到 {save_path}")
            except Exception as e:
                logging.error(f"保存图表到 {save_path} 失败: {e}")
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
        fig, ax = self._create_figure(f"目标雷达切换频次 - {algorithm_name}")
        
        # 计算每个目标的雷达切换次数
        target_switches = {}
        
        # 按目标ID和时间排序
        sorted_data = sorted(allocation_data, key=lambda x: (x['target_id'], x['start_time']))
        
        # 计算切换次数
        for target_id in set([t['target_id'] for t in sorted_data]):
            target_allocs = [a for a in sorted_data if a['target_id'] == target_id]
            switches = 0
            prev_radar = None
            
            for alloc in target_allocs:
                current_radar = alloc['radar_id']
                if prev_radar is not None and current_radar != prev_radar:
                    switches += 1
                prev_radar = current_radar
            
            target_switches[target_id] = switches
        
        # 绘制条形图
        target_ids = sorted(target_switches.keys())
        switch_counts = [target_switches[tid] for tid in target_ids]
        
        bars = ax.bar([f"目标{tid}" for tid in target_ids], switch_counts, color='skyblue', edgecolor='navy')
        
        # 添加数值标签
        for bar, count in zip(bars, switch_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}',
                   ha='center', va='bottom')
        
        # 设置坐标轴标签
        ax.set_xlabel('目标')
        ax.set_ylabel('雷达切换次数')
        
        # 添加平均切换次数线
        avg_switches = np.mean(list(target_switches.values()))
        ax.axhline(y=avg_switches, color='red', linestyle='--', alpha=0.7)
        ax.text(len(target_ids) - 0.5, avg_switches + 0.2, f'平均: {avg_switches:.2f}', color='red')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
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
        fig, ax = self._create_figure("算法收敛曲线对比")
        
        # 绘制每个算法的收敛曲线
        for i, (algorithm, values) in enumerate(convergence_data.items()):
            iterations = list(range(1, len(values) + 1))
            ax.plot(iterations, values, marker='o', markersize=4, label=algorithm, 
                   color=plt.cm.tab10.colors[i % 10], linewidth=2)
        
        # 设置坐标轴标签
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('跟踪覆盖率 (跟踪时长/目标在辐射范围总时长)')
        
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
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_boxplot(self, 
                    performance_data: Dict[str, List[float]],
                    metric_name: str = "跟踪覆盖率",
                    save_path: Optional[str] = None) -> None:
        """
        绘制箱线图比较不同算法的性能
        
        Args:
            performance_data: 性能数据，格式为 {算法名称: [运行1的性能值, 运行2的性能值, ...]}
            metric_name: 性能指标名称
            save_path: 保存路径，如果为None则显示图表
        """
        fig, ax = self._create_figure(f"算法性能箱线图 - {metric_name}")
        
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
        ax.set_xlabel('算法')
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
        plt.title("算法综合性能对比", size=15, y=1.1)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示图表
        if save_path:
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
