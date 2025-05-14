# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

@email : xianpuji@hhu.edu.cn
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union
from scipy.stats import linregress
from wave_tools import save_figure

def plot_multiple_wave_trends(
    wave_filters: List[xr.DataArray],
    wave_names: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    ylims: Optional[List[Optional[Tuple[float, float]]]] = None,
    save_path: str = 'Fig_wave_trend_1979-2020.png'
) -> None:
    """
    绘制多个热带波动在不同区域的年际变化趋势，并保存图像。

    Parameters
    ----------
    wave_filters : list of xarray.DataArray
        每个波动的滤波后 OLR 数据。
    wave_names : list of str, optional
        每个波动的名称，用作图表标题。
    regions : list of str, optional
        要绘制的区域名称，默认为全球和几个关键区域。
    ylims : list of tuple or None, optional
        每个波动图的 y 轴范围。例如 [(7, 10), (5, 9), None]，None 表示自动设置。
    save_path : str
        图像保存的文件路径，默认保存为当前目录下的 'Fig_wave_trend_1979-2020.png'。
    """
    if wave_names is None:
        wave_names = ['Kelvin', 'ER', 'MRG', 'TD']
    if regions is None:
        regions = ['Global', 'Indian-Pacific', 'Indian', 'EastPacific', 'Africa']
    if ylims is None:
        ylims = [None] * len(wave_filters)

    region_bounds = {
        'Global': (0, 360),
        'Indian-Pacific': (130, 250),
        'Indian': (55, 130),
        'EastPacific': (250, 345),
        'Africa': ((345, 360), (0, 55))  # 两段拼接
    }

    plt.rcParams.update({
        'axes.linewidth': 1,
        'font.family': 'Arial',
        'font.size': 12
    })

    n = len(wave_filters)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows + 1), dpi=300)
    axes = axes.flatten()

    for i, (ds_filter, wave_name) in enumerate(zip(wave_filters, wave_names)):
        ax = axes[i]
        color_cycle = plt.cm.Pastel1.colors
        region_colors = dict(zip(regions, color_cycle))
        all_values = []

        for region in regions:
            if region == 'Africa':
                ds_region = xr.concat([
                    ds_filter.sel(lon=slice(345, 360)),
                    ds_filter.sel(lon=slice(0, 55))
                ], dim='lon')
            else:
                lon_bounds = region_bounds[region]
                ds_region = ds_filter.sel(lon=slice(*lon_bounds))

            ds_region = ds_region.sel(time=slice('1979', '2020'), lat=slice(-15, 15))
            ds_inter_annual = ds_region.groupby('time.year').std('time')
            ds_trend = ds_inter_annual.mean(['lat', 'lon'])

            years = ds_trend['year'].values
            values = ds_trend.values
            all_values.append(values)

            slope, intercept, r, p, stderr = linregress(years, values)
            fit_line = slope * years + intercept

            sig = '**' if p < 0.01 else '*' if p < 0.05 else '^' if p < 0.1 else ''
            color = region_colors[region]
            ax.plot(years, values, color=color, label=f'{region}: {slope:.4f}/yr (p={p:.4f}) {sig}')
            ax.plot(years, fit_line, '--', color=color, linewidth=1)

        ax.set_title(f'{wave_name} Interannual Trend')
        ax.set_xticks(np.arange(1980, 2025, 5))
        ax.grid(True, linestyle='--', linewidth=0.5)

        if i % ncols == 0:
            ax.set_ylabel('Interannual Std Dev')
        if i >= nrows * ncols - ncols:
            ax.set_xlabel('Year')

        if ylims[i] is not None:
            ax.set_ylim(ylims[i])
        else:
            all_flat = np.concatenate(all_values)
            buffer = 0.05 * (all_flat.max() - all_flat.min())
            ax.set_ylim(all_flat.min() - buffer, all_flat.max() + buffer)

        ax.legend(fontsize=10, loc='best', frameon=False)

    # 删除多余子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    save_figure(fig, save_path)
    plt.show()

