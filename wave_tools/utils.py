
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import colors
from typing import Optional, List, Tuple


def filter_series(series, min_wn, max_wn):
    return series[(series.index >= min_wn) & (series.index <= max_wn)]


def save_figure(
    fig: plt.Figure,
    filename: str = 'meridional_mean',
    folder: Optional[str] = None,
    fmt: str = 'pdf',
    dpi: int = 600
) -> None:
    """
    保存 matplotlib 生成的图像文件。

    参数：
    --------
    fig : matplotlib.figure.Figure
        要保存的图像对象。
    filename : str, optional
        保存的文件名（不含扩展名），默认 'meridional_mean'。
    folder : str, optional
        保存文件的文件夹路径，默认当前工作目录。
    fmt : str, optional
        文件格式，例如 'pdf'、'png'、'jpg'，默认 'pdf'。
    dpi : int, optional
        保存图像的分辨率，默认 600 dpi。

    返回：
    --------
    None
    """

    # 确定保存路径
    if folder is None:
        folder = os.getcwd()

    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f'Folder {folder} has been created.')
    else:
        print(f'Folder {folder} already exists.')

    # 完整的输出路径，自动加上后缀
    outpath = os.path.join(folder, f"{filename}.{fmt}")

    # 保存图像
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight', format=fmt)
    print(f'Figure saved at: {outpath}')

def get_curve(
    he: Optional[List[float]] = None,
    fmax: Optional[List[float]] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    计算 Kelvin 波（CCKW）能量带的包络曲线坐标。
    """
    if he is None:
        he = [8, 25, 90]
    if fmax is None:
        fmax = [1/3, 1/2.25, 0.5]

    g = 9.8
    re = 6371e3
    s2d = 86400

    kw_x = []
    kw_y = []

    for v in range(len(he)):
        s_min = (g * he[0]) ** 0.5 / (2 * np.pi * re) * s2d
        s_max = (g * he[-1]) ** 0.5 / (2 * np.pi * re) * s2d
        kw_tmax = 20

        kw_x.append(np.array([
            2,
            1 / kw_tmax / s_min,
            14,
            14,
            fmax[0] / s_max,
            2,
            2
        ]))

        kw_y.append(np.array([
            1 / kw_tmax,
            1 / kw_tmax,
            14 * s_min,
            fmax[0],
            fmax[0],
            2 * s_max,
            1 / 20
        ]))

    return kw_x, kw_y


def create_cmap_from_string(color_string: str) -> colors.ListedColormap:
    """
    根据给定的颜色字符串创建一个反转的颜色映射（Colormap）。

    输入:
        color_string (str): 一个字符串，每行包含一个颜色，可以是标准颜色名、RGB 值（如 '1,0,0' 表示红色），或十六进制颜色代码（如 '#FF5733'）。

    输出:
        ListedColormap: 返回一个基于输入颜色字符串的反转颜色映射对象（Colormap）。

    示例:
        color_string =('
        #FF5733
        #33FF57
        #3357FF
       ')
        
        cmap = create_cmap_from_string(color_string)
        plt.imshow(data, cmap=cmap)
        plt.colorbar()
        plt.show()
    """
    # 去除多余的空白行并分割每行
    color_list = color_string.strip().split('\n')
    
    # 创建并返回反转后的颜色映射对象
    return colors.ListedColormap(color_list[::-1])