import os
import matplotlib.pyplot as plt
from typing import Optional

def save_figure(
    fig: plt.Figure,
    filename: str = 'mean',
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
        保存的文件名（不含扩展名），默认 'mean'。
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
