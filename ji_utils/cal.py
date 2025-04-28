import numpy as np
from typing import Tuple, List, Optional

def compute_dx_dy(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据经纬度计算实际空间距离 dx（东西向）和 dy（南北向），单位为米。

    参数：
    --------
    lat : np.ndarray
        一维纬度数组（单位：度），shape = (n_lat,)
    lon : np.ndarray
        一维经度数组（单位：度），shape = (n_lon,)

    返回：
    --------
    dx : np.ndarray
        二维数组，表示每个网格点的东西向网格间距（单位：米），shape = (n_lat, n_lon)
    dy : np.ndarray
        二维数组，表示每个网格点的南北向网格间距（单位：米），shape = (n_lat, n_lon)

    # second method：
    import numpy as np
    import xarray as xr

    # 假设 lat 和 lon 是 xarray.DataArray 的纬度、经度坐标
    R = 6371e3  # 地球半径（米）
    # 计算经度方向的实际距离增量（单位：米）
    dlon = np.deg2rad(np.gradient(lon))  # 经度间隔（弧度）
    dx = R * np.cos(np.deg2rad(lat)) * dlon[:, None, None]  # 广播到数据维度
    # 计算纬度方向的实际距离增量（单位：米）
    dlat = np.deg2rad(np.gradient(lat))  # 纬度间隔（弧度）
    dy = R * dlat[:, None]  # 广播到数据维度
    # 计算经度方向偏导（x 方向）
    ds_dx = xr_data.differentiate('lon') / dx  # 或用 np.gradient(s, dx, axis=3)
    # 计算纬度方向偏导（y 方向）
    ds_dy = xr_data.differentiate('lat') / dy  # 或用 np.gradient(s, dy, axis=2)
    
    """
    R = 6371e3
    pi = np.pi

    xlon, ylat = np.meshgrid(lon, lat)
    dlony, dlonx = np.gradient(xlon)
    dlaty, dlatx = np.gradient(ylat)

    dx = R * np.cos(ylat * pi / 180) * dlonx * pi / 180
    dy = R * dlaty * pi / 180

    return dx, dy

def get_curve(
    he: Optional[List[float]] = None,
    fmax: Optional[List[float]] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    计算 Kelvin 波（CCKW）能量带的包络曲线坐标。

    参数：
    --------
    he : List[float], optional
        浅水深度列表（单位：米），默认 [8, 25, 90]。
    fmax : List[float], optional
        最大频率列表（单位：cycles/day），默认 [1/3, 1/2.25, 0.5]。

    返回：
    --------
    kw_x : List[np.ndarray]
        列表，包含每种浅水深度下 CCKW 区域边界的 x 坐标数组。
    kw_y : List[np.ndarray]
        列表，包含每种浅水深度下 CCKW 区域边界的 y 坐标数组。

    """

    # 默认值设置
    if he is None:
        he = [8, 25, 90]
    if fmax is None:
        fmax = [1/3, 1/2.25, 0.5]

    # 常数定义
    g = 9.8          # 重力加速度（m/s²）
    re = 6371e3      # 地球半径（m）
    s2d = 86400      # 秒到天的换算因子

    # 准备 CCKW 区域的边界曲线
    kw_x = []
    kw_y = []

    for v in range(len(he)):  # 按照 he/fmax 的长度循环
        s_min = (g * he[0]) ** 0.5 / (2 * np.pi * re) * s2d  # 最小斜率 (对应 he 最小值)
        s_max = (g * he[-1]) ** 0.5 / (2 * np.pi * re) * s2d # 最大斜率 (对应 he 最大值)

        kw_tmax = 20  # 最大周期（天）

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

