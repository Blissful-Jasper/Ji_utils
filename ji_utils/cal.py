import numpy as np
from typing import Tuple

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
