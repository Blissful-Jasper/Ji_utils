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
    """
    R = 6371e3
    pi = np.pi

    xlon, ylat = np.meshgrid(lon, lat)
    dlony, dlonx = np.gradient(xlon)
    dlaty, dlatx = np.gradient(ylat)

    dx = R * np.cos(ylat * pi / 180) * dlonx * pi / 180
    dy = R * dlaty * pi / 180

    return dx, dy
