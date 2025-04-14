import numpy as np
import xarray as xr
from typing import Tuple
from metpy.units import units

# =================== 1. 数据加载 =======================
def load_data(path: str, var: str, lat_range=(-15, 15)) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """加载并预处理数据"""
    ds = xr.open_dataset(path).sortby('lat').sel(lat=slice(*lat_range))
    return ds[var], ds.lon.values, ds.lat.values

# =================== 2. DSE/MSE计算 =======================
def calc_dse(T: xr.DataArray, z: xr.DataArray, plev: np.ndarray) -> xr.DataArray:
    """计算干静能 DSE = Cp*T + g*z"""
    Cpd = 1004  # J/K/kg
    g = 9.8     # m/s²
    return Cpd * T + g * z  # 单位：J/kg

def calc_mse(T: xr.DataArray, z: xr.DataArray, qv: xr.DataArray, plev: np.ndarray, saturation=False) -> xr.DataArray:
    """计算湿静能 MSE"""
    Cpd = 1004
    g = 9.8
    Lv = 2.25e6
    epsilon = 0.622

    if saturation:
        e = 6.1094 * np.exp(17.625*(T-273.15)/(T-273.15+243.04))  # 饱和水汽压 (hPa)
        qvs = epsilon * e / (plev / 100)  # 转换为 hPa
        return Cpd*T + g*z + Lv*qvs
    else:
        return Cpd*T + g*z + Lv*qv

# =================== 3. 经纬度转换为 dx, dy =======================
def compute_dx_dy(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = 6371e3
    pi = np.pi
    xlon, ylat = np.meshgrid(lon, lat)
    dlonx = np.gradient(xlon, axis=1)
    dlaty = np.gradient(ylat, axis=0)
    dx = R * np.cos(ylat * pi / 180) * dlonx * pi / 180
    dy = R * dlaty * pi / 180
    return dx, dy

# =================== 4. 计算Q项和导数 =======================
def compute_energy_budget(ta_path, zg_path, ua_path, va_path, wa_path) -> xr.Dataset:
    # 读取数据
    ta, lon, lat = load_data(ta_path, 'ta')
    zg, _, _     = load_data(zg_path, 'zg')
    ua, _, _     = load_data(ua_path, 'ua')
    va, _, _     = load_data(va_path, 'va')
    wa, _, _     = load_data(wa_path, 'wap')

    plev = ta.plev.values * units.Pa

    # 计算 DSE
    dse = calc_dse(ta, zg, plev)

    # dx, dy
    dx, dy = compute_dx_dy(lat, lon)

    # 时间导数 ∂s/∂t
    ds_dt = np.gradient(dse, axis=0) / 86400  # s⁻¹

    # 空间导数 ∂s/∂x 和 ∂s/∂y
    ds_dx = np.gradient(dse, axis=-1) / dx[np.newaxis, np.newaxis, :, :]
    ds_dy = np.gradient(dse, axis=-2) / dy[np.newaxis, np.newaxis, :, :]

    # 垂直导数 ∂s/∂p
    ds_dp = np.gradient(dse, plev, axis=1)

    # Q项计算
    Q = ds_dt + ua * ds_dx + va * ds_dy + wa * ds_dp

    # 输出所有变量为 xarray.Dataset
    result = xr.Dataset({
        'DSE': dse.data,
        'ds_dt': (dse.dims, ds_dt.data),
        'ds_dx': (dse.dims, ds_dx.data),
        'ds_dy': (dse.dims, ds_dy.data),
        'ds_dp': (dse.dims, ds_dp.data),
        'Q': (dse.dims, Q.data),
    }, coords=dse.coords)

    return result
