# matsuno/constants.py

import numpy as np

# 地球常数
a = 6.371e6               # 地球半径 (m)
omega = 7.292e-5          # 地球自转角速度 (rad/s)
g = 9.81                  # 重力加速度 (m/s²)

def beta_parameters(latitude: float) -> tuple[float, float]:
    """计算 beta 值和赤道周长"""
    beta = 2 * omega * np.cos(np.radians(latitude)) / a
    perimeter = 2 * np.pi * a * np.cos(np.radians(latitude))
    return beta, perimeter
