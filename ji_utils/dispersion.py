# matsuno/dispersion.py

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from .constants import beta_parameters, g

def wn_array(max_wn: int = 50, n_wn: int = 500):
    return np.linspace(-max_wn, max_wn, n_wn)

def wn2k(wn, perimeter):
    return 2 * np.pi * wn / perimeter

def afreq2freq(angular_frequency):
    period = 2 * np.pi / angular_frequency
    frequency = 1 / period
    return period, frequency

# === 色散关系定义 === #

def kelvin_dispersion(k, he, beta):
    return np.where(k <= 0, np.nan, np.sqrt(g * he) * k)

def mrg_dispersion(k, he, beta):
    omega = np.sqrt(g * he) * k
    return -omega * omega / beta

def eig_dispersion(k, he, beta, n):
    return np.sqrt(g * he) * np.sqrt(k**2 + (2*n + 1) * beta / (np.sqrt(g * he)))

def er_dispersion(k, he, beta, n):
    def dispersion(w, k_val, n_val, he_val, beta_val):
        return w**2 - beta_val * w / (w**2 - g * he_val * k_val**2) - (2 * n_val + 1) * beta_val

    omega = []
    for k_val in k:
        try:
            sol = fsolve(dispersion, 0.0001, args=(k_val, n, he, beta))
            omega.append(sol[0])
        except:
            omega.append(np.nan)
    return np.array(omega)

# === 主调用函数 === #

def compute_dispersion_curve(mode: str, he: float, latitude: float = 0,
                              max_wn: int = 50, n_wn: int = 500, n: int = 0) -> pd.DataFrame:
    wn = wn_array(max_wn, n_wn)
    beta, perimeter = beta_parameters(latitude)
    k = wn2k(wn, perimeter)

    mode = mode.lower()
    if mode == 'kelvin':
        omega = kelvin_dispersion(k, he, beta)
        label = f'Kelvin(he={he}m)'
    elif mode == 'mrg':
        omega = mrg_dispersion(k, he, beta)
        label = f'MRG(he={he}m)'
    elif mode == 'eig':
        omega = eig_dispersion(k, he, beta, n)
        label = f'EIG(n={n}, he={he}m)'
    elif mode == 'er':
        omega = er_dispersion(k, he, beta, n)
        label = f'ER(n={n}, he={he}m)'
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    _, frequency = afreq2freq(omega)
    df = pd.DataFrame({label: frequency}, index=wn)
    df.index.name = 'Wavenumber'
    return df
