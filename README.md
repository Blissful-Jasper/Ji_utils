# ji_utils

A small Python utility to compute real-world grid spacing (dx, dy) from lat/lon arrays.

## Usage

```python
from ji_utils import compute_dx_dy

lat = np.linspace(-30, 30, 61)
lon = np.linspace(0, 359, 360)

dx, dy = compute_dx_dy(lat, long)
```

## 绘制kelvin波段频散曲线

```python
from cckw_tools import plot_cckw_envelope, get_curve

# 调用画图
plot_cckw_envelope(he=[8, 12, 25, 50, 90], fmax=[1/3,  0.5])
```
![image](https://github.com/user-attachments/assets/b20e291b-acd8-4dc0-8b37-af852c952fa3)
