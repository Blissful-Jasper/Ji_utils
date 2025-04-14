# ji_utils

A small Python utility to compute real-world grid spacing (dx, dy) from lat/lon arrays.

## Usage

```python
from ji_utils import compute_dx_dy

lat = np.linspace(-30, 30, 61)
lon = np.linspace(0, 359, 360)

dx, dy = compute_dx_dy(lat, lon)
