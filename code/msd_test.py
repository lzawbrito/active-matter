from src.mathutils import msd 
from math import isclose

t = [0, 1, 2, 3, 4, 5]
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 2, 3, 4, 5]
expected = [0, 2, 8, 18, 32, 50]

msds = []
for delta_t in t: 
    msds.append(msd(x, y, t[1] - t[0], delta_t=delta_t))

for i, j in zip(msds, expected):
    assert isclose(i, j, abs_tol=1e-10)

