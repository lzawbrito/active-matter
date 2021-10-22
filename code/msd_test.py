from src.mathutils import msd, msds_for_lag
import numpy as np
from math import isclose
import matplotlib.pyplot as plt
import time 

t = np.linspace(0, 100, 1000) #[0, 1, 2, 3, 4, 5]
x = np.linspace(0, 100, 1000) #[0, 1, 2, 3, 4, 5]
y = np.linspace(0, 100, 1000) #[0, 1, 2, 3, 4, 5]
expected = [0, 2, 8, 18, 32, 50]


print("Running non-concurrent version")
# start_time = time.time()
# msds = []
# for delta_t in t: 
#     msds.append(msd(x, y, t[1] - t[0], delta_t=delta_t))


# for i, j in zip(msds, expected):
#     try:
#         assert isclose(i, j, abs_tol=1e-10)
#     except AssertionError: 
#         print("Given: " + str(i) + ", expected: " + str(j))
# print("Elapsed time:" + str(time.time() - start_time))

print("Running concurrent version")
start_time = time.time()
msds = []
msds2 = msds_for_lag(x, y, t[1] - t[0], t)
for i, j in zip(msds2, expected):
    try:
        assert isclose(i, j, abs_tol=1e-10)
    except AssertionError: 
        print("Given: " + str(i) + ", expected: " + str(j))
print("Elapsed time:" + str(time.time() - start_time))

fig, ax = plt.subplots()

plt.loglog(t, msds2)
plt.loglog(t, np.array(t)** 2)
plt.show()
