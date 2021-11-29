import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import gamma 


def d_alpha(x, alpha, dx, f): 
    return (1 / dx ** alpha) * gamma(1 + alpha) * (f(x + dx) - f(x))


h = 0.0001
alpha = np.arange(0.1, 1.1, 0.1)
x = np.arange(0, 1, 0.01)

fig, ax = plt.subplots()
for a in alpha:
    ax.plot(x, d_alpha(x, a, h, lambda _x: _x ** 2), label=str(np.round(a, 4)))

plt.grid()
plt.legend()
plt.show()