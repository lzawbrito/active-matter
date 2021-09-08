from autograd import numpy as np
from autograd import grad

def rectangular_bdy(x):
    if x[0] <= x[1] and -x[0] <= x[1]:
        return x[1]
    elif x[0] >= x[1] and -x[0] <= x[1]:
        return x[0]
    elif x[0] >= x[1] and -x[0] >= x[1]:
        return -x[1] 
    elif x[0] <= x[1] and -x[0] >= x[1]: 
        return -x[0]
    else:
        return x[0]

def circular_bdy(x): 
    return np.sqrt(x[0] ** 2 + x[1] ** 2)

if __name__ == '__main__': 
    print(grad(rectangular_bdy)((1., -2.)))