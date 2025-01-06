import perm
import cmath
import numpy as np
cutoff = 2
arr = [[complex(1.5, 1),complex(9, 0)],[complex(3,4), complex(5,6)]]
helper = (np.array([[0,1],[1,0]]),np.array([[0,0,1]]), np.array([[0,1,1]]),np.array([[0,0], [0,1]]),np.array([[1,1]]))
nparr = np.array(arr, dtype=np.cdouble, order='C')
z = perm.calc_perm(6,nparr, helper)
print(z)