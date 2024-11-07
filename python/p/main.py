import perm
import cmath
import numpy as np
cutoff = 5
arr = [[complex(1.5, 1),complex(9, 0)],[complex(3,4), complex(5,6)]]
nparr = np.array(arr, dtype=np.cdouble, order='C')
z = perm.calc_perm(cutoff, nparr)
print(z)