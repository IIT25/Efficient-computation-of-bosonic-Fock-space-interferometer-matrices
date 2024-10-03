import perm
import numpy as np
cutoff = 5
arr = [[1.0,2.0],[3.0,4.0]]
nparr = np.array(arr, dtype=np.float64, order='C')
z = perm.calc_perm(cutoff, nparr)
print(z)