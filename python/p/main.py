import perm
import numpy as np
import ctypes 
cutoff = 3
arr = {1,2,3,4}
#initMatrix = np.array([[1, 2], [3, 4]])
subspace_rep = [None] *3
subspace_rep_pointer = []
for i in range(0, cutoff):
    subspace_rep[i] = np.zeros((i,i))
    subspace_rep_pointer.append(subspace_rep[i].ctypes.data)
print(subspace_rep)
print(subspace_rep_pointer)
perm.MatrixPointer()
print(np.array(perm.MatrixAdd(4,5), copy = False))
print(subspace_rep)