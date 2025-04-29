import numpy as np
import perm
from validators import validate_backward
import numpy as np
    

def test_backward():
    def interferometer():
        return np.array([[ 0.70711+0.j     , -0.35355+0.61237j], 
       [ 0.35355+0.61237j,  0.70711+0.j     ]], dtype= np.complex128)
    cutoff = np.array([3], dtype=np.uint64)
    d = np.array([2], dtype=np.uint64)
    upstream = np.ones(perm.total_size(interferometer(), 3), dtype=np.complex128)
    grad = perm.fs_interferometer_grad(cutoff, d, interferometer(), upstream)
    validate_backward(cutoff[0], interferometer(),grad, upstream)