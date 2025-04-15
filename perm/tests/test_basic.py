from __future__ import annotations

import scikit_build_example as m
import numpy as np
import perm

def test_version():
    def interferometer():
        return np.array(
            [
                [
                    -0.11035524 + 0.43053175j,
                    0.16672794 - 0.47819775j,
                    0.01831264 - 0.07497556j,
                    0.44214383 + 0.23308614j,
                    0.49732734 - 0.20707815j,
                ],
                [
                    -0.2226116 - 0.39735917j,
                    0.30063104 - 0.55866103j,
                    0.00329231 - 0.28009116j,
                    0.23376629 - 0.22854039j,
                    -0.43588261 + 0.12139052j,
                ],
                [
                    0.24737463 - 0.24638855j,
                    -0.31444021 - 0.28467995j,
                    -0.16425914 - 0.33655956j,
                    -0.26884137 + 0.68435951j,
                    -0.07230938 - 0.10989756j,
                ],
                [
                    0.34411568 + 0.00861778j,
                    -0.26208576 - 0.04842002j,
                    0.64971999 - 0.05882925j,
                    0.29164646 + 0.0682574j,
                    0.01229276 + 0.54314998j,
                ],
                [
                    -0.59541546 - 0.01014536j,
                    -0.28295784 + 0.10016806j,
                    0.57604147 - 0.13381814j,
                    -0.07509227 + 0.08557502j,
                    -0.12237335 - 0.42143858j,
                ],
            ], dtype=np.complex128
    )
    cutoff = np.array([3], dtype=np.uint64)
    resj, dimsj, helper_idxj, helper_sqrtj = perm._get_interferometer_on_fock_space_fwd(cutoff, interferometer())
    print(dimsj)
    assert resj[0] == np.array([1.0], dtype=np.complex128)
    assert np.isclose(resj[1][0][0], interferometer()[0][0])
    print(resj)
    print(helper_idxj)
    print(helper_sqrtj)
    assert m.__version__ == "0.0.1"

