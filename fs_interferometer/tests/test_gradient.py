import numpy as np
import fs_interferometer
import pytest
from validators import validate_backward
import numpy as np
import jax

jax.config.update("jax_platform_name", "cpu")


def test_backward_2_3_ones():
    def interferometer():
        return np.array(
            [
                [0.70711 + 0.0j, -0.35355 + 0.61237j],
                [0.35355 + 0.61237j, 0.70711 + 0.0j],
            ],
            dtype=np.complex128,
        )

    cutoff = np.array([3], dtype=np.uint64)
    d = np.array([2], dtype=np.uint64)
    upstream = np.ones(
        fs_interferometer.total_size(interferometer(), 3), dtype=np.complex128
    )
    grad = fs_interferometer.fs_interferometer_grad(
        cutoff, d, interferometer(), upstream
    )
    validate_backward(cutoff[0], interferometer(), grad, upstream)


def test_backward_1_3_pimag():
    def interferometer():
        return np.array([[0.5 + 0.86603j]], dtype=np.complex128)

    cutoff = np.array([3], dtype=np.uint64)
    d = np.array([1], dtype=np.uint64)
    upstream = np.zeros(
        fs_interferometer.total_size(interferometer(), 3), dtype=np.complex128
    )
    for i in range(len(upstream)):
        upstream[i] = 0.0 + (i * 1.1j)
    grad = fs_interferometer.fs_interferometer_grad(
        cutoff, d, interferometer(), upstream
    )
    validate_backward(cutoff[0], interferometer(), grad, upstream)


def test_backward_5_4_pimag():
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
            ],
            dtype=np.complex128,
        )

    cutoff = np.array([4], dtype=np.uint64)
    d = np.array([5], dtype=np.uint64)
    upstream = np.zeros(
        fs_interferometer.total_size(interferometer(), 4), dtype=np.complex128
    )
    for i in range(len(upstream)):
        upstream[i] = 0.0 + (i * 1.1j)
    grad = fs_interferometer.fs_interferometer_grad(
        cutoff, d, interferometer(), upstream
    )
    validate_backward(cutoff[0], interferometer(), grad, upstream)


def test_backward_1_6_realneg():
    def interferometer():
        return np.array([[0.5 + 0.86603j]], dtype=np.complex128)

    cutoff = np.array([6], dtype=np.uint64)
    d = np.array([1], dtype=np.uint64)
    upstream = np.zeros(
        fs_interferometer.total_size(interferometer(), 6), dtype=np.complex128
    )
    for i in range(len(upstream)):
        upstream[i] = (-1.1) * i + 0.0j
    print(upstream)
    grad = fs_interferometer.fs_interferometer_grad(
        cutoff, d, interferometer(), upstream
    )
    validate_backward(cutoff[0], interferometer(), grad, upstream)


def test_backward_2_6_mixed1():
    def interferometer():
        return np.array(
            [[0.86603 + 0.0j, 0.0 - 0.5j], [-0.0 - 0.5j, 0.86603 + 0.0j]],
            dtype=np.complex128,
        )

    cutoff = np.array([6], dtype=np.uint64)
    d = np.array([2], dtype=np.uint64)
    upstream = np.zeros(
        fs_interferometer.total_size(interferometer(), 6), dtype=np.complex128
    )
    for i in range(len(upstream)):
        upstream[i] = i * (-1.1) + (i % 2) * 0.9j
        if i % 3 == 0:
            upstream[i] *= -1.0
    print(upstream)
    grad = fs_interferometer.fs_interferometer_grad(
        cutoff, d, interferometer(), upstream
    )
    validate_backward(cutoff[0], interferometer(), grad, upstream)


def test_backward_2_4_mixed2():
    def interferometer():
        return np.array(
            [[0.86603 + 0.0j, 0.0 - 0.5j], [-0.0 - 0.5j, 0.86603 + 0.0j]],
            dtype=np.complex128,
        )

    cutoff = np.array([4], dtype=np.uint64)
    d = np.array([2], dtype=np.uint64)
    upstream = np.zeros(
        fs_interferometer.total_size(interferometer(), 4), dtype=np.complex128
    )
    for i in range(len(upstream)):
        upstream[i] = i * (-1.1) + (i % 2) * 0.9j
        if i % 3 == 0:
            upstream[i] *= -1.0
    grad = fs_interferometer.fs_interferometer_grad(
        cutoff, d, interferometer(), upstream
    )
    validate_backward(cutoff[0], interferometer(), grad, upstream)
