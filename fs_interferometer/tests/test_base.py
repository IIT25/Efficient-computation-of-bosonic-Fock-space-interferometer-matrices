import numpy as np
import fs_interferometer
import pytest
from fs_interferometer import _get_interferometer_on_fock_space_xla
from validators import validate_forward
import jax

jax.config.update("jax_platform_name", "cpu")


def test_forward_1_4():
    def interferometer():
        return np.array([[0.5 + 0.86603j]])

    cutoff = np.array([4], dtype=np.uint64)
    d = np.array([1], dtype=np.uint64)
    resj = fs_interferometer._get_interferometer_on_fock_space_xla(
        cutoff, d, interferometer()
    )
    validate_forward(4, interferometer(), resj)


def test_forward_2_3():
    def interferometer():
        return np.array(
            [
                [0.70711 + 0.0j, -0.35355 + 0.61237j],
                [0.35355 + 0.61237j, 0.70711 + 0.0j],
            ]
        )

    cutoff = np.array([3], dtype=np.uint64)
    d = np.array([2], dtype=np.uint64)
    resj = fs_interferometer._get_interferometer_on_fock_space_xla(
        cutoff, d, interferometer()
    )
    validate_forward(3, interferometer(), resj)


def test_forward_5_3():
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

    cutoff = np.array([3], dtype=np.uint64)
    d = np.array([5], dtype=np.uint64)
    resj = fs_interferometer._get_interferometer_on_fock_space_xla(
        cutoff, d, interferometer()
    )
    validate_forward(3, interferometer(), resj)


def test_forward_1_6():
    def interferometer():
        return np.array([[0.5 + 0.86603j]])

    cutoff = np.array([6], dtype=np.uint64)
    d = np.array([1], dtype=np.uint64)
    resj = fs_interferometer._get_interferometer_on_fock_space_xla(
        cutoff, d, interferometer()
    )
    validate_forward(6, interferometer(), resj)


def test_forward_2_6():
    def interferometer():
        return np.array([[0.86603 + 0.0j, 0.0 - 0.5j], [-0.0 - 0.5j, 0.86603 + 0.0j]])

    cutoff = np.array([6], dtype=np.uint64)
    d = np.array([2], dtype=np.uint64)
    resj = fs_interferometer._get_interferometer_on_fock_space_xla(
        cutoff, d, interferometer()
    )
    validate_forward(6, interferometer(), resj)


def test_invalid_interferometer():
    def interferometer():
        return np.array([[0.86603 + 0.0j, 0.0 - 0.5j]])

    cutoff = np.array([3], dtype=np.uint64)
    d = np.array([2], dtype=np.uint64)
    with pytest.raises(Exception) as e_info:
        _get_interferometer_on_fock_space_xla(cutoff, d, interferometer())
    assert e_info.value.args[0] == "Interferometer has to be a square matrix"
    assert e_info.value.args[1] == "1 x 2"


def test_invalid_cutoff():
    def interferometer():
        return np.array([[0.86603 + 0.0j, 0.0 - 0.5j]])

    cutoff = np.array([0], dtype=np.uint64)
    d = np.array([2], dtype=np.uint64)
    with pytest.raises(Exception) as e_info:
        _get_interferometer_on_fock_space_xla(cutoff, d, interferometer())
    assert e_info.value.args[0] == "Cutoff has to be at least 2"
    assert e_info.value.args[1] == "0"
