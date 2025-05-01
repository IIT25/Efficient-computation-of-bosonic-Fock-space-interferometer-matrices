from piquasso._simulators.fock.calculations import calculate_interferometer_helper_indices
from piquasso._simulators.connectors.numpy_.interferometer import calculate_interferometer_on_fock_space
import numpy as np


def _calculate_subspace_grad(
    row_index,
    col_index,
    previous_subspace_representation,
    subspace_index_tuple,
    interferometer,
    previous_subspace_grad,
):
    (
        subspace_indices,
        first_nonzero_indices,
        first_subspace_indices,
        sqrt_occupation_numbers,
        sqrt_first_occupation_numbers,
    ) = subspace_index_tuple

    matrix_dim = sqrt_occupation_numbers.shape[0]

    subspace_grad = np.zeros(shape=(matrix_dim, matrix_dim), dtype=interferometer.dtype)

    for jdx in range(matrix_dim):
        for idx, first_nonzero_index in enumerate(first_nonzero_indices):
            if first_nonzero_index != row_index:
                continue

            subspace_grad[idx, jdx] += (
                previous_subspace_representation[
                    first_subspace_indices[idx],
                    subspace_indices[jdx, col_index],
                ]
                * sqrt_occupation_numbers[jdx, col_index]
            )

        for idx in range(matrix_dim):
            for kdx in range(sqrt_occupation_numbers.shape[1]):
                subspace_grad[idx, jdx] += (
                    sqrt_occupation_numbers[jdx, kdx]
                    * interferometer[first_nonzero_indices[idx], kdx]
                    * previous_subspace_grad[
                        first_subspace_indices[idx], subspace_indices[jdx, kdx]
                    ]
                )

            subspace_grad[idx, jdx] /= sqrt_first_occupation_numbers[idx]

    return subspace_grad
def interferometer_gradient(interferometer, subspace_representations, index_tuple, upstream):
        d = len(interferometer)
        cutoff = len(index_tuple[0]) + 2

        (
            subspace_index_tensor,
            first_nonzero_index_tensor,
            first_subspace_index_tensor,
            sqrt_occupation_numbers_tensor,
            sqrt_first_occupation_numbers_tensor,
        ) = index_tuple

        full_kl_grad = []
        for row_index in range(d):
            full_kl_grad.append([0] * d)
            for col_index in range(d):
                second_subspace_grad = np.zeros(
                    shape=interferometer.shape, dtype=interferometer.dtype
                )
                second_subspace_grad[row_index, col_index] = 1.0

                previous_subspace_grad = second_subspace_grad

                for p in range(2, cutoff):
                    previous_subspace_representation = subspace_representations[p - 1]

                    subspace_index_tuple = (
                        subspace_index_tensor[p - 2],
                        first_nonzero_index_tensor[p - 2],
                        first_subspace_index_tensor[p - 2],
                        sqrt_occupation_numbers_tensor[p - 2],
                        sqrt_first_occupation_numbers_tensor[p - 2],
                    )

                    subspace_grad = _calculate_subspace_grad(
                        row_index,
                        col_index,
                        previous_subspace_representation,
                        subspace_index_tuple,
                        interferometer,
                        previous_subspace_grad,
                    )

                    full_kl_grad[row_index][col_index] += np.einsum(
                        "ij,ij", upstream[p], np.conj(subspace_grad)
                    )

                    previous_subspace_grad = subspace_grad

        partial_result = np.array(full_kl_grad, dtype=interferometer.dtype)
        one_particle_term = upstream[1]
        print(partial_result)
        result = partial_result + one_particle_term

        return result



def validate_forward(cutoff, interferometer, resj):
    index_tuple = calculate_interferometer_helper_indices(
            d=len(interferometer), cutoff=cutoff
        )
    subspace_representations = calculate_interferometer_on_fock_space(
            interferometer, index_tuple
        )
    sub_flattened = []
    for i in range(len(subspace_representations)):
        sub_flattened = np.concatenate((sub_flattened, subspace_representations[i].ravel()))
    res_flattened = []
    for i in range(len(resj)):
        res_flattened = np.concatenate((res_flattened, resj[i].ravel()))
    for i in range(len(sub_flattened)):
        assert np.isclose(res_flattened[i], sub_flattened[i])

def validate_backward(cutoff, interferometer, gradj, upstream_buff):
    d = len(interferometer)
    index_tuple = calculate_interferometer_helper_indices(
            d=d, cutoff=cutoff
        )
    subspace_representations = calculate_interferometer_on_fock_space(
            interferometer, index_tuple
        )
    sub_flattened = []
    for i in range(len(subspace_representations)):
        sub_flattened = np.concatenate((sub_flattened, subspace_representations[i].ravel()))
    upstream = []
    idx = 0
    for i in range(cutoff):
        size = len(subspace_representations[i])
        upstream.append(np.zeros((size, size), dtype=np.complex128))
        for j in range(size):
            for k in range(size):
                upstream[i][j][k] = upstream_buff[idx]
                idx += 1
    grad = interferometer_gradient(interferometer, subspace_representations, index_tuple, upstream)
    for i in range(d):
        for j in range(d):
            assert np.isclose(gradj[i][j], grad[i][j])