#ifndef CALC_HELPER
#define CALC_HELPER

#include "conversions.cpp"
#include "matrix.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <ostream>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vcruntime_typeinfo.h>
#include <vector>

auto calculate_interferometer_helper_indices(int d, int cutoff) {
  // space = nb_get_fock_space_basis(d=d, cutoff=cutoff);
}
/*def calculate_interferometer_helper_indices(d, cutoff):
    space = nb_get_fock_space_basis(d=d, cutoff=cutoff)

    basis = np.empty((space.shape[0], d), dtype=space.dtype)
    first_subpace_indices_space = np.empty(shape=space.shape[0],
   dtype=space.dtype) sqrt_first_occupation_numbers = np.empty(len(space),
   dtype=np.float64)

    first_nonzero_space_index = np.empty(shape=space.shape[0],
   dtype=space.dtype)

    sqrt_space = np.empty(shape=space.shape, dtype=np.float64)

    for i in range(len(space)):
        current_basis = space[i]
        sqrt_space[i] = np.sqrt(current_basis)
        found_first = False
        for j in range(d):
            current_basis[j] -= 1
            basis[i, j] = get_index_in_fock_subspace(current_basis)

            if not found_first and current_basis[j] >= 0:
                first_nonzero_space_index[i] = j
                first_subpace_indices_space[i] = basis[i, j]
                found_first = True
                sqrt_first_occupation_numbers[i] = sqrt_space[i, j]

            current_basis[j] += 1

    subspace_index_tensor = []

    first_subspace_index_tensor = []
    first_nonzero_index_tensor = []

    sqrt_occupation_numbers_tensor = []
    sqrt_first_occupation_numbers_tensor = []

    indices = cutoff_fock_space_dim_array(cutoff=np.arange(1, cutoff + 1), d=d)

    for n in range(2, cutoff):
        subspace_range = np.arange(indices[n - 1], indices[n])
        subspace_index_tensor.append(
            np.mod(basis[subspace_range], indices[n - 1] - indices[n - 2])
        )

        first_nonzero_index_tensor.append(first_nonzero_space_index[subspace_range])
        first_subspace_index_tensor.append(first_subpace_indices_space[subspace_range])

        sqrt_occupation_numbers_tensor.append(sqrt_space[subspace_range])
        sqrt_first_occupation_numbers_tensor.append(
            sqrt_first_occupation_numbers[subspace_range]
        )

    return (
        subspace_index_tensor,
        first_nonzero_index_tensor,
        first_subspace_index_tensor,
        sqrt_occupation_numbers_tensor,
        sqrt_first_occupation_numbers_tensor,
    )
*/

int binomialCoeff(int n, int k) {
  if (k == 0) {
    return 1;
  }
  return (n * binomialCoeff(n - 1, k - 1)) / k;
}

int cutoff_fock_space_dim(int cutoff, int d) {
  return binomialCoeff(d + cutoff - 1, d);
}

int symmetric_subspace_cardinality(int d, int n) {
  return binomialCoeff(d + n - 1, n);
}

Matrix<int> partitions(int boxes, int particles,
                       const Matrix<int> &out = Matrix<int>()) {
  int positions = particles + boxes - 1;
  if (positions == -1 || boxes == 0) {
    return Matrix<int>(1, 0);
  }
  int size = binomialCoeff(positions, boxes - 1);
  Matrix<int> result;
  if (out.data == nullptr) {
    result = Matrix<int>(size, boxes);
  } else {
    result = out;
  }
  std::vector<int> separators(boxes - 1);
  std::iota(std::begin(separators), std::end(separators), 0);
  int index = size - 1;
  while (true) {
    int prev = -1;
    for (size_t i = 0; i < boxes - 1; i++) {
      result(index, i) = separators[i] - prev - 1;
      prev = separators[i];
    }
    result(index, boxes - 1) = positions - prev - 1;
    index -= 1;

    if (index < 0) {
      break;
    }
    int i = boxes - 2;
    while (separators[i] == positions - (boxes - 1 - i)) {
      i -= 1;
    }

    separators[i] += 1;
    for (size_t j = i + 1; j < boxes - 1; j++) {
      separators[j] = separators[j - 1] + 1;
    }
  }
  return result;
}

/*def nb_get_fock_space_basis(d: int, cutoff: int) -> np.ndarray:
    size = cutoff_fock_space_dim(cutoff=cutoff, d=d)

    ret = np.empty((size, d), dtype=np.int32)
    current_row = 0
    for n in range(cutoff):
        num_rows = symmetric_subspace_cardinality(d, n)
        out = ret[current_row : current_row + num_rows, :]
        _ = partitions(boxes=d, particles=n, out=out)
        current_row += num_rows

    return ret
*/
pybind11::array_t<int> get_fock_space_basis(int d, int cutoff) {
  int size = cutoff_fock_space_dim(cutoff, d);

  Matrix<int> ret = Matrix<int>(size, d);
  ret.printdim();
  int current_row = 0;
  for (size_t n = 0; n < cutoff; n++) {
    int num_rows = symmetric_subspace_cardinality(d, n);
    Matrix<int> out = ret.rowslice(current_row, current_row + num_rows);
    partitions(d, n, out);
    out.print();
    current_row += num_rows;
  }
  return to_pyarray(ret);
}

#endif