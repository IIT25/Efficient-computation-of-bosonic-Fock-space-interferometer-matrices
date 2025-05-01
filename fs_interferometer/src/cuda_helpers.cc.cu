#include "matrix.hpp"
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cuComplex.h>
#include <cuda/std/complex>

__device__ double atomicMul(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val * __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
__device__ double atomicDiv(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(__longlong_as_double(assumed) / val));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ void binom_coeff(unsigned long *t_, unsigned long *k_, size_t tid,
                            double *result) {
  unsigned long k = *k_;
  unsigned long t = *t_;
  if (tid < k / 2) {
    double temp_result = (2 * tid + 2) * (2 * tid + 1);
    atomicDiv(result, temp_result);
  } else if (k % 2 == 1 && tid == k / 2) {
    atomicMul(result, t);
    atomicDiv(result, k);
  } else if (tid < k) {
    double temp_result = (t - (2 * tid - k)) * (t - (2 * tid - k) - 1);
    atomicMul(result, temp_result);
  } else {
    return;
  }
}

__device__ int binomialCoeff(int n, int k) {
  if (n < 0) {
    return 0;
  }
  if (k == 0) {
    return 1;
  }
  return (n * binomialCoeff(n - 1, k - 1)) / k;
}
__device__ int symmetric_subspace_cardinality(int d, int n) {
  return binomialCoeff(d + n - 1, n);
}
__device__ Matrix<int> partitions(int boxes, int particles,
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
  Matrix<int> separators = Matrix<int>(1, boxes - 1);
  separators.iota(0);
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

__device__ Matrix<int> cutoff_fock_space_dim_array(Matrix<int> cutoff, int d) {
  Matrix<int> ret = Matrix<int>(cutoff.rows, cutoff.cols);
  for (size_t i = 0; i < cutoff.cols; i++) {
    ret[i] = binomialCoeff(d + cutoff[i] - 1, d);
  }
  return ret;
}

__device__ int get_index_in_fock_subspace(Matrix<int> element) {

  int sum_ = 0;
  int accumulator = 0;
  for (size_t i = 0; i < element.cols - 1; i++) {
    sum_ += element[element.cols - i - 1];
    accumulator += binomialCoeff(sum_ + i, i + 1);
  }
  return accumulator;
}

__device__ Matrix<cuda::std::complex<double>> _calculate_subspace_grad(
    int row_index, int col_index,
    Matrix<cuda::std::complex<double>> previous_subspace_representation,
    Matrix<int> subspace_indices, Matrix<int> first_subspace_indices,
    Matrix<int> first_nonzero_indices, Matrix<double> sqrt_occupation_numbers,
    Matrix<double> sqrt_first_occupation_numbers,
    Matrix<cuda::std::complex<double>> interferometer,
    Matrix<cuda::std::complex<double>> previous_subspace_grad) {
  int matrix_dim = sqrt_occupation_numbers.rows;
  Matrix<cuda::std::complex<double>> subspace_grad =
      Matrix<cuda::std::complex<double>>(matrix_dim, matrix_dim);
  for (int jdx = 0; jdx < matrix_dim; jdx++) {
    for (int idx = 0; idx < first_nonzero_indices.cols; idx++) {
      int first_nonzero_index = first_nonzero_indices[idx];
      if (first_nonzero_index != row_index) {
        continue;
      }
      subspace_grad(idx, jdx) +=
          previous_subspace_representation(first_subspace_indices[idx],
                                           subspace_indices(jdx, col_index)) *
          sqrt_occupation_numbers(jdx, col_index);
    }
    for (int idx = 0; idx < matrix_dim; idx++) {
      for (int kdx = 0; kdx < sqrt_occupation_numbers.cols; kdx++) {
        subspace_grad(idx, jdx) +=
            sqrt_occupation_numbers(jdx, kdx) *
            interferometer(first_nonzero_indices[idx], kdx) *
            previous_subspace_grad(first_subspace_indices[idx],
                                   subspace_indices(jdx, kdx));
      }
      subspace_grad(idx, jdx) /= sqrt_first_occupation_numbers[idx];
    }
  }
  return subspace_grad;
}
__device__ Matrix<cuda::std::complex<double>>
conj(Matrix<cuda::std::complex<double>> m) {
  Matrix<cuda::std::complex<double>> res =
      Matrix<cuda::std::complex<double>>(m.rows, m.cols);
  for (int idx = 0; idx < m.size(); idx++) {
    res[idx] = cuda::std::complex<double>((*(m.data + idx)).real(),
                                          -(*(m.data + idx)).imag());
  }
  return res;
}

__device__ cuda::std::complex<double>
einsum_ij_ij(Matrix<cuda::std::complex<double>> m,
             Matrix<cuda::std::complex<double>> n) {
  size_t eq_rows = min(n.rows, m.rows);
  size_t eq_cols = min(n.cols, m.cols);
  cuda::std::complex<double> sum = 0;
  for (int row_idx = 0; row_idx < eq_rows; row_idx++) {
    for (int col_idx = 0; col_idx < eq_cols; col_idx++) {
      sum += n(row_idx, col_idx) * m(row_idx, col_idx);
    }
  }
  return sum;
}