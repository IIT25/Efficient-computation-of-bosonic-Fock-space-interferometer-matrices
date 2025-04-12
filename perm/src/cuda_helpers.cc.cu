#include "matrix.hpp"
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