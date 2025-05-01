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
#include <vector>

int binomialCoeff(int n, int k) {
  if (n < 0) {
    return 0;
  }
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

Matrix<int> get_fock_space_basis(int d, int cutoff) {
  int size = cutoff_fock_space_dim(cutoff, d);

  Matrix<int> ret = Matrix<int>(size, d);
  int current_row = 0;
  for (size_t n = 0; n < cutoff; n++) {
    int num_rows = symmetric_subspace_cardinality(d, n);
    Matrix<int> out = ret.rowsliceR(current_row, current_row + num_rows);
    partitions(d, n, out);
    current_row += num_rows;
  }
  return ret;
}

int get_index_in_fock_subspace(Matrix<int> element) {

  int sum_ = 0;
  int accumulator = 0;
  for (size_t i = 0; i < element.cols - 1; i++) {
    sum_ += element[element.cols - i - 1];
    accumulator += binomialCoeff(sum_ + i, i + 1);
  }
  return accumulator;
}

Matrix<int> cutoff_fock_space_dim_array(Matrix<int> cutoff, int d) {
  Matrix<int> ret = Matrix<int>(cutoff.rows, cutoff.cols);
  for (size_t i = 0; i < cutoff.cols; i++) {
    ret[i] = binomialCoeff(d + cutoff[i] - 1, d);
  }
  return ret;
}

std::tuple<std::vector<Matrix<int>>, std::vector<Matrix<int>>,
           std::vector<Matrix<int>>, std::vector<Matrix<double>>,
           std::vector<Matrix<double>>>
calculate_interferometer_helper_indices(int d, int cutoff) {
  Matrix<int> space = get_fock_space_basis(d = d, cutoff = cutoff);
  Matrix<int> basis = Matrix<int>(space.rows, d);
  Matrix<int> first_subpace_indices_space = Matrix<int>(1, space.rows);
  Matrix<double> sqrt_first_occupation_numbers = Matrix<double>(1, space.rows);
  Matrix<int> first_nonzero_space_index = Matrix<int>(1, space.rows);
  Matrix<double> sqrt_space = Matrix<double>(space.rows, space.cols);

  for (size_t i = 0; i < space.rows; i++) {
    Matrix<int> current_basis = space.rowidx(i);
    Matrix<double> sqrt_source;
    sqrt_space.rowidxR(i, &sqrt_source);
    current_basis.sqrt(sqrt_source);
    bool found_first = false;
    for (size_t j = 0; j < d; j++) {
      current_basis[j] -= 1;
      basis(i, j) = get_index_in_fock_subspace(current_basis);
      if (!found_first && current_basis[j] >= 0) {
        first_nonzero_space_index[i] = j;
        first_subpace_indices_space[i] = basis(i, j);
        found_first = true;
        sqrt_first_occupation_numbers[i] = sqrt_space(i, j);
      }
      current_basis[j] += 1;
    }
  }

  Matrix<int> cutoffM = Matrix<int>(1, cutoff);
  cutoffM.iota(1);
  std::vector<Matrix<int>> subspace_index_tensor = std::vector<Matrix<int>>();
  std::vector<Matrix<int>> first_subspace_index_tensor =
      std::vector<Matrix<int>>();
  std::vector<Matrix<int>> first_nonzero_index_tensor =
      std::vector<Matrix<int>>();
  std::vector<Matrix<double>> sqrt_occupation_numbers_tensor =
      std::vector<Matrix<double>>();
  std::vector<Matrix<double>> sqrt_first_occupation_numbers_tensor =
      std::vector<Matrix<double>>();
  Matrix<int> indices = cutoff_fock_space_dim_array(cutoffM, d);
  for (size_t n = 2; n < cutoff; n++) {
    Matrix<int> subspace_range = Matrix<int>(1, indices[n] - indices[n - 1]);
    subspace_range.iota(indices[n - 1]);
    subspace_index_tensor.push_back(
        (*basis[subspace_range]).mod(indices[n - 1] - indices[n - 2]));

    first_nonzero_index_tensor.push_back(
        (*first_nonzero_space_index[subspace_range]));
    first_subspace_index_tensor.push_back(
        (*first_subpace_indices_space[subspace_range]));

    sqrt_occupation_numbers_tensor.push_back((*sqrt_space[subspace_range]));
    sqrt_first_occupation_numbers_tensor.push_back(
        (*sqrt_first_occupation_numbers[subspace_range]));
  }
  return std::make_tuple(subspace_index_tensor, first_nonzero_index_tensor,
                         first_subspace_index_tensor,
                         sqrt_occupation_numbers_tensor,
                         sqrt_first_occupation_numbers_tensor);
}

#endif