#ifndef CALC_PERM_CPP
#define CALC_PERM_CPP

#include "calc_helper.cpp"
#include "conversions.cpp"
#include "matrix.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <ostream>
#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <tuple>
#include <type_traits>
#include <vcruntime_typeinfo.h>
#include <vector>

std::vector<py::array_t<std::complex<double>>>
calc_perm(Matrix<std::complex<double>> interferometer,
          std::tuple<std::vector<Matrix<int>>, std::vector<Matrix<int>>,
                     std::vector<Matrix<int>>, std::vector<Matrix<double>>,
                     std::vector<Matrix<double>>>
              helper_indices) {
  // declare, init
  std::vector<Matrix<int>> subspace_indices_array = std::get<0>(helper_indices);
  std::vector<Matrix<int>> first_nonzero_indices_array =
      std::get<1>(helper_indices);
  std::vector<Matrix<int>> first_subspace_indices_array =
      std::get<2>(helper_indices);
  std::vector<Matrix<double>> sqrt_occupation_numbers_array =
      std::get<3>(helper_indices);
  std::vector<Matrix<double>> sqrt_first_occupation_numbers_array =
      std::get<4>(helper_indices);

  const size_t cutoff = subspace_indices_array.size() + 2;
  std::vector<Matrix<std::complex<double>>> subspace_representations =
      std::vector<Matrix<std::complex<double>>>();
  Matrix<std::complex<double>> first =
      Matrix<std::complex<double>>(1, 1, new std::complex<double>(1));
  subspace_representations.push_back(first);
  subspace_representations.push_back(interferometer);

  std::vector<py::array_t<std::complex<double>>> array_ts =
      std::vector<py::array_t<std::complex<double>>>();
  for (size_t n = 0; n < cutoff - 2; n++) {
    Matrix<int> subspace_indices = subspace_indices_array[n];
    Matrix<int> first_subspace_indices = first_subspace_indices_array[n];
    Matrix<int> first_nonzero_indices = first_nonzero_indices_array[n];
    Matrix<double> sqrt_occupation_numbers = sqrt_occupation_numbers_array[n];
    Matrix<double> sqrt_first_occupation_numbers =
        sqrt_first_occupation_numbers_array[n];

    Matrix<std::complex<double>> previous_representation =
        subspace_representations[n + 1];
    Matrix<std::complex<double>> representation = Matrix<std::complex<double>>(
        first_nonzero_indices.cols, sqrt_occupation_numbers.rows);

    for (size_t k = 0; k < first_nonzero_indices.cols; k++) {
      std::complex<double> denominator = sqrt_first_occupation_numbers[k];
      Matrix<std::complex<double>> previous_representation_indexed =
          previous_representation.rowidx(first_subspace_indices[k]);
      for (size_t j = 0; j < sqrt_occupation_numbers.cols; j++) {
        std::complex<double> one_particle_contrib =
            interferometer(first_nonzero_indices[k], j) / denominator;
        for (size_t i = 0; i < sqrt_occupation_numbers.rows; i++) {
          representation(k, i) +=
              one_particle_contrib * sqrt_occupation_numbers(i, j) *
              previous_representation_indexed[subspace_indices(i, j)];
        }
      }
    }
    subspace_representations.push_back(representation);
  }

  // conversion to py::array_t
  for (Matrix<std::complex<double>> m : subspace_representations) {
    array_ts.push_back(to_pyarray(m));
  }
  return array_ts;
}
std::vector<int>
calc_perm_nc(Matrix<std::complex<double>> interferometer,
             std::tuple<std::vector<Matrix<int>>, std::vector<Matrix<int>>,
                        std::vector<Matrix<int>>, std::vector<Matrix<double>>,
                        std::vector<Matrix<double>>>
                 helper_indices,
             Matrix<std::complex<double>> &first) {
  // declare, init
  std::vector<Matrix<int>> subspace_indices_array = std::get<0>(helper_indices);
  std::vector<Matrix<int>> first_nonzero_indices_array =
      std::get<1>(helper_indices);
  std::vector<Matrix<int>> first_subspace_indices_array =
      std::get<2>(helper_indices);
  std::vector<Matrix<double>> sqrt_occupation_numbers_array =
      std::get<3>(helper_indices);
  std::vector<Matrix<double>> sqrt_first_occupation_numbers_array =
      std::get<4>(helper_indices);

  const size_t cutoff = subspace_indices_array.size() + 2;
  std::vector<Matrix<std::complex<double>>> subspace_representations =
      std::vector<Matrix<std::complex<double>>>();
  subspace_representations.push_back(first);
  subspace_representations.push_back(interferometer);

  std::vector<py::array_t<std::complex<double>>> array_ts =
      std::vector<py::array_t<std::complex<double>>>();
  for (size_t n = 0; n < cutoff - 2; n++) {
    Matrix<int> subspace_indices = subspace_indices_array[n];
    Matrix<int> first_subspace_indices = first_subspace_indices_array[n];
    Matrix<int> first_nonzero_indices = first_nonzero_indices_array[n];
    Matrix<double> sqrt_occupation_numbers = sqrt_occupation_numbers_array[n];
    Matrix<double> sqrt_first_occupation_numbers =
        sqrt_first_occupation_numbers_array[n];

    Matrix<std::complex<double>> previous_representation =
        subspace_representations[n + 1];
    Matrix<std::complex<double>> representation = Matrix<std::complex<double>>(
        first_nonzero_indices.cols, sqrt_occupation_numbers.rows);

    for (size_t k = 0; k < first_nonzero_indices.cols; k++) {
      std::complex<double> denominator = sqrt_first_occupation_numbers[k];
      Matrix<std::complex<double>> previous_representation_indexed =
          previous_representation.rowidx(first_subspace_indices[k]);
      for (size_t j = 0; j < sqrt_occupation_numbers.cols; j++) {
        std::complex<double> one_particle_contrib =
            interferometer(first_nonzero_indices[k], j) / denominator;
        for (size_t i = 0; i < sqrt_occupation_numbers.rows; i++) {
          representation(k, i) +=
              one_particle_contrib * sqrt_occupation_numbers(i, j) *
              previous_representation_indexed[subspace_indices(i, j)];
        }
      }
    }
    subspace_representations.push_back(representation);
  }
  std::vector<int> dims = std::vector<int>();
  dims.push_back(first.rows);
  for (int i = 1; i < subspace_representations.size(); i++) {
    first = first.horizontal_concat(subspace_representations[i]);
    dims.push_back(subspace_representations[i].rows);
  }

  return dims;
}

std::vector<py::array_t<std::complex<double>>>
_get_interferometer_on_fock_space(
    py::array_t<std::complex<double>,
                pybind11::array::c_style | py::array::forcecast>
        interferometer,
    int cutoff) {
  Matrix<std::complex<double>> interf = numpy_to_matrix(interferometer);
  return calc_perm(
      interf, calculate_interferometer_helper_indices(interf.rows, cutoff));
}
std::vector<int> _get_interferometer_on_fock_space_nc(
    Matrix<std::complex<double>> interferometer, int cutoff,
    Matrix<std::complex<double>> &out) {
  std::vector<int> dims = calc_perm_nc(
      interferometer,
      calculate_interferometer_helper_indices(interferometer.rows, cutoff),
      out);
  return dims;
}
#endif