#ifndef JAX_UTILS_CPP
#define JAX_UTILS_CPP

#include "calc_fs_interferometer.cpp"
#include "matrix.hpp"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
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
#include <utility>
#include <vector>

namespace ffi = xla::ffi;
// A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
// In this example, we treat all leading dimensions as batch dimensions, so this
// function returns the total number of elements in the buffer, and the size of
// the last dimension.
template <ffi::DataType T>
std::pair<size_t, size_t> GetDims(const ffi::Buffer<T> &buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

ffi::Error _get_interferometer_on_fock_space_Impl(
    ffi ::Buffer<ffi::U64> cutoff, ffi::Buffer<ffi::U64> d_,
    ffi::Buffer<ffi::C128> interferometer, ffi::ResultBuffer<ffi::C128> y,
    ffi::ResultBuffer<ffi::U64> y_dims) {
  int d = d_.typed_data()[0];
  Matrix<std::complex<double>> interferometerc =
      Matrix<std::complex<double>>(d, d, &(interferometer.typed_data()[0]));
  Matrix<std::complex<double>> res =
      Matrix<std::complex<double>>(1, 1, new std::complex<double>(1));
  std::vector<int> dims = std::vector<int>();
  _get_interferometer_on_fock_space_nc(interferometerc, cutoff.typed_data()[0],
                                       res, dims);
  ;
  for (size_t i = 0; i < res.size(); i++) {
    y->typed_data()[i] = res[i];
  }
  for (int i = 0; i < cutoff.typed_data()[0]; i++) {
    y_dims->typed_data()[i] = dims[i];
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(_get_interferometer_on_fock_space_xla,
                              _get_interferometer_on_fock_space_Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::U64>>());
ffi::Error _get_interferometer_on_fock_space_fwd_impl(
    ffi::Buffer<ffi::U64> cutoff, ffi::Buffer<ffi::U64> d,
    ffi::Buffer<ffi::C128> interferometer, ffi::ResultBuffer<ffi::C128> y,
    ffi::ResultBuffer<ffi::U64> y_dims, ffi::ResultBuffer<ffi::U32> helper_idx,
    ffi::ResultBuffer<ffi::F64> helper_sqrt) {
  auto [totalSize, lastDim] = GetDims(interferometer);
  Matrix<std::complex<double>> interferometerc = Matrix<std::complex<double>>(
      totalSize / lastDim, lastDim, &(interferometer.typed_data()[0]));
  Matrix<std::complex<double>> res =
      Matrix<std::complex<double>>(1, 1, new std::complex<double>(1));
  std::vector<int> dims = std::vector<int>();
  std::tuple<std::vector<Matrix<int>>, std::vector<Matrix<int>>,
             std::vector<Matrix<int>>, std::vector<Matrix<double>>,
             std::vector<Matrix<double>>>
      helper_indices = _get_interferometer_on_fock_space_nc(
          interferometerc, cutoff.typed_data()[0], res, dims);
  ;
  for (size_t i = 0; i < res.size(); i++) {
    y->typed_data()[i] = res[i];
  }
  for (int i = 0; i < cutoff.typed_data()[0]; i++) {
    y_dims->typed_data()[i] = dims[i];
  }
  std::vector<Matrix<int>> subspace_indices_array = std::get<0>(helper_indices);
  std::vector<Matrix<int>> first_nonzero_indices_array =
      std::get<1>(helper_indices);
  std::vector<Matrix<int>> first_subspace_indices_array =
      std::get<2>(helper_indices);
  std::vector<Matrix<double>> sqrt_occupation_numbers_array =
      std::get<3>(helper_indices);
  std::vector<Matrix<double>> sqrt_first_occupation_numbers_array =
      std::get<4>(helper_indices);
  int current_idx = 0;
  int current_sqrt_idx = 0;
  for (int i = 0; i < cutoff.typed_data()[0] - 2; i++) {
    for (int j = 0; j < subspace_indices_array[i].size(); j++) {
      helper_idx->typed_data()[current_idx] = subspace_indices_array[i][j];
      helper_sqrt->typed_data()[current_sqrt_idx] =
          sqrt_occupation_numbers_array[i][j];
      current_sqrt_idx += 1;
      current_idx += 1;
    }
    for (int j = 0; j < first_nonzero_indices_array[i].size(); j++) {
      helper_idx->typed_data()[current_idx] = first_nonzero_indices_array[i][j];
      current_idx += 1;
    }
    for (int j = 0; j < first_subspace_indices_array[i].size(); j++) {
      helper_idx->typed_data()[current_idx] =
          first_subspace_indices_array[i][j];
      current_idx += 1;
    }
    for (int j = 0; j < sqrt_first_occupation_numbers_array[i].size(); j++) {
      helper_sqrt->typed_data()[current_sqrt_idx] =
          sqrt_first_occupation_numbers_array[i][j];
      current_sqrt_idx += 1;
    }
  }
  return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    _get_interferometer_on_fock_space_fwd,
    _get_interferometer_on_fock_space_fwd_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Ret<ffi::Buffer<ffi::C128>>() // result
        .Ret<ffi::Buffer<ffi::U64>>()  // result dimensions
        .Ret<ffi::Buffer<ffi::U32>>()  // helper_idx
        .Ret<ffi::Buffer<ffi::F64>>()  // helper_sqrt
);

Matrix<std::complex<double>> _calculate_subspace_grad(
    int row_index, int col_index,
    Matrix<std::complex<double>> previous_subspace_representation,
    Matrix<int> subspace_indices, Matrix<int> first_subspace_indices,
    Matrix<int> first_nonzero_indices, Matrix<double> sqrt_occupation_numbers,
    Matrix<double> sqrt_first_occupation_numbers,
    Matrix<std::complex<double>> interferometer,
    Matrix<std::complex<double>> previous_subspace_grad) {
  int matrix_dim = sqrt_occupation_numbers.rows;
  Matrix<std::complex<double>> subspace_grad =
      Matrix<std::complex<double>>(matrix_dim, matrix_dim);
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
ffi::Error _get_interferometer_on_fock_space_bwd_impl(
    ffi::Buffer<ffi::U64> cutoff_, ffi::Buffer<ffi::C128> interferometer,
    ffi::Buffer<ffi::U64> d_, ffi::Buffer<ffi::C128> y,
    ffi::Buffer<ffi::U64> y_dim, ffi::Buffer<ffi::U32> helper_idx,
    ffi::Buffer<ffi::F64> helper_sqrt, ffi::Buffer<ffi::C128> upstream_buff,
    ffi::ResultBuffer<ffi::C128> result) {
  // unwrap helper_idx, helper_sqrt
  std::vector<Matrix<int>> subspace_index_tensor = std::vector<Matrix<int>>();
  std::vector<Matrix<int>> first_subspace_index_tensor =
      std::vector<Matrix<int>>();
  std::vector<Matrix<int>> first_nonzero_index_tensor =
      std::vector<Matrix<int>>();
  std::vector<Matrix<double>> sqrt_occupation_numbers_tensor =
      std::vector<Matrix<double>>();
  std::vector<Matrix<double>> sqrt_first_occupation_numbers_tensor =
      std::vector<Matrix<double>>();
  auto [totalSize_y_dim, lastDim_y_dim] = GetDims(y_dim);
  auto [totalSize, lastDim] = GetDims(interferometer);
  Matrix<std::complex<double>> interferometerc = Matrix<std::complex<double>>(
      totalSize / lastDim, lastDim, &(interferometer.typed_data()[0]));
  int c_idx = 0;
  int c_sqrt_idx = 0;
  for (int i = 2; i < totalSize_y_dim; i++) {
    size_t si_size = y_dim.typed_data()[i] * interferometerc.rows;
    size_t fni_size = y_dim.typed_data()[i];
    size_t fsi_size = y_dim.typed_data()[i];
    size_t son_size = y_dim.typed_data()[i] * interferometerc.rows;
    size_t sfon_size = y_dim.typed_data()[i];

    Matrix<int> si = Matrix<int>(
        y_dim.typed_data()[i], interferometerc.rows,
        (reinterpret_cast<int *>(&(helper_idx.typed_data()[c_idx]))));
    subspace_index_tensor.push_back(si);
    c_idx += si_size;

    Matrix<int> fni = Matrix<int>(
        1, y_dim.typed_data()[i],
        (reinterpret_cast<int *>(&(helper_idx.typed_data()[c_idx]))));
    first_nonzero_index_tensor.push_back(fni);
    c_idx += fni_size;

    Matrix<int> fsi = Matrix<int>(
        1, y_dim.typed_data()[i],
        (reinterpret_cast<int *>(&(helper_idx.typed_data()[c_idx]))));
    first_subspace_index_tensor.push_back(fsi);
    c_idx += fsi_size;

    Matrix<double> son = Matrix<double>(
        y_dim.typed_data()[i], interferometerc.rows,
        (reinterpret_cast<double *>(&(helper_sqrt.typed_data()[c_sqrt_idx]))));
    sqrt_occupation_numbers_tensor.push_back(son);
    c_sqrt_idx += son_size;

    Matrix<double> sfon = Matrix<double>(
        1, y_dim.typed_data()[i],
        (reinterpret_cast<double *>(&(helper_sqrt.typed_data()[c_sqrt_idx]))));
    sqrt_first_occupation_numbers_tensor.push_back(sfon);
    c_sqrt_idx += sfon_size;
  }
  // unwrap representations and upstream
  int result_start_idx = 0;
  std::vector<Matrix<std::complex<double>>> subspace_representations =
      std::vector<Matrix<std::complex<double>>>();
  std::vector<Matrix<std::complex<double>>> upstream =
      std::vector<Matrix<std::complex<double>>>();
  auto [total_size, n] = GetDims(y_dim);
  std::vector<int> y_dims_vec(y_dim.typed_data(), y_dim.typed_data() + n);
  for (int d : y_dims_vec) {
    Matrix<std::complex<double>> subspace_m =
        Matrix<std::complex<double>>(d, d, &(y.typed_data()[result_start_idx]));
    subspace_representations.push_back(subspace_m);
    Matrix<std::complex<double>> upstream_m = Matrix<std::complex<double>>(
        d, d, &(upstream_buff.typed_data()[result_start_idx]));
    upstream.push_back(upstream_m);
    result_start_idx += d * d;
  }
  // gradient
  int d = interferometerc.rows;
  int cutoff = totalSize_y_dim;
  Matrix<std::complex<double>> full_kl_grad =
      Matrix<std::complex<double>>(interferometerc.rows, interferometerc.cols);
  full_kl_grad.zeros();
  for (int row_index = 0; row_index < d; row_index++) {
    for (int col_index = 0; col_index < d; col_index++) {
      Matrix<std::complex<double>> second_subspace_grad =
          Matrix<std::complex<double>>(interferometerc.rows,
                                       interferometerc.cols);
      second_subspace_grad.zeros();
      second_subspace_grad(row_index, col_index) = 1.0;
      Matrix<std::complex<double>> previous_subspace_grad =
          second_subspace_grad;
      for (int p = 2; p < cutoff; p++) {
        Matrix<std::complex<double>> previous_subspace_representation =
            subspace_representations[p - 1];
        Matrix<int> subspace_indices = subspace_index_tensor[p - 2];
        Matrix<int> first_subspace_indices = first_subspace_index_tensor[p - 2];
        Matrix<int> first_nonzero_indices = first_nonzero_index_tensor[p - 2];
        Matrix<double> sqrt_occupation_numbers =
            sqrt_occupation_numbers_tensor[p - 2];
        Matrix<double> sqrt_first_occupation_numbers =
            sqrt_first_occupation_numbers_tensor[p - 2];
        Matrix<std::complex<double>> subspace_grad = _calculate_subspace_grad(
            row_index, col_index, previous_subspace_representation,
            subspace_indices, first_subspace_indices, first_nonzero_indices,
            sqrt_occupation_numbers, sqrt_first_occupation_numbers,
            interferometerc, previous_subspace_grad);
        full_kl_grad(row_index, col_index) +=
            upstream[p].einsum_ij_ij(subspace_grad.conj());
        previous_subspace_grad = subspace_grad;
      }
    }
  }
  full_kl_grad.add(upstream[1]);
  // convert result to buffer
  for (size_t i = 0; i < full_kl_grad.size(); i++) {
    result->typed_data()[i] = full_kl_grad[i];
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(_get_interferometer_on_fock_space_bwd,
                              _get_interferometer_on_fock_space_bwd_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>());

template <typename T> py::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid
  // handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}
#endif