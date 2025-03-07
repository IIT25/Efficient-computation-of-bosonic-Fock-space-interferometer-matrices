#ifndef JAX_UTILS_CPP
#define JAX_UTILS_CPP

#include "calc_perm.cpp"
#include "matrix.hpp"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
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

namespace ffi = xla::ffi;
// A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
// In this example, we treat all leading dimensions as batch dimensions, so this
// function returns the total number of elements in the buffer, and the size of
// the last dimension.
template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

ffi::Error _get_interferometer_on_fock_space_Impl(
    ffi ::Buffer<ffi::U64> cutoff, ffi::Buffer<ffi::C128> interferometer,
    ffi::ResultBuffer<ffi::C128> y, ffi::ResultBuffer<ffi::U64> y_dims) {
  auto [totalSize, lastDim] = GetDims(interferometer);
  Matrix<std::complex<double>> interferometerc = Matrix<std::complex<double>>(
      totalSize / lastDim, lastDim, &(interferometer.typed_data()[0]));
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
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::U64>>());
ffi::Error _get_interferometer_on_fock_space_fwd_impl(
    ffi::Buffer<ffi::U64> cutoff, ffi::Buffer<ffi::C128> interferometer,
    ffi::ResultBuffer<ffi::C128> y, ffi::ResultBuffer<ffi::U64> y_dims,
    ffi::ResultBuffer<ffi::U64> helper_idx,
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
      helper_idx->typed_data()[current_idx] =
          first_subspace_indices_array[i][j];
      current_idx += 1;
      helper_sqrt->typed_data()[current_sqrt_idx] =
          sqrt_first_occupation_numbers_array[i][j];
      current_sqrt_idx += 1;
    }
  }
  std::cout << "helper max index on c++ side " << current_idx << std::endl;
  std::cout << "helper max sqrt index on c++ side " << current_sqrt_idx
            << std::endl;
  return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    _get_interferometer_on_fock_space_fwd,
    _get_interferometer_on_fock_space_fwd_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Ret<ffi::Buffer<ffi::C128>>() // result
        .Ret<ffi::Buffer<ffi::U64>>()  // result dimensions
        .Ret<ffi::Buffer<ffi::U64>>()  // helper_idx
        .Ret<ffi::Buffer<ffi::F64>>()  // helper_sqrt
);
ffi::Error _get_interferometer_on_fock_space_bwd_impl(
    ffi::Buffer<ffi::U64> cutoff, ffi::Buffer<ffi::C128> interferometer,
    ffi::Buffer<ffi::C128> y, ffi::Buffer<ffi::U64> y_dim,
    ffi::Buffer<ffi::U64> helper_idx, ffi::Buffer<ffi::F64> helper_sqrt,
    ffi::ResultBuffer<ffi::C128> result) {

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(_get_interferometer_on_fock_space_bwd,
                              _get_interferometer_on_fock_space_bwd_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::C128>>());

template <typename T> py::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid
  // handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}
#endif