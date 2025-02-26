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

ffi::Error
_get_interferometer_on_fock_space_Impl(ffi ::Buffer<ffi::U64> cutoff,
                                       ffi::Buffer<ffi::C128> interferometer,
                                       ffi::ResultBuffer<ffi::C128> y) {
  auto [totalSize, lastDim] = GetDims(interferometer);
  Matrix<std::complex<double>> interferometerc = Matrix<std::complex<double>>(
      totalSize / lastDim, lastDim, &(interferometer.typed_data()[0]));
  Matrix<std::complex<double>> res =
      Matrix<std::complex<double>>(1, 1, new std::complex<double>(1));
  std::vector<int> dims = _get_interferometer_on_fock_space_nc(
      interferometerc, cutoff.typed_data()[0], res);
  ;
  for (size_t i = 0; i < res.size(); i++) {
    y->typed_data()[i] = res[i];
  }
  /*for (int i = res.size(); i < cutoff.typed_data()[0] + res.size(); i++) {
    y->typed_data()[i] = dims[i];
  }*/
  /*for (size_t i = 0; i < res.size() + cutoff.typed_data()[0]; i++) {
    std::cout << y->typed_data()[i] << std::endl;
  }*/
  std::cout << "reeee" << std::endl;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(_get_interferometer_on_fock_space_xla,
                              _get_interferometer_on_fock_space_Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>());
ffi::Error _get_interferometer_on_fock_space_fwd_impl(
    ffi::Buffer<ffi::U64> cutoff, ffi::Buffer<ffi::C128> interferometer,
    ffi::ResultBuffer<ffi::C128> y, ffi::ResultBuffer<ffi::C128> result) {
  return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(_get_interferometer_on_fock_space_fwd,
                              _get_interferometer_on_fock_space_fwd_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>());
ffi::Error _get_interferometer_on_fock_space_bwd_impl(
    ffi::Buffer<ffi::U64> cutoff, ffi::Buffer<ffi::C128> interferometer,
    ffi::ResultBuffer<ffi::C128> ct_y, ffi::ResultBuffer<ffi::C128> ct_x,
    ffi::ResultBuffer<ffi::C128> result) {

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(_get_interferometer_on_fock_space_bwd,
                              _get_interferometer_on_fock_space_bwd_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>());

template <typename T> py::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid
  // handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}
#endif