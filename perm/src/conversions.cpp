#ifndef CONVERSIONS
#define CONVERSIONS

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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vcruntime_typeinfo.h>
#include <vector>

namespace py = pybind11;
template <typename T>
Matrix<T>
numpy_to_matrix(py::array_t<T, pybind11::array::c_style | py::array::forcecast>
                    numpy_array) {
  py::buffer_info bufferinfo = numpy_array.request();
  size_t rows = bufferinfo.shape[0];
  size_t cols = bufferinfo.shape[1];
  T *data = static_cast<T *>(bufferinfo.ptr);
  Matrix<T> matrix = Matrix<T>(rows, cols, data);
  // matrix.print();
  return matrix;
}
template <typename T>
Matrix<T> numpy_to_matrix(py::tuple tuple, size_t n, T t) {
  py::array_t<T> numpy_array =
      tuple[n]
          .cast<py::array_t<T,
                            pybind11::array::c_style | py::array::forcecast>>();
  py::buffer_info bufferinfo = numpy_array.request();
  size_t rows = 1;
  size_t cols;
  if (bufferinfo.shape.size() == 1) {
    cols = bufferinfo.shape[0];
  } else {
    rows = bufferinfo.shape[0];
    cols = bufferinfo.shape[1];
  }
  T *data = static_cast<T *>(bufferinfo.ptr);
  Matrix<T> matrix = Matrix<T>(rows, cols, data);
  return matrix;
}
template <typename T>
Matrix<T> numpy_to_matrix1(py::tuple tuple, size_t n, T t) {
  py::array_t<T> numpy_array =
      tuple[n]
          .cast<py::array_t<T,
                            pybind11::array::c_style | py::array::forcecast>>();

  py::buffer_info bufferinfo = numpy_array.request();
  size_t rows = 1;
  size_t cols;
  if (bufferinfo.shape.size() == 1) {
    cols = bufferinfo.shape[0];
  } else {
    rows = bufferinfo.shape[0];
    cols = bufferinfo.shape[1];
  }
  T *data = static_cast<T *>(bufferinfo.ptr);
  Matrix<T> matrix = Matrix<T>(rows, cols, data);
  return matrix;
}
template <typename T> py::array_t<T> to_pyarray(Matrix<T> m) {
  return py::array_t<T>(py::buffer_info(m.data,
                                        sizeof(T), // itemsize
                                        py::format_descriptor<T>::format(),
                                        2,                // ndim
                                        {m.rows, m.cols}, // shape
                                        {m.cols * sizeof(T), sizeof(T)}
                                        // strides
                                        ));
}
#endif
