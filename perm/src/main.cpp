#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <ostream>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vcruntime_typeinfo.h>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
template <typename T>
std::vector<std::vector<T>> numpy_to_matrix(
    pybind11::array_t<T, pybind11::array::c_style | py::array::forcecast>
        numpy_array) {
  py::buffer_info bufferinfo = numpy_array.request();

  size_t rows = bufferinfo.shape[0];
  size_t cols = bufferinfo.shape[1];
  std::vector<std::vector<T>> matrix;
  T *data = static_cast<T *>(bufferinfo.ptr);
  for (int i = 0; i < rows; i++) {
    matrix.push_back(std::vector<T>());
    for (int j = 0; j < cols; j++) {
      matrix[i].push_back(*data);
      data++;
    }
  }

  return matrix;
}
template <typename T>
std::vector<std::vector<T>> add_to_matrix(std::vector<std::vector<T>> m) {
  m.push_back(m.back());
  T sum_of_elems;
  for (std::vector<T> &row : m) {
    sum_of_elems = 0;
    for (auto &n : row)
      sum_of_elems += n;
    row.push_back(sum_of_elems);
  }
  return m;
}
template <typename T>
std::vector<std::vector<std::vector<T>>>
calc_perm(int cutoff,
          pybind11::array_t<T, pybind11::array::c_style | py::array::forcecast>
              interferometer) {
  std::vector<std::vector<std::vector<T>>> matrices;
  std::vector<T> first;
  first.push_back(T(1));
  matrices.push_back({first});
  std::vector<std::vector<T>> intfrm = numpy_to_matrix(interferometer);
  matrices.push_back(intfrm);
  for (int i = 1; i < cutoff; i++) {
    matrices.push_back(add_to_matrix(matrices[i]));
  }
  return matrices;
}

PYBIND11_MODULE(_core, m) {

  m.doc() = R"pbdoc(
        Permanent calculator
        -----------------------

        .. currentmodule:: perm

        .. autosummary::
           :toctree: _generate

           calc_perm
    )pbdoc";
  m.def("calc_perm", &calc_perm<std::complex<double>>,
        py::return_value_policy::take_ownership,
        R"pbdoc(
        Calculates the subspace representation of the matrix.

    )pbdoc");
  m.def("calc_perm", &calc_perm<double>,
        py::return_value_policy::take_ownership, R"pbdoc(
        Calculates the subspace representation of the matrix.

    )pbdoc");

  m.def("calc_perm", &calc_perm<float>, py::return_value_policy::take_ownership,
        R"pbdoc(
        Calculates the subspace representation of the matrix.

    )pbdoc");

  m.def("calc_perm", &calc_perm<int>, py::return_value_policy::take_ownership,
        R"pbdoc(
        Calculates the subspace representation of the matrix.

  )pbdoc");
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
