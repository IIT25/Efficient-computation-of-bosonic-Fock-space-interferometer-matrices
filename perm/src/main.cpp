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
#include <vcruntime_typeinfo.h>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
template <typename T>
Matrix<T> numpy_to_matrix(
    pybind11::array_t<T, pybind11::array::c_style | py::array::forcecast>
        numpy_array) {
  py::buffer_info bufferinfo = numpy_array.request();
  size_t rows;
  size_t cols;
  if (bufferinfo.shape.size() > 1) {
    rows = bufferinfo.shape[0];
    cols = bufferinfo.shape[1];
  } else {
    rows = 1;
    cols = bufferinfo.shape[0];
  }

  T *data = static_cast<T *>(bufferinfo.ptr);
  Matrix<T> matrix = Matrix<T>(rows, cols, data);
  matrix.print();
  return matrix;
}
template <typename T> Matrix<T> add_to_matrix(Matrix<T> m) {
  Matrix<T> d = Matrix<T>(m);
  return d;
}
template <typename T> py::array_t<T> to_pyarray(Matrix<T> m) {
  return py::array_t<T>(py::buffer_info(m.data,
                                        sizeof(T), // itemsize
                                        py::format_descriptor<T>::format(),
                                        2,                // ndim
                                        {m.cols, m.rows}, // shape
                                        {m.rows * sizeof(T), sizeof(T)}
                                        // strides
                                        ));
}
template <typename T>
std::vector<py::array_t<T>>
calc_perm(int cutoff,
          pybind11::array_t<T, pybind11::array::c_style | py::array::forcecast>
              interferometer,
          pybind11::tuple helper) {
  // declare, init
  std::vector<Matrix<T>> matrices;
  std::vector<py::array_t<T>> array_ts;

  // conversions
  Matrix<T> intfrm = numpy_to_matrix(interferometer);
  Matrix<T> subspace_indices_matrix = numpy_to_matrix(
      helper[0]
          .cast<pybind11::array_t<T, pybind11::array::c_style |
                                         py::array::forcecast>>());
  Matrix<T> first_nonzero_indices_matrix = numpy_to_matrix(
      helper[1]
          .cast<pybind11::array_t<T, pybind11::array::c_style |
                                         py::array::forcecast>>());
  Matrix<T> first_subspace_indices_matrix = numpy_to_matrix(
      helper[2]
          .cast<pybind11::array_t<T, pybind11::array::c_style |
                                         py::array::forcecast>>());
  Matrix<T> sqrt_occupation_numbers_matrix = numpy_to_matrix(
      helper[3]
          .cast<pybind11::array_t<T, pybind11::array::c_style |
                                         py::array::forcecast>>());
  Matrix<T> sqrt_first_occupation_numbers_matrix = numpy_to_matrix(
      helper[4]
          .cast<pybind11::array_t<T, pybind11::array::c_style |
                                         py::array::forcecast>>());

  // subspace repr
  Matrix<T> first = Matrix<T>(1, 1, new T(1));
  matrices.push_back(first);
  matrices.push_back(intfrm);
  /*for (int i = 1; i < cutoff; i++) {
    matrices.push_back(add_to_matrix(matrices[i]));
  }*/

  // conversion to py::array_t
  for (Matrix<T> m : matrices) {
    array_ts.push_back(to_pyarray(m));
  }
  return array_ts;
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

  py::class_<Matrix<std::complex<double>>>(m, "Matrix", py::buffer_protocol())
      .def_buffer([](Matrix<std::complex<double>> &m) -> py::buffer_info {
        return py::buffer_info(
            m.data,                       /* Pointer to buffer */
            sizeof(std::complex<double>), /* Size of one scalar */
            py::format_descriptor<std ::complex<double>>::format(), /* Python
                                                        struct-style format
                                                        descriptor */
            2,                /* Number of dimensions */
            {m.rows, m.cols}, /* Buffer dimensions */
            {sizeof(std::complex<double>) *
                 m.cols, /* Strides (in bytes) for each index */
             sizeof(std::complex<double>)});
      });
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
