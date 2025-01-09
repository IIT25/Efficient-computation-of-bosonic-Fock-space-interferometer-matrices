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

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

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
  /*std::cout << "\n";
  for (size_t dim : bufferinfo.shape) {

    std::cout << dim << "  ";
  }
  std::cout << "\n";*/
  size_t rows = 1;
  size_t cols;
  if (bufferinfo.shape.size() == 1) {
    cols = bufferinfo.shape[0];
  } else {
    rows = bufferinfo.shape[0];
    cols = bufferinfo.shape[1];
  }
  T *data = static_cast<T *>(bufferinfo.ptr);
  std::cout << *data;
  Matrix<T> matrix = Matrix<T>(rows, cols, data);
  return matrix;
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
calc_perm(py::array_t<T, pybind11::array::c_style | py::array::forcecast>
              interferometer,
          pybind11::tuple helper_indices) {
  // declare, init
  py::tuple subspace_indices_array = helper_indices[0];
  py::tuple first_nonzero_indices_array = helper_indices[1];
  py::tuple first_subspace_indices_array = helper_indices[2];
  py::tuple sqrt_occupation_numbers_array = helper_indices[3];
  py::tuple sqrt_first_occupation_numbers_array = helper_indices[4];

  const size_t cutoff = subspace_indices_array.size() + 2;
  std::vector<Matrix<T>> subspace_representations = std::vector<Matrix<T>>();
  // std::cout << cutoff << "\n";
  Matrix<T> first = Matrix<T>(1, 1, new T(1));
  Matrix<T> intfrm = numpy_to_matrix(interferometer);
  subspace_representations.push_back(first);
  subspace_representations.push_back(intfrm);

  std::vector<py::array_t<T>> array_ts = std::vector<py::array_t<T>>();
  for (size_t n = 0; n < cutoff - 2; n++) {
    Matrix<int> subspace_indices =
        numpy_to_matrix1(subspace_indices_array, n, int(0));
    Matrix<int> first_subspace_indices =
        numpy_to_matrix(first_subspace_indices_array, n, int(0));
    Matrix<int> first_nonzero_indices =
        numpy_to_matrix(first_nonzero_indices_array, n, int(0));
    Matrix<double> sqrt_occupation_numbers =
        numpy_to_matrix(sqrt_occupation_numbers_array, n, double(0));
    Matrix<double> sqrt_first_occupation_numbers =
        numpy_to_matrix(sqrt_first_occupation_numbers_array, n, double(0));

    Matrix<T> previous_representation = subspace_representations[n + 1];
    Matrix<T> representation =
        Matrix<T>(first_nonzero_indices.cols, sqrt_occupation_numbers.rows);

    for (size_t k = 0; k < first_nonzero_indices.cols; k++) {
      T denominator = sqrt_first_occupation_numbers[k];
      Matrix<T> previous_representation_indexed =
          previous_representation.rowidx(first_subspace_indices[k]);
      for (size_t j = 0; j < sqrt_occupation_numbers.cols; j++) {
        T one_particle_contrib =
            intfrm(first_nonzero_indices[k], j) / denominator;
        for (size_t i = 0; i < sqrt_occupation_numbers.rows; i++) {
          /*std::cout << subspace_indices(i, j) << "   i: " << i << " j: " << j
                    << "  k: " << k << "  n: " << n << "\n";*/
          representation(k, i) +=
              one_particle_contrib * sqrt_occupation_numbers(i, j) *
              previous_representation_indexed[subspace_indices(i, j)];
        }
      }
    }
    subspace_representations.push_back(representation);
  }

  // conversion to py::array_t
  for (Matrix<T> m : subspace_representations) {
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
        py::return_value_policy::take_ownership,
        R"pbdoc(
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
