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
#include <vcruntime_typeinfo.h>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

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
  m.def("get_fock_space_basis", &get_fock_space_basis,
        py::return_value_policy::take_ownership,
        R"pbdoc(
        Calculates the subspace representation of the matrix.

    )pbdoc");
  m.def("_get_interferometer_on_fock_space", //
        &_get_interferometer_on_fock_space,
        py::return_value_policy::take_ownership);
  m.def("calc_perm", &calc_perm, py::return_value_policy::take_ownership,
        R"pbdoc(
        Calculates the subspace representation of the matrix.

    )pbdoc");
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
