#include "calc_helper.cpp"
#include "calc_perm.cpp"
#include "jax_utils.cpp"
#include "matrix.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

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

  m.def("registrations", []() {
    py::dict registrations;
    registrations["_get_interferometer_on_fock_space_xla"] =
        EncapsulateFfiCall(_get_interferometer_on_fock_space_xla);
    registrations["_get_interferometer_on_fock_space_fwd"] =
        EncapsulateFfiCall(_get_interferometer_on_fock_space_fwd);
    registrations["_get_interferometer_on_fock_space_bwd"] =
        EncapsulateFfiCall(_get_interferometer_on_fock_space_bwd);
    return registrations;
  });
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
