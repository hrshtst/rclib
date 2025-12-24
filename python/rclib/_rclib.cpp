#include "rclib/Model.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, etc.

namespace py = pybind11;

PYBIND11_MODULE(_rclib, m) {
  m.doc() = "pybind11 example plugin"; // Optional module docstring

  // Bind Reservoir base class
  py::class_<Reservoir, std::shared_ptr<Reservoir>>(m, "Reservoir")
      .def("advance", &Reservoir::advance)
      .def("resetState", &Reservoir::resetState)
      .def("getState", &Reservoir::getState);

  // Bind RandomSparseReservoir
  py::class_<RandomSparseReservoir, Reservoir, std::shared_ptr<RandomSparseReservoir>>(m, "RandomSparseReservoir")
      .def(py::init<int, double, double, double, double, bool, unsigned int>(), py::arg("n_neurons"),
           py::arg("spectral_radius"), py::arg("sparsity"), py::arg("leak_rate"), py::arg("input_scaling"),
           py::arg("include_bias") = false, py::arg("seed") = 42);

  // Bind NvarReservoir
  py::class_<NvarReservoir, Reservoir, std::shared_ptr<NvarReservoir>>(m, "NvarReservoir")
      .def(py::init<int>(), py::arg("num_lags"));

  // Bind Readout base class
  py::class_<Readout, std::shared_ptr<Readout>>(m, "Readout")
      .def("fit", &Readout::fit)
      .def("partialFit", &Readout::partialFit)
      .def("predict", &Readout::predict);

  // Bind RidgeReadout
  py::class_<RidgeReadout, Readout, std::shared_ptr<RidgeReadout>>(m, "RidgeReadout")
      .def(py::init<double, bool>(), py::arg("alpha"), py::arg("include_bias"));

  // Bind RlsReadout
  py::class_<RlsReadout, Readout, std::shared_ptr<RlsReadout>>(m, "RlsReadout")
      .def(py::init<double, double, bool>(), py::arg("lambda_"), py::arg("delta"), py::arg("include_bias"));

  // Bind LmsReadout
  py::class_<LmsReadout, Readout, std::shared_ptr<LmsReadout>>(m, "LmsReadout")
      .def(py::init<double, bool>(), py::arg("learning_rate"), py::arg("include_bias"));

  // Bind Model class
  py::class_<Model>(m, "Model")
      .def(py::init<>())
      .def("addReservoir", &Model::addReservoir)
      .def("setReadout", &Model::setReadout)
      .def("fit", &Model::fit, py::arg("inputs"), py::arg("targets"), py::arg("washout_len") = 0)
      .def("predict", &Model::predict, py::arg("inputs"), py::arg("reset_state_before_predict") = true)
      .def("getReservoir", &Model::getReservoir)
      .def("getReadout", &Model::getReadout)
      .def("predictOnline", &Model::predictOnline)
      .def("predictGenerative", &Model::predictGenerative, py::arg("prime_inputs"), py::arg("n_steps"))
      .def("resetReservoirs", &Model::resetReservoirs);
}
