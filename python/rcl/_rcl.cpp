#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h> // For std::vector, etc.

#include "rcl/Model.h"
#include "rcl/reservoirs/RandomSparseReservoir.h"
#include "rcl/reservoirs/NvarReservoir.h"
#include "rcl/readouts/RidgeReadout.h"
#include "rcl/readouts/RlsReadout.h"
#include "rcl/readouts/LmsReadout.h"

namespace py = pybind11;

PYBIND11_MODULE(_rcl, m) {
    m.doc() = "pybind11 example plugin"; // Optional module docstring

    // Bind Reservoir base class
    py::class_<Reservoir, std::shared_ptr<Reservoir>>(m, "Reservoir")
        .def("advance", &Reservoir::advance)
        .def("resetState", &Reservoir::resetState)
        .def("getState", &Reservoir::getState);

    // Bind RandomSparseReservoir
    py::class_<RandomSparseReservoir, Reservoir, std::shared_ptr<RandomSparseReservoir>>(m, "RandomSparseReservoir")
        .def(py::init<int, double, double, double, double, bool>(),
             py::arg("n_neurons"), py::arg("spectral_radius"), py::arg("sparsity"),
             py::arg("leak_rate"), py::arg("input_scaling"), py::arg("include_bias"));

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
        .def(py::init<double, bool>(),
             py::arg("alpha"), py::arg("include_bias"));

    // Bind RlsReadout
    py::class_<RlsReadout, Readout, std::shared_ptr<RlsReadout>>(m, "RlsReadout")
        .def(py::init<double, double, bool>(),
             py::arg("lambda_"), py::arg("delta"), py::arg("include_bias"));

    // Bind LmsReadout
    py::class_<LmsReadout, Readout, std::shared_ptr<LmsReadout>>(m, "LmsReadout")
        .def(py::init<double, bool>(),
             py::arg("learning_rate"), py::arg("include_bias"));

    // Bind Model class
    py::class_<Model>(m, "Model")
        .def(py::init<>())
        .def("addReservoir", &Model::addReservoir)
        .def("setReadout", &Model::setReadout)
        .def("fit", &Model::fit, py::arg("inputs"), py::arg("targets"), py::arg("washout_len") = 0)
        .def("predict", &Model::predict, py::arg("inputs"), py::arg("reset_state_before_predict") = true)
        .def("getReservoir", &Model::getReservoir) // Added
        .def("getReadout", &Model::getReadout)    // Added
        .def("predictOnline", &Model::predictOnline)
        .def("predictGenerative", &Model::predictGenerative, py::arg("prime_inputs"), py::arg("n_steps"));
}
