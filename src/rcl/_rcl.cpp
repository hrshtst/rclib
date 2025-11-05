#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "rcl/Model.h"
#include "rcl/reservoirs/RandomSparseReservoir.h"
#include "rcl/reservoirs/NvarReservoir.h"
#include "rcl/readouts/RidgeReadout.h"
#include "rcl/readouts/RlsReadout.h"
#include "rcl/readouts/LmsReadout.h"

namespace py = pybind11;

PYBIND11_MODULE(_rcl, m) {
    m.doc() = "Reservoir Computing Library";

    py::class_<Model, std::shared_ptr<Model>>(m, "Model")
        .def(py::init<>())
        .def("addReservoir", &Model::addReservoir, py::arg("res"), py::arg("connection_type") = "serial")
        .def("setReadout", &Model::setReadout)
        .def("fit", &Model::fit)
        .def("predict", &Model::predict)
        .def("predictOnline", &Model::predictOnline);

    py::class_<Reservoir, std::shared_ptr<Reservoir>>(m, "Reservoir");

    py::class_<RandomSparseReservoir, Reservoir, std::shared_ptr<RandomSparseReservoir>>(m, "RandomSparseReservoir")
        .def(py::init<int, double, double, double, bool>(),
             py::arg("n_neurons"),
             py::arg("spectral_radius"),
             py::arg("sparsity"),
             py::arg("leak_rate"),
             py::arg("include_bias") = false);

    py::class_<NvarReservoir, Reservoir, std::shared_ptr<NvarReservoir>>(m, "NvarReservoir")
        .def(py::init<int>(), py::arg("num_lags"));


    py::class_<Readout, std::shared_ptr<Readout>>(m, "Readout");

    py::class_<RidgeReadout, Readout, std::shared_ptr<RidgeReadout>>(m, "RidgeReadout")
        .def(py::init<double, bool>(),
             py::arg("alpha") = 1e-8,
             py::arg("include_bias") = true);

    py::class_<RlsReadout, Readout, std::shared_ptr<RlsReadout>>(m, "RlsReadout")
        .def(py::init<double, double, bool>(),
             py::arg("lambda") = 0.99,
             py::arg("delta") = 1.0,
             py::arg("include_bias") = true);

    py::class_<LmsReadout, Readout, std::shared_ptr<LmsReadout>>(m, "LmsReadout")
        .def(py::init<double, bool>(),
             py::arg("learning_rate") = 0.01,
             py::arg("include_bias") = true);
}