#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, etc.
#include <sstream>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(_rclib, m) {
  m.doc() = "rclib C++ core: high-performance reservoir computing (reservoirs, "
            "readouts, and models) backed by Eigen, exposed via pybind11.";

  // Raised by every model save/load failure; a RuntimeError subclass in Python.
  py::register_exception<SerializationError>(m, "SerializationError", PyExc_RuntimeError);

  // Bind Reservoir base class
  py::class_<Reservoir, std::shared_ptr<Reservoir>>(m, "Reservoir")
      .def("advance", &Reservoir::advance)
      .def("resetState", &Reservoir::resetState)
      .def("getState", &Reservoir::getState)
      .def("getOutputDim", &Reservoir::getOutputDim)
      .def("getInputDim", &Reservoir::getInputDim);

  // Bind RandomSparseReservoir
  py::class_<RandomSparseReservoir, Reservoir, std::shared_ptr<RandomSparseReservoir>> random_sparse(
      m, "RandomSparseReservoir");

  py::enum_<RandomSparseReservoir::SpectralRadiusMethod>(random_sparse, "SpectralRadiusMethod")
      .value("POWER_ITERATION", RandomSparseReservoir::SpectralRadiusMethod::POWER_ITERATION)
      .value("DENSE", RandomSparseReservoir::SpectralRadiusMethod::DENSE)
      .export_values();

  random_sparse
      .def(py::init<int, double, double, double, double, bool, unsigned int,
                    RandomSparseReservoir::SpectralRadiusMethod>(),
           py::arg("n_neurons"), py::arg("spectral_radius"), py::arg("sparsity"), py::arg("leak_rate"),
           py::arg("input_scaling"), py::arg("include_bias") = false, py::arg("seed") = 42,
           py::arg("spectral_radius_method") = RandomSparseReservoir::SpectralRadiusMethod::POWER_ITERATION)
      .def("getNNeurons", &RandomSparseReservoir::getNNeurons)
      .def("getSpectralRadius", &RandomSparseReservoir::getSpectralRadius)
      .def("getSparsity", &RandomSparseReservoir::getSparsity)
      .def("getLeakRate", &RandomSparseReservoir::getLeakRate)
      .def("getInputScaling", &RandomSparseReservoir::getInputScaling)
      .def("getIncludeBias", &RandomSparseReservoir::getIncludeBias)
      .def("getSeed", &RandomSparseReservoir::getSeed)
      .def("getSpectralRadiusMethod", &RandomSparseReservoir::getSpectralRadiusMethod);

  // Bind NvarReservoir
  py::class_<NvarReservoir, Reservoir, std::shared_ptr<NvarReservoir>>(m, "NvarReservoir")
      .def(py::init<int, int>(), py::arg("num_lags"), py::arg("polynomial_order") = 1)
      .def("getNumLags", &NvarReservoir::getNumLags)
      .def("getPolynomialOrder", &NvarReservoir::getPolynomialOrder);

  // Bind Readout base class
  py::class_<Readout, std::shared_ptr<Readout>>(m, "Readout")
      .def("fit", &Readout::fit)
      .def("partialFit", &Readout::partialFit)
      .def("predict", &Readout::predict)
      .def("getInputDim", &Readout::getInputDim);

  // Bind RidgeReadout
  py::class_<RidgeReadout, Readout, std::shared_ptr<RidgeReadout>> ridge(m, "RidgeReadout");

  // Bind Solver Enum
  py::enum_<RidgeReadout::Solver>(ridge, "Solver")
      .value("AUTO", RidgeReadout::Solver::AUTO)
      .value("CHOLESKY", RidgeReadout::Solver::CHOLESKY)
      .value("DUAL_CHOLESKY", RidgeReadout::Solver::DUAL_CHOLESKY)
      .value("CONJUGATE_GRADIENT", RidgeReadout::Solver::CONJUGATE_GRADIENT)
      .value("CONJUGATE_GRADIENT_IMPLICIT", RidgeReadout::Solver::CONJUGATE_GRADIENT_IMPLICIT)
      .export_values();

  ridge
      .def(py::init<double, bool, RidgeReadout::Solver, double>(), py::arg("alpha"), py::arg("include_bias"),
           py::arg("solver") = RidgeReadout::Solver::AUTO, py::arg("tolerance") = 1e-10)
      .def("getAlpha", &RidgeReadout::getAlpha)
      .def("getTolerance", &RidgeReadout::getTolerance)
      .def("getSolver", &RidgeReadout::getSolver)
      .def("getEffectiveSolver", &RidgeReadout::getEffectiveSolver)
      .def("getIncludeBias", &RidgeReadout::getIncludeBias)
      .def("getWeights", &RidgeReadout::getWeights, py::return_value_policy::copy,
           "The fitted weights as a copy: shape (n_features + include_bias, n_outputs), bias row last.");

  // Bind RlsReadout
  py::class_<RlsReadout, Readout, std::shared_ptr<RlsReadout>> rls(m, "RlsReadout");

  py::enum_<RlsReadout::Solver>(rls, "Solver")
      .value("RANK1_UPDATE", RlsReadout::Solver::RANK1_UPDATE)
      .value("RANK_K_UPDATE", RlsReadout::Solver::RANK_K_UPDATE)
      .export_values();

  rls.def(py::init<double, double, bool, RlsReadout::Solver>(), py::arg("lambda_"), py::arg("delta"),
          py::arg("include_bias"), py::arg("solver") = RlsReadout::Solver::RANK1_UPDATE)
      .def("getLambda", &RlsReadout::getLambda)
      .def("getDelta", &RlsReadout::getDelta)
      .def("getIncludeBias", &RlsReadout::getIncludeBias)
      .def("getSolver", &RlsReadout::getSolver);

  // Bind LmsReadout
  py::class_<LmsReadout, Readout, std::shared_ptr<LmsReadout>>(m, "LmsReadout")
      .def(py::init<double, bool>(), py::arg("learning_rate"), py::arg("include_bias"))
      .def("getLearningRate", &LmsReadout::getLearningRate)
      .def("getIncludeBias", &LmsReadout::getIncludeBias);

  // Bind Model class
  py::class_<Model>(m, "Model")
      .def(py::init<>())
      .def("addReservoir", &Model::addReservoir)
      .def("setReadout", &Model::setReadout)
      .def("fit", &Model::fit, py::arg("inputs"), py::arg("targets"), py::arg("washout_len") = 0)
      .def("partialFit", &Model::partialFit, py::arg("input"), py::arg("target"))
      .def("predict", &Model::predict, py::arg("inputs"), py::arg("reset_state_before_predict") = true)
      .def("getReservoir", &Model::getReservoir)
      .def("getReadout", &Model::getReadout)
      .def("getNumReservoirs", &Model::getNumReservoirs)
      .def("getConnectionType", &Model::getConnectionType)
      .def("predictOnline", &Model::predictOnline)
      .def("predictGenerative", &Model::predictGenerative, py::arg("prime_inputs"), py::arg("n_steps"))
      .def("resetReservoirs", &Model::resetReservoirs)
      .def("save", py::overload_cast<const std::string &>(&Model::save, py::const_), py::arg("path"),
           "Save the model to a file, replacing any existing file atomically.")
      .def_static("load", py::overload_cast<const std::string &>(&Model::load), py::arg("path"),
                  "Load a model file written by Model.save.")
      .def(
          "dumps",
          [](const Model &model) {
            std::ostringstream buffer;
            model.save(buffer);
            return py::bytes(buffer.str());
          },
          "Serialize the model to bytes in the model file format.")
      .def_static(
          "loads",
          [](const py::bytes &data) {
            std::istringstream input{std::string(data)};
            return Model::load(input);
          },
          py::arg("data"), "Load a model from bytes produced by Model.dumps.");
}
