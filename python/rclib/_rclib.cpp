#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/learning/RewardModulatedHebbian.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RewardModulatedReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <optional>
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
                    RandomSparseReservoir::SpectralRadiusMethod, double>(),
           py::arg("n_neurons"), py::arg("spectral_radius"), py::arg("sparsity"), py::arg("leak_rate"),
           py::arg("input_scaling"), py::arg("include_bias") = false, py::arg("seed") = 42,
           py::arg("spectral_radius_method") = RandomSparseReservoir::SpectralRadiusMethod::POWER_ITERATION,
           py::arg("bias_scaling") = 1.0)
      .def("getNNeurons", &RandomSparseReservoir::getNNeurons)
      .def("getSpectralRadius", &RandomSparseReservoir::getSpectralRadius)
      .def("getSparsity", &RandomSparseReservoir::getSparsity)
      .def("getLeakRate", &RandomSparseReservoir::getLeakRate)
      .def("getInputScaling", &RandomSparseReservoir::getInputScaling)
      .def("getIncludeBias", &RandomSparseReservoir::getIncludeBias)
      .def("getSeed", &RandomSparseReservoir::getSeed)
      .def("getSpectralRadiusMethod", &RandomSparseReservoir::getSpectralRadiusMethod)
      .def("getBiasScaling", &RandomSparseReservoir::getBiasScaling)
      // The nonzero entries as numpy arrays, so that the core needs no scipy: W[rows, cols] = values.
      .def(
          "getRecurrentWeights",
          [](const RandomSparseReservoir &res) {
            const Eigen::SparseMatrix<double> &weights = res.getRecurrentWeights();
            Eigen::VectorXi rows(weights.nonZeros());
            Eigen::VectorXi cols(weights.nonZeros());
            Eigen::VectorXd values(weights.nonZeros());
            Eigen::Index k = 0;
            for (Eigen::Index j = 0; j < weights.outerSize(); ++j) {
              for (Eigen::SparseMatrix<double>::InnerIterator it(weights, j); it; ++it, ++k) {
                rows(k) = static_cast<int>(it.row());
                cols(k) = static_cast<int>(it.col());
                values(k) = it.value();
              }
            }
            return py::make_tuple(rows, cols, values);
          },
          "The recurrent weights W_res as (rows, cols, values), copies of the nonzero entries: W[rows, cols] = "
          "values gives the (n_neurons, n_neurons) matrix, applied as state @ W.")
      .def("getInputWeights", &RandomSparseReservoir::getInputWeights, py::return_value_policy::copy,
           "The input weights W_in as a copy, (input_dim, n_neurons), applied as input @ W_in; raises before the "
           "first advance.")
      .def("getBias", &RandomSparseReservoir::getBias, py::return_value_policy::copy,
           "The bias as a copy, (n_neurons,); zero without include_bias.");

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

  // Reward-modulated Hebbian learning: the rule, the learning state and the readout.
  using Rule = RewardModulatedHebbianRule;
  py::class_<Rule> rule(m, "RewardModulatedHebbianRule",
                        "Configuration of a reward-modulated Hebbian rule; see rclib.learning.");
  py::enum_<Rule::Distribution>(rule, "Distribution").value("UNIFORM", Rule::UNIFORM).value("GAUSSIAN", Rule::GAUSSIAN);
  py::enum_<Rule::PostSignal>(rule, "PostSignal")
      .value("NOISE", Rule::NOISE)
      .value("HIGH_PASS", Rule::HIGH_PASS)
      .value("POTENTIAL", Rule::POTENTIAL);
  py::enum_<Rule::Update>(rule, "Update").value("ONLINE", Rule::ONLINE).value("TRIAL", Rule::TRIAL);
  py::enum_<Rule::Modulator>(rule, "Modulator")
      .value("DIFFERENCE", Rule::DIFFERENCE)
      .value("HEAVISIDE", Rule::HEAVISIDE);
  py::enum_<Rule::Baseline>(rule, "Baseline")
      .value("EMA", Rule::EMA)
      .value("WINDOW", Rule::WINDOW)
      .value("NONE", Rule::NONE);
  rule.def(py::init<>())
      .def_readwrite("learning_rate", &Rule::learning_rate)
      .def_readwrite("noise_distribution", &Rule::noise_distribution)
      .def_readwrite("noise_amplitude", &Rule::noise_amplitude)
      .def_readwrite("noise_probability", &Rule::noise_probability)
      .def_readwrite("noise_seed", &Rule::noise_seed)
      .def_readwrite("post", &Rule::post)
      .def_readwrite("post_decay", &Rule::post_decay)
      .def_readwrite("eligibility_power", &Rule::eligibility_power)
      .def_readwrite("update", &Rule::update)
      .def_readwrite("trace_decay", &Rule::trace_decay)
      .def_readwrite("modulator", &Rule::modulator)
      .def_readwrite("baseline", &Rule::baseline)
      .def_readwrite("baseline_decay", &Rule::baseline_decay)
      .def_readwrite("baseline_window", &Rule::baseline_window)
      .def_readwrite("learning_rate_decay_steps", &Rule::learning_rate_decay_steps)
      .def_readwrite("max_update", &Rule::max_update)
      .def_readwrite("decorrelation", &Rule::decorrelation)
      .def("validate", &Rule::validate);

  // The learning state is read-only from Python; matrices are returned as copies.
  py::class_<RewardModulatedHebbian>(m, "RewardModulatedHebbian")
      .def("getRule", &RewardModulatedHebbian::getRule, py::return_value_policy::copy)
      .def("getEligibility", &RewardModulatedHebbian::getEligibility, py::return_value_policy::copy)
      .def("getCovariance", &RewardModulatedHebbian::getCovariance, py::return_value_policy::copy)
      .def("getEligibilityBound", &RewardModulatedHebbian::getEligibilityBound)
      .def("getCovarianceBound", &RewardModulatedHebbian::getCovarianceBound)
      .def("getPostAverage", &RewardModulatedHebbian::getPostAverage, py::return_value_policy::copy)
      .def("isPostAverageInitialized", &RewardModulatedHebbian::isPostAverageInitialized)
      .def("hasEligibility", &RewardModulatedHebbian::hasEligibility)
      .def("getUpdateCount", &RewardModulatedHebbian::getUpdateCount)
      .def("getNoiseState", &RewardModulatedHebbian::getNoiseState)
      .def("getBaselines", &RewardModulatedHebbian::getBaselines,
           "The reward history of each key: the EMA value, or the window oldest first.")
      .def("getBaseline", &RewardModulatedHebbian::getBaseline, py::arg("key"),
           "The baseline the next reward of `key` is compared with; None before its first reward.")
      .def("getLearningRate", &RewardModulatedHebbian::getLearningRate, "The learning rate the next update uses.");

  py::class_<RewardModulatedReadout, Readout, std::shared_ptr<RewardModulatedReadout>>(m, "RewardModulatedReadout")
      .def(py::init<int, const Rule &, bool>(), py::arg("n_outputs"), py::arg("rule"), py::arg("include_bias") = true)
      .def("act", py::overload_cast<const Eigen::MatrixXd &, bool>(&RewardModulatedReadout::act), py::arg("states"),
           py::arg("explore") = true, "The output of each row; exploring adds noise and records the eligibility.")
      .def("actWithNoise",
           py::overload_cast<const Eigen::MatrixXd &, const Eigen::MatrixXd &>(&RewardModulatedReadout::act),
           py::arg("states"), py::arg("noise"), "Explores with the given noise instead of drawing it.")
      .def(
          "reinforce",
          [](RewardModulatedReadout &readout, double reward, int key, std::optional<double> baseline) {
            readout.reinforce(reward, key, baseline);
          },
          py::arg("reward"), py::arg("key") = 0, py::arg("baseline") = py::none())
      .def(
          "modulate", [](RewardModulatedReadout &readout, double modulator) { readout.modulate(modulator); },
          py::arg("modulator"))
      .def("clearEligibility", &RewardModulatedReadout::clearEligibility)
      .def("resetActivity", &RewardModulatedReadout::resetActivity)
      .def("getWeights", &RewardModulatedReadout::getWeights, py::return_value_policy::copy,
           "The weights as a copy: shape (n_features + include_bias, n_outputs), bias row last.")
      .def("setWeights", &RewardModulatedReadout::setWeights, py::arg("weights"))
      .def("getNOutputs", &RewardModulatedReadout::getNOutputs)
      .def("getIncludeBias", &RewardModulatedReadout::getIncludeBias)
      .def("getRule", &RewardModulatedReadout::getRule, py::return_value_policy::copy)
      .def("getLearning", &RewardModulatedReadout::getLearning, py::return_value_policy::reference_internal,
           "The learning state, valid while the readout lives.");

  // Bind Model class
  py::class_<Model>(m, "Model")
      .def(py::init<>())
      .def("addReservoir", &Model::addReservoir)
      .def("setReadout", &Model::setReadout)
      .def("fit", &Model::fit, py::arg("inputs"), py::arg("targets"), py::arg("washout_len") = 0)
      .def("fitSequences", &Model::fitSequences, py::arg("inputs"), py::arg("targets"), py::arg("washout_len") = 0)
      .def("partialFit", &Model::partialFit, py::arg("input"), py::arg("target"))
      .def("predict", &Model::predict, py::arg("inputs"), py::arg("reset_state_before_predict") = true)
      .def("getReservoir", &Model::getReservoir)
      .def("getReadout", &Model::getReadout)
      .def("getNumReservoirs", &Model::getNumReservoirs)
      .def("getConnectionType", &Model::getConnectionType)
      .def("predictOnline", &Model::predictOnline)
      .def("predictGenerative", &Model::predictGenerative, py::arg("prime_inputs"), py::arg("n_steps"))
      .def("resetReservoirs", &Model::resetReservoirs)
      .def("act", &Model::act, py::arg("inputs"), py::arg("explore") = true)
      .def("reinforce", &Model::reinforce, py::arg("reward"), py::arg("key") = 0, py::arg("baseline") = py::none())
      .def("modulate", &Model::modulate, py::arg("modulator"))
      .def("clearEligibility", &Model::clearEligibility)
      .def("hasReadout", &Model::hasReadout)
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
