#include "rclib/reservoirs/RewardModulatedReservoir.h"

#include "RandomWeights.h"
#include "rclib/Serialization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef RCLIB_USE_OPENMP
#  include <omp.h>
#endif

namespace {

// Separates this reservoir's noise from a readout's with the same rule and seed, and from that
// of reservoirs with other weight seeds.
constexpr std::uint64_t reservoir_stream_salt = 0x5265736572766F69ULL; // "Reservoi"

// Stable wire codes, the same as RandomSparseReservoir's.
std::uint8_t methodToWire(RandomSparseReservoir::SpectralRadiusMethod method) {
  return method == RandomSparseReservoir::DENSE ? 1 : 0;
}

RandomSparseReservoir::SpectralRadiusMethod methodFromWire(std::uint8_t code) {
  switch (code) {
    case 0:
      return RandomSparseReservoir::POWER_ITERATION;
    case 1:
      return RandomSparseReservoir::DENSE;
    default:
      throw SerializationError("RewardModulatedReservoir: unknown spectral_radius_method code " + std::to_string(code) +
                               ".");
  }
}

void writeMask(BinaryWriter &writer, const std::optional<std::vector<int>> &mask) {
  writer.writeBool(mask.has_value());
  if (mask) {
    writer.writeUInt(static_cast<std::uint32_t>(mask->size()));
    for (int index : *mask) {
      writer.writeInt(index);
    }
  }
}

std::optional<std::vector<int>> readMask(BinaryReader &reader) {
  if (!reader.readBool()) {
    return std::nullopt;
  }
  const std::uint32_t count = reader.readUInt();
  if (count * 4ULL > reader.remainingBytes()) {
    throw SerializationError("RewardModulatedReservoir: a plastic neuron list is longer than the data.");
  }
  std::vector<int> mask;
  mask.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    mask.push_back(reader.readInt());
  }
  return mask;
}

std::optional<std::vector<int>> normalizeMask(std::optional<std::vector<int>> mask, int n_neurons, const char *name) {
  if (!mask) {
    return mask;
  }
  std::sort(mask->begin(), mask->end());
  if (std::adjacent_find(mask->begin(), mask->end()) != mask->end()) {
    throw std::invalid_argument(std::string(name) + " must list distinct neurons.");
  }
  if (!mask->empty() && (mask->front() < 0 || mask->back() >= n_neurons)) {
    throw std::invalid_argument(std::string(name) + " must hold neuron indices in [0, n_neurons).");
  }
  return mask;
}

std::vector<bool> maskFlags(const std::optional<std::vector<int>> &mask, int n_neurons) {
  std::vector<bool> flags(static_cast<std::size_t>(n_neurons), !mask);
  if (mask) {
    for (int index : *mask) {
      flags[static_cast<std::size_t>(index)] = true;
    }
  }
  return flags;
}

} // namespace

std::uint64_t RewardModulatedReservoir::streamSalt(unsigned int seed) {
  return reservoir_stream_salt ^ (static_cast<std::uint64_t>(seed) << 32);
}

RewardModulatedReservoir::RewardModulatedReservoir(NoWeights, int n_neurons, double spectral_radius, double sparsity,
                                                   double leak_rate, double input_scaling, bool include_bias,
                                                   unsigned int seed,
                                                   RandomSparseReservoir::SpectralRadiusMethod spectral_radius_method,
                                                   double bias_scaling, RewardModulatedHebbian learning,
                                                   std::optional<std::vector<int>> plastic_presynaptic,
                                                   std::optional<std::vector<int>> plastic_postsynaptic)
    : n_neurons(n_neurons), spectral_radius(spectral_radius), sparsity(sparsity), leak_rate(leak_rate),
      input_scaling(input_scaling), include_bias(include_bias), seed(seed),
      spectral_radius_method(spectral_radius_method), bias_scaling(bias_scaling),
      plastic_presynaptic(std::move(plastic_presynaptic)), plastic_postsynaptic(std::move(plastic_postsynaptic)),
      learning(std::move(learning)) {
  validateParameters();
  this->plastic_presynaptic = normalizeMask(std::move(this->plastic_presynaptic), n_neurons, "plastic_presynaptic");
  this->plastic_postsynaptic = normalizeMask(std::move(this->plastic_postsynaptic), n_neurons, "plastic_postsynaptic");
  state = Eigen::MatrixXd::Zero(1, n_neurons);
  potential = Eigen::MatrixXd::Zero(1, n_neurons);
  drive = Eigen::MatrixXd::Zero(1, n_neurons);
}

RewardModulatedReservoir::RewardModulatedReservoir(int n_neurons, double spectral_radius, double sparsity,
                                                   double leak_rate, double input_scaling, bool include_bias,
                                                   unsigned int seed,
                                                   RandomSparseReservoir::SpectralRadiusMethod spectral_radius_method,
                                                   double bias_scaling, const RewardModulatedHebbianRule &rule,
                                                   std::optional<std::vector<int>> plastic_presynaptic,
                                                   std::optional<std::vector<int>> plastic_postsynaptic)
    : RewardModulatedReservoir(NoWeights{}, n_neurons, spectral_radius, sparsity, leak_rate, input_scaling,
                               include_bias, seed, spectral_radius_method, bias_scaling,
                               RewardModulatedHebbian(rule, streamSalt(seed), "RewardModulatedReservoir"),
                               std::move(plastic_presynaptic), std::move(plastic_postsynaptic)) {
  random_weights::RecurrentWeights weights =
      random_weights::generateRecurrentWeights(n_neurons, spectral_radius, sparsity, include_bias, seed,
                                               spectral_radius_method, bias_scaling, "RewardModulatedReservoir");
  W_res = std::move(weights.W_res);
  bias = std::move(weights.bias);
  learning.initialize(n_neurons, n_neurons);
}

void RewardModulatedReservoir::validateParameters() const {
  // The checks and messages of RandomSparseReservoir, whose hyperparameters these are.
  if (n_neurons <= 0) {
    throw std::invalid_argument("n_neurons must be positive.");
  }
  if (!std::isfinite(spectral_radius) || spectral_radius < 0.0) {
    throw std::invalid_argument("spectral_radius must be finite and non-negative.");
  }
  if (!(sparsity >= 0.0 && sparsity <= 1.0)) {
    throw std::invalid_argument("sparsity must be in [0, 1].");
  }
  if (!(leak_rate > 0.0 && leak_rate <= 1.0)) {
    throw std::invalid_argument("leak_rate must be in (0, 1].");
  }
  if (!std::isfinite(input_scaling) || input_scaling < 0.0) {
    throw std::invalid_argument("input_scaling must be finite and non-negative.");
  }
  if (!std::isfinite(bias_scaling) || bias_scaling < 0.0) {
    throw std::invalid_argument("bias_scaling must be finite and non-negative.");
  }
  if (spectral_radius_method != RandomSparseReservoir::POWER_ITERATION &&
      spectral_radius_method != RandomSparseReservoir::DENSE) {
    throw std::invalid_argument("spectral_radius_method must be POWER_ITERATION or DENSE.");
  }
}

const Eigen::MatrixXd &RewardModulatedReservoir::getInputWeights() const {
  if (!W_in_initialized) {
    throw std::runtime_error("RewardModulatedReservoir draws W_in at the first step; step before getInputWeights.");
  }
  return W_in;
}

void RewardModulatedReservoir::step(const Eigen::MatrixXd &input, const Eigen::RowVectorXd *noise) {
  if (input.rows() != 1) {
    throw std::invalid_argument("RewardModulatedReservoir expects a single input row.");
  }
  if (!W_in_initialized) {
    W_in = random_weights::generateInputWeights(static_cast<int>(input.cols()), n_neurons, input_scaling, seed);
    W_in_initialized = true;
  } else if (input.cols() != W_in.rows()) {
    throw std::invalid_argument("input dimension changed after RewardModulatedReservoir initialization.");
  }

  // The drive is computed exactly as RandomSparseReservoir computes its pre-activation.
  drive.noalias() = input * W_in;
  drive += bias;
  const double *state_ptr = state.data();
  double *drive_ptr = drive.data();
#ifdef RCLIB_USE_OPENMP
#  ifdef RCLIB_ADAPTIVE_PARALLELIZATION
#    pragma omp parallel for if (!omp_in_parallel() && n_neurons > 1000)
#  else
#    pragma omp parallel for if (!omp_in_parallel())
#  endif
#endif
  for (int j = 0; j < n_neurons; ++j) {
    double dot = 0.0;
    for (Eigen::SparseMatrix<double>::InnerIterator it(W_res, j); it; ++it) {
      dot += state_ptr[it.index()] * it.value();
    }
    drive_ptr[j] += dot;
  }

  const Eigen::RowVectorXd previous_rate = noise != nullptr ? Eigen::RowVectorXd(state.row(0)) : Eigen::RowVectorXd();
  // Column-major 1 x n_neurons like the drive and the state, so that the expressions, and tanh
  // in particular, are evaluated as in RandomSparseReservoir, vectorized.
  potential.array() = (1.0 - leak_rate) * potential.array() + leak_rate * drive.array();
  if (noise != nullptr) {
    potential.row(0) += *noise; // outside the leak, so a kick stays in full in the potential
  }
  if (!potential.allFinite()) {
    throw std::runtime_error("RewardModulatedReservoir: the potential is not finite.");
  }
  state.array() = potential.array().tanh();
  if (noise != nullptr) {
    learning.stageRow(previous_rate, potential.row(0), *noise);
  }
}

const Eigen::MatrixXd &RewardModulatedReservoir::advance(const Eigen::MatrixXd &input) {
  step(input, nullptr);
  return state;
}

const Eigen::MatrixXd &RewardModulatedReservoir::stageAct(const Eigen::MatrixXd &input, bool explore) {
  learning.discardRow();
  if (!explore) {
    step(input, nullptr);
    return state;
  }
  // Noise-free steps leave the generator alone.
  const Eigen::RowVectorXd noise = learning.sampleNoise();
  step(input, &noise);
  return state;
}

void RewardModulatedReservoir::checkInputs(const Eigen::MatrixXd &inputs) const {
  if (inputs.rows() == 0 || inputs.cols() == 0) {
    throw std::invalid_argument("inputs must be a non-empty 2D matrix.");
  }
  if (W_in_initialized && inputs.cols() != W_in.rows()) {
    throw std::invalid_argument("input dimension changed after RewardModulatedReservoir initialization.");
  }
  if (!inputs.allFinite()) {
    throw std::invalid_argument("inputs must be finite.");
  }
}

Eigen::MatrixXd RewardModulatedReservoir::act(const Eigen::MatrixXd &inputs, bool explore) {
  checkInputs(inputs);
  Eigen::MatrixXd rates(inputs.rows(), n_neurons);
  for (Eigen::Index t = 0; t < inputs.rows(); ++t) {
    rates.row(t) = stageAct(inputs.row(t), explore);
    if (explore) {
      learning.commitRow();
    }
  }
  return rates;
}

Eigen::MatrixXd RewardModulatedReservoir::act(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &noise) {
  checkInputs(inputs);
  if (noise.rows() != inputs.rows() || noise.cols() != n_neurons || !noise.allFinite()) {
    throw std::invalid_argument("noise must be finite with one row per input and one column per neuron.");
  }
  Eigen::MatrixXd rates(inputs.rows(), n_neurons);
  for (Eigen::Index t = 0; t < inputs.rows(); ++t) {
    learning.discardRow();
    const Eigen::RowVectorXd row_noise = noise.row(t);
    step(inputs.row(t), &row_noise);
    learning.commitRow();
    rates.row(t) = state.row(0);
  }
  return rates;
}

void RewardModulatedReservoir::resetState() {
  state.setZero();
  potential.setZero();
  learning.resetActivity();
}

void RewardModulatedReservoir::prepareWeights() {
  prepared_values.reset();
  const Eigen::MatrixXd *delta = learning.preparedDelta();
  if (delta == nullptr) {
    return;
  }
  const std::vector<bool> presynaptic = maskFlags(plastic_presynaptic, n_neurons);
  const std::vector<bool> postsynaptic = maskFlags(plastic_postsynaptic, n_neurons);
  Eigen::VectorXd values = Eigen::Map<const Eigen::VectorXd>(W_res.valuePtr(), W_res.nonZeros());
  const auto *outer = W_res.outerIndexPtr();
  const auto *inner = W_res.innerIndexPtr();
  // Only existing synapses from a plastic pre-synaptic to a plastic post-synaptic neuron change.
  for (int j = 0; j < n_neurons; ++j) {
    if (!postsynaptic[static_cast<std::size_t>(j)]) {
      continue;
    }
    for (auto k = outer[j]; k < outer[j + 1]; ++k) {
      const auto i = inner[k];
      if (presynaptic[static_cast<std::size_t>(i)]) {
        values(k) += (*delta)(i, j);
      }
    }
  }
  if (!values.allFinite()) {
    learning.discardUpdate();
    throw std::runtime_error("RewardModulatedReservoir: the new recurrent weights are not finite.");
  }
  prepared_values = std::move(values);
}

void RewardModulatedReservoir::prepareReinforce(double reward, int key, std::optional<double> baseline) {
  learning.prepareReinforce(reward, key, baseline);
  prepareWeights();
}

void RewardModulatedReservoir::prepareModulate(double modulator) {
  learning.prepareModulate(modulator);
  prepareWeights();
}

void RewardModulatedReservoir::commitUpdate() {
  learning.commitUpdate(); // throws, changing nothing, if no update is prepared
  if (prepared_values) {
    Eigen::Map<Eigen::VectorXd>(W_res.valuePtr(), W_res.nonZeros()) = *prepared_values;
    prepared_values.reset();
  }
}

void RewardModulatedReservoir::discardUpdate() {
  prepared_values.reset();
  learning.discardUpdate();
}

void RewardModulatedReservoir::checkConsistency() const {
  translateSerializationErrors("RewardModulatedReservoir", [&] {
    validateParameters();
    for (const auto *mask : {&plastic_presynaptic, &plastic_postsynaptic}) {
      if (*mask && normalizeMask(*mask, n_neurons, "a plastic neuron list") != *mask) {
        throw SerializationError("RewardModulatedReservoir: a plastic neuron list is not sorted.");
      }
    }
    const auto n = static_cast<Eigen::Index>(n_neurons);
    if (W_res.rows() != n || W_res.cols() != n) {
      throw SerializationError("RewardModulatedReservoir: W_res must be n_neurons x n_neurons.");
    }
    if (bias.size() != n || state.rows() != 1 || state.cols() != n || potential.rows() != 1 || potential.cols() != n) {
      throw SerializationError("RewardModulatedReservoir: bias, state and potential must have n_neurons entries.");
    }
    if (W_in_initialized && (W_in.rows() < 1 || W_in.cols() != n)) {
      throw SerializationError("RewardModulatedReservoir: W_in must have at least one row and n_neurons columns.");
    }
    learning.checkConsistency();
    if (!learning.isInitialized() || learning.getNPre() != n || learning.getNPost() != n) {
      throw SerializationError("RewardModulatedReservoir: the learning state does not match the neurons.");
    }
  });
}

void RewardModulatedReservoir::save(BinaryWriter &writer) const {
  translateSerializationErrors("RewardModulatedReservoir", [&] {
    checkConsistency();
    writer.writeInt(n_neurons);
    writer.writeDouble(spectral_radius);
    writer.writeDouble(sparsity);
    writer.writeDouble(leak_rate);
    writer.writeDouble(input_scaling);
    writer.writeBool(include_bias);
    writer.writeUInt(seed);
    writer.writeU8(methodToWire(spectral_radius_method));
    writer.writeDouble(bias_scaling);
    writeMask(writer, plastic_presynaptic);
    writeMask(writer, plastic_postsynaptic);
    writer.writeSparse(W_res);
    writer.writeMatrix(bias);
    writer.writeMatrix(state);
    writer.writeMatrix(potential);
    writer.writeBool(W_in_initialized);
    if (W_in_initialized) {
      writer.writeMatrix(W_in);
    }
    learning.save(writer);
  });
}

std::shared_ptr<RewardModulatedReservoir> RewardModulatedReservoir::load(BinaryReader &reader) {
  return translateSerializationErrors("RewardModulatedReservoir", [&] {
    const int n_neurons = reader.readInt();
    const double spectral_radius = reader.readDouble();
    const double sparsity = reader.readDouble();
    const double leak_rate = reader.readDouble();
    const double input_scaling = reader.readDouble();
    const bool include_bias = reader.readBool();
    const unsigned int seed = reader.readUInt();
    const RandomSparseReservoir::SpectralRadiusMethod method = methodFromWire(reader.readU8());
    const double bias_scaling = reader.readDouble();
    std::optional<std::vector<int>> presynaptic = readMask(reader);
    std::optional<std::vector<int>> postsynaptic = readMask(reader);
    Eigen::SparseMatrix<double> W_res = reader.readSparse();
    const Eigen::MatrixXd bias = reader.readMatrix();
    Eigen::MatrixXd state = reader.readMatrix();
    Eigen::MatrixXd potential = reader.readMatrix();
    const bool W_in_initialized = reader.readBool();
    Eigen::MatrixXd W_in;
    if (W_in_initialized) {
      W_in = reader.readMatrix();
    }
    RewardModulatedHebbian learning =
        RewardModulatedHebbian::load(reader, streamSalt(seed), "RewardModulatedReservoir");
    if (bias.rows() != 1) {
      throw SerializationError("RewardModulatedReservoir: the bias must be one row.");
    }
    // The private constructor validates the hyperparameters and masks and draws nothing.
    std::shared_ptr<RewardModulatedReservoir> reservoir(new RewardModulatedReservoir(
        NoWeights{}, n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, include_bias, seed, method,
        bias_scaling, std::move(learning), std::move(presynaptic), std::move(postsynaptic)));
    reservoir->W_res = std::move(W_res);
    reservoir->W_res.makeCompressed();
    reservoir->bias = bias.row(0);
    reservoir->state = std::move(state);
    reservoir->potential = std::move(potential);
    reservoir->W_in = std::move(W_in);
    reservoir->W_in_initialized = W_in_initialized;
    reservoir->checkConsistency();
    return reservoir;
  });
}
