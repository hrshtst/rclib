#include "rclib/reservoirs/RandomSparseReservoir.h"

#include "RandomWeights.h"
#include "rclib/Serialization.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef RCLIB_USE_OPENMP
#  include <omp.h>
#endif

namespace {

// Stable wire codes: the file format must not depend on the enum's declaration order.
std::uint8_t methodToWire(RandomSparseReservoir::SpectralRadiusMethod method) {
  switch (method) {
    case RandomSparseReservoir::POWER_ITERATION:
      return 0;
    case RandomSparseReservoir::DENSE:
      return 1;
  }
  throw SerializationError("RandomSparseReservoir: unknown spectral_radius_method value " +
                           std::to_string(static_cast<int>(method)) + ".");
}

RandomSparseReservoir::SpectralRadiusMethod methodFromWire(std::uint8_t code) {
  switch (code) {
    case 0:
      return RandomSparseReservoir::POWER_ITERATION;
    case 1:
      return RandomSparseReservoir::DENSE;
    default:
      throw SerializationError("RandomSparseReservoir: unknown spectral_radius_method code " + std::to_string(code) +
                               ".");
  }
}

} // namespace

RandomSparseReservoir::RandomSparseReservoir(int n_neurons, double spectral_radius, double sparsity, double leak_rate,
                                             double input_scaling, bool include_bias, unsigned int seed,
                                             SpectralRadiusMethod spectral_radius_method, double bias_scaling)
    : n_neurons(n_neurons), spectral_radius(spectral_radius), sparsity(sparsity), leak_rate(leak_rate),
      input_scaling(input_scaling), include_bias(include_bias), spectral_radius_method(spectral_radius_method),
      bias_scaling(bias_scaling), W_in_initialized(false) {
  validateParameters();

  state = Eigen::MatrixXd::Zero(1, n_neurons);

  random_weights::RecurrentWeights weights =
      random_weights::generateRecurrentWeights(n_neurons, spectral_radius, sparsity, include_bias, seed,
                                               spectral_radius_method, bias_scaling, "RandomSparseReservoir");
  W_res = std::move(weights.W_res);
  bias = std::move(weights.bias);

  // W_in is drawn at the first advance(), from its own generator seeded with seed + 1.
  this->seed = seed;
}

void RandomSparseReservoir::validateParameters() const {
  if (n_neurons <= 0) {
    throw std::invalid_argument("n_neurons must be positive.");
  }
  // Range checks are written so that NaN fails them; one-sided bounds also need isfinite.
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
  if (spectral_radius_method != POWER_ITERATION && spectral_radius_method != DENSE) {
    throw std::invalid_argument("spectral_radius_method must be POWER_ITERATION or DENSE.");
  }
}

void RandomSparseReservoir::initialize_W_in(int input_dim) {
  if (input_dim <= 0) {
    throw std::invalid_argument("input_dim must be positive.");
  }
  W_in = random_weights::generateInputWeights(input_dim, n_neurons, input_scaling, seed);
  W_in_initialized = true;
}

const Eigen::MatrixXd &RandomSparseReservoir::advance(const Eigen::MatrixXd &input) {
  if (input.rows() != 1) {
    throw std::invalid_argument("RandomSparseReservoir::advance expects a single input row.");
  }
  if (!W_in_initialized) {
    initialize_W_in(input.cols());
    temp_state.resize(state.rows(), state.cols());
  } else if (input.cols() != W_in.rows()) {
    throw std::invalid_argument("input dimension changed after RandomSparseReservoir initialization.");
  }

  // 1. Initialize temp_state with (input * W_in + bias)
  // This is dense-dense operation, usually fast.
  temp_state.noalias() = input * W_in;
  temp_state += bias;

  // 2. Add recurrent contribution: state * W_res
  // We perform this manually to ensure efficiency and enable threading.
  // W_res is Column-Major. We compute output elements j (columns) independently.
  // y_j = sum_i (x_i * A_{ij})
  const double *state_ptr = state.data();
  double *temp_ptr = temp_state.data();

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
      // it.index() is the row index (i)
      // it.value() is W_{ij}
      dot += state_ptr[it.index()] * it.value();
    }
    temp_ptr[j] += dot;
  }

  // 3. Activation and Leak
  state.array() = (1.0 - leak_rate) * state.array() + leak_rate * temp_state.array().tanh();

  return state;
}

void RandomSparseReservoir::resetState() { state.setZero(); }

const Eigen::MatrixXd &RandomSparseReservoir::getState() const { return state; }

int RandomSparseReservoir::getOutputDim(int /*input_dim*/) const { return n_neurons; }

const Eigen::MatrixXd &RandomSparseReservoir::getInputWeights() const {
  if (!W_in_initialized) {
    throw std::runtime_error("RandomSparseReservoir draws W_in at the first advance; advance before getInputWeights.");
  }
  return W_in;
}

void RandomSparseReservoir::checkConsistency() const {
  translateSerializationErrors("RandomSparseReservoir", [&] {
    validateParameters();
    const auto n = static_cast<Eigen::Index>(n_neurons);
    if (W_res.rows() != n || W_res.cols() != n) {
      throw SerializationError("RandomSparseReservoir: W_res must be n_neurons x n_neurons.");
    }
    if (bias.size() != n) {
      throw SerializationError("RandomSparseReservoir: bias must have n_neurons entries.");
    }
    if (state.rows() != 1 || state.cols() != n) {
      throw SerializationError("RandomSparseReservoir: state must be 1 x n_neurons.");
    }
    if (W_in_initialized && (W_in.rows() < 1 || W_in.cols() != n)) {
      throw SerializationError("RandomSparseReservoir: W_in must have at least one row and n_neurons columns.");
    }
  });
}

void RandomSparseReservoir::save(BinaryWriter &writer) const {
  static_assert(sizeof(unsigned int) == sizeof(std::uint32_t), "the seed is stored as u32");
  translateSerializationErrors("RandomSparseReservoir", [&] {
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
    writer.writeSparse(W_res);
    writer.writeMatrix(bias);
    writer.writeMatrix(state);
    // W_in only exists once the first input has fixed its width.
    writer.writeBool(W_in_initialized);
    if (W_in_initialized) {
      writer.writeMatrix(W_in);
    }
  });
}

std::shared_ptr<RandomSparseReservoir> RandomSparseReservoir::load(BinaryReader &reader) {
  return translateSerializationErrors("RandomSparseReservoir", [&] {
    // The private constructor skips generating W_res, which would be discarded anyway.
    std::shared_ptr<RandomSparseReservoir> res(new RandomSparseReservoir());
    res->n_neurons = reader.readInt();
    res->spectral_radius = reader.readDouble();
    res->sparsity = reader.readDouble();
    res->leak_rate = reader.readDouble();
    res->input_scaling = reader.readDouble();
    res->include_bias = reader.readBool();
    res->seed = reader.readUInt();
    // Version 1 files predate the option; they were all built by power iteration.
    res->spectral_radius_method = reader.formatVersion() >= 2 ? methodFromWire(reader.readU8()) : POWER_ITERATION;
    // Files before version 3 predate the option; their bias was drawn unscaled.
    res->bias_scaling = reader.formatVersion() >= 3 ? reader.readDouble() : 1.0;
    res->W_res = reader.readSparse();
    const Eigen::MatrixXd bias = reader.readMatrix();
    if (bias.rows() != 1) { // checked before assigning into a row vector
      throw SerializationError("RandomSparseReservoir: bias must be a row vector.");
    }
    res->bias = bias;
    res->state = reader.readMatrix();
    res->W_in_initialized = reader.readBool();
    if (res->W_in_initialized) {
      res->W_in = reader.readMatrix();
    }
    res->checkConsistency();
    res->temp_state.resize(res->state.rows(), res->state.cols());
    return res;
  });
}
