#include "rclib/readouts/RewardModulatedReadout.h"

#include "rclib/Serialization.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace {

// Separates this readout's noise from a reservoir's with the same rule and seed.
constexpr std::uint64_t readout_stream_salt = 0x526561646F757400ULL; // "Readout"

} // namespace

RewardModulatedReadout::RewardModulatedReadout(int n_outputs, const RewardModulatedHebbianRule &rule, bool include_bias)
    : n_outputs(n_outputs), include_bias(include_bias), learning(rule, readout_stream_salt, "RewardModulatedReadout") {
  if (n_outputs < 1) {
    throw std::invalid_argument("n_outputs must be positive.");
  }
}

void RewardModulatedReadout::fit(const Eigen::MatrixXd & /*states*/, const Eigen::MatrixXd & /*targets*/) {
  throw std::logic_error("RewardModulatedReadout learns from rewards; use act and reinforce instead of fit.");
}

void RewardModulatedReadout::partialFit(const Eigen::MatrixXd & /*state*/, const Eigen::MatrixXd & /*target*/) {
  throw std::logic_error("RewardModulatedReadout learns from rewards; use act and reinforce instead of partialFit.");
}

int RewardModulatedReadout::getInputDim() const {
  return isInitialized() ? static_cast<int>(W.rows()) - (include_bias ? 1 : 0) : 0;
}

void RewardModulatedReadout::initialize(Eigen::Index n_features) {
  W = Eigen::MatrixXd::Zero(n_features + (include_bias ? 1 : 0), n_outputs);
  learning.initialize(W.rows(), W.cols());
}

void RewardModulatedReadout::checkStates(const Eigen::MatrixXd &states) const {
  if (states.rows() == 0 || states.cols() == 0) {
    throw std::invalid_argument("states must be a non-empty 2D matrix.");
  }
  if (isInitialized() && states.cols() != getInputDim()) {
    throw std::invalid_argument("states column count does not match the RewardModulatedReadout weights.");
  }
  if (!states.allFinite()) {
    throw std::invalid_argument("states must be finite.");
  }
}

Eigen::RowVectorXd RewardModulatedReadout::withBias(const Eigen::RowVectorXd &state) const {
  if (!include_bias) {
    return state;
  }
  Eigen::RowVectorXd pre(state.size() + 1);
  pre << state, 1.0;
  return pre;
}

Eigen::MatrixXd RewardModulatedReadout::predict(const Eigen::MatrixXd &states) {
  if (!isInitialized()) {
    throw std::runtime_error("RewardModulatedReadout must act or have its weights set before predict.");
  }
  checkStates(states);
  // Row by row, the same arithmetic as a noise-free act, so the two agree bit for bit.
  Eigen::MatrixXd output(states.rows(), n_outputs);
  for (Eigen::Index t = 0; t < states.rows(); ++t) {
    output.row(t) = withBias(states.row(t)) * W;
  }
  return output;
}

Eigen::RowVectorXd RewardModulatedReadout::stage(const Eigen::RowVectorXd &state, const Eigen::RowVectorXd &noise,
                                                 bool explore) {
  learning.discardRow();
  const Eigen::RowVectorXd pre = withBias(state);
  Eigen::RowVectorXd output = pre * W;
  if (explore) {
    output += noise;
  }
  if (!output.allFinite()) {
    throw std::runtime_error("RewardModulatedReadout: the output is not finite.");
  }
  if (explore) {
    learning.stageRow(pre, output, noise);
  }
  return output;
}

Eigen::RowVectorXd RewardModulatedReadout::stageAct(const Eigen::RowVectorXd &state, bool explore) {
  checkStates(state);
  if (!isInitialized()) {
    initialize(state.size());
  }
  // Noise-free steps leave the generator alone.
  const Eigen::RowVectorXd noise = explore ? learning.sampleNoise() : Eigen::RowVectorXd::Zero(n_outputs);
  return stage(state, noise, explore);
}

Eigen::RowVectorXd RewardModulatedReadout::stageAct(const Eigen::RowVectorXd &state, const Eigen::RowVectorXd &noise) {
  checkStates(state);
  if (noise.size() != n_outputs || !noise.allFinite()) {
    throw std::invalid_argument("noise must be finite with one entry per output.");
  }
  if (!isInitialized()) {
    initialize(state.size());
  }
  return stage(state, noise, true);
}

Eigen::MatrixXd RewardModulatedReadout::act(const Eigen::MatrixXd &states, bool explore) {
  checkStates(states);
  Eigen::MatrixXd outputs(states.rows(), n_outputs);
  for (Eigen::Index t = 0; t < states.rows(); ++t) {
    outputs.row(t) = stageAct(states.row(t), explore);
    if (explore) {
      learning.commitRow();
    }
  }
  return outputs;
}

Eigen::MatrixXd RewardModulatedReadout::act(const Eigen::MatrixXd &states, const Eigen::MatrixXd &noise) {
  checkStates(states);
  if (noise.rows() != states.rows() || noise.cols() != n_outputs || !noise.allFinite()) {
    throw std::invalid_argument("noise must be finite with one row per state and one column per output.");
  }
  Eigen::MatrixXd outputs(states.rows(), n_outputs);
  for (Eigen::Index t = 0; t < states.rows(); ++t) {
    outputs.row(t) = stageAct(states.row(t), noise.row(t));
    learning.commitRow();
  }
  return outputs;
}

const Eigen::MatrixXd &RewardModulatedReadout::getWeights() const {
  if (!isInitialized()) {
    throw std::runtime_error("RewardModulatedReadout has no weights before its first act or setWeights.");
  }
  return W;
}

void RewardModulatedReadout::setWeights(const Eigen::MatrixXd &weights) {
  const Eigen::Index min_rows = 1 + (include_bias ? 1 : 0);
  if (isInitialized() ? (weights.rows() != W.rows() || weights.cols() != W.cols())
                      : (weights.rows() < min_rows || weights.cols() != n_outputs)) {
    throw std::invalid_argument("weights must have shape (n_features" + std::string(include_bias ? " + 1" : "") +
                                ", n_outputs), matching the existing weights once they exist.");
  }
  if (!weights.allFinite()) {
    throw std::invalid_argument("weights must be finite.");
  }
  if (!isInitialized()) {
    initialize(weights.rows() - (include_bias ? 1 : 0));
  }
  W = weights;
}

void RewardModulatedReadout::prepareWeights() {
  prepared_weights.reset();
  const Eigen::MatrixXd *delta = learning.preparedDelta();
  if (delta == nullptr) {
    return;
  }
  Eigen::MatrixXd candidate = W + *delta;
  if (!candidate.allFinite()) {
    learning.discardUpdate();
    throw std::runtime_error("RewardModulatedReadout: the new weights are not finite.");
  }
  prepared_weights = std::move(candidate);
}

void RewardModulatedReadout::prepareReinforce(double reward, int key, std::optional<double> baseline) {
  learning.prepareReinforce(reward, key, baseline);
  prepareWeights();
}

void RewardModulatedReadout::prepareModulate(double modulator) {
  learning.prepareModulate(modulator);
  prepareWeights();
}

void RewardModulatedReadout::commitUpdate() {
  learning.commitUpdate(); // throws, changing nothing, if no update is prepared
  if (prepared_weights) {
    W = std::move(*prepared_weights);
    prepared_weights.reset();
  }
}

void RewardModulatedReadout::discardUpdate() {
  prepared_weights.reset();
  learning.discardUpdate();
}

void RewardModulatedReadout::checkConsistency() const {
  translateSerializationErrors("RewardModulatedReadout", [&] {
    if (n_outputs < 1) {
      throw SerializationError("RewardModulatedReadout: n_outputs must be positive.");
    }
    learning.checkConsistency();
    if (isInitialized() && (W.rows() < 1 + (include_bias ? 1 : 0) || W.cols() != n_outputs ||
                            learning.getNPre() != W.rows() || learning.getNPost() != W.cols())) {
      throw SerializationError("RewardModulatedReadout: the weights do not match the outputs or the learning state.");
    }
  });
}

void RewardModulatedReadout::save(BinaryWriter &writer) const {
  translateSerializationErrors("RewardModulatedReadout", [&] {
    checkConsistency();
    writer.writeInt(n_outputs);
    writer.writeBool(include_bias);
    learning.save(writer); // the rule, then the learning state
    if (isInitialized()) {
      writer.writeMatrix(W);
    }
  });
}

std::shared_ptr<RewardModulatedReadout> RewardModulatedReadout::load(BinaryReader &reader) {
  return translateSerializationErrors("RewardModulatedReadout", [&] {
    const int n_outputs = reader.readInt();
    const bool include_bias = reader.readBool();
    RewardModulatedHebbian learning =
        RewardModulatedHebbian::load(reader, readout_stream_salt, "RewardModulatedReadout");
    auto readout = std::make_shared<RewardModulatedReadout>(n_outputs, learning.getRule(), include_bias); // validates
    if (learning.isInitialized()) {
      readout->W = reader.readMatrix();
    }
    readout->learning = std::move(learning);
    readout->checkConsistency();
    return readout;
  });
}
