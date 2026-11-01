#include "rclib/learning/RewardModulatedHebbian.h"

#include "rclib/Serialization.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

constexpr double pi = 3.14159265358979323846;
/// Largest reward window, so that a malformed or huge setting cannot exhaust memory.
constexpr int max_baseline_window = 1000000;

bool inUnitInterval(double value, bool include_one) {
  return value >= 0.0 && (include_one ? value <= 1.0 : value < 1.0);
}

void requirePositive(const std::optional<double> &value, const char *name) {
  if (value && !(std::isfinite(*value) && *value > 0.0)) {
    throw std::invalid_argument(std::string(name) + " must be finite and positive when set.");
  }
}

double maxAbs(const Eigen::RowVectorXd &row) { return row.size() == 0 ? 0.0 : row.cwiseAbs().maxCoeff(); }

// Stable wire codes: the file format must not depend on the enums' declaration order.
using Rule = RewardModulatedHebbianRule;

std::uint8_t toWire(Rule::Distribution value) { return value == Rule::UNIFORM ? 0 : 1; }
std::uint8_t toWire(Rule::PostSignal value) { return value == Rule::NOISE ? 0 : value == Rule::HIGH_PASS ? 1 : 2; }
std::uint8_t toWire(Rule::Update value) { return value == Rule::ONLINE ? 0 : 1; }
std::uint8_t toWire(Rule::Modulator value) { return value == Rule::DIFFERENCE ? 0 : 1; }
std::uint8_t toWire(Rule::Baseline value) { return value == Rule::EMA ? 0 : value == Rule::WINDOW ? 1 : 2; }

template <typename Enum> Enum fromWire(std::uint8_t code, std::initializer_list<Enum> values, const char *name) {
  if (code >= values.size()) {
    throw SerializationError(std::string("unknown ") + name + " code " + std::to_string(code) + ".");
  }
  return *(values.begin() + code);
}

void writeOptional(BinaryWriter &writer, const std::optional<double> &value) {
  writer.writeBool(value.has_value());
  if (value) {
    writer.writeDouble(*value);
  }
}

std::optional<double> readOptional(BinaryReader &reader) {
  if (!reader.readBool()) {
    return std::nullopt;
  }
  return reader.readDouble();
}

void writeRule(BinaryWriter &writer, const Rule &rule) {
  writer.writeDouble(rule.learning_rate);
  writer.writeU8(toWire(rule.noise_distribution));
  writer.writeDouble(rule.noise_amplitude);
  writer.writeDouble(rule.noise_probability);
  writer.writeUInt(rule.noise_seed);
  writer.writeU8(toWire(rule.post));
  writer.writeDouble(rule.post_decay);
  writer.writeDouble(rule.eligibility_power);
  writer.writeU8(toWire(rule.update));
  writer.writeDouble(rule.trace_decay);
  writer.writeU8(toWire(rule.modulator));
  writer.writeU8(toWire(rule.baseline));
  writer.writeDouble(rule.baseline_decay);
  writer.writeInt(rule.baseline_window);
  writeOptional(writer, rule.learning_rate_decay_steps);
  writeOptional(writer, rule.max_update);
  writeOptional(writer, rule.decorrelation);
}

Rule readRule(BinaryReader &reader) {
  Rule rule;
  rule.learning_rate = reader.readDouble();
  rule.noise_distribution = fromWire(reader.readU8(), {Rule::UNIFORM, Rule::GAUSSIAN}, "noise_distribution");
  rule.noise_amplitude = reader.readDouble();
  rule.noise_probability = reader.readDouble();
  rule.noise_seed = reader.readUInt();
  rule.post = fromWire(reader.readU8(), {Rule::NOISE, Rule::HIGH_PASS, Rule::POTENTIAL}, "post");
  rule.post_decay = reader.readDouble();
  rule.eligibility_power = reader.readDouble();
  rule.update = fromWire(reader.readU8(), {Rule::ONLINE, Rule::TRIAL}, "update");
  rule.trace_decay = reader.readDouble();
  rule.modulator = fromWire(reader.readU8(), {Rule::DIFFERENCE, Rule::HEAVISIDE}, "modulator");
  rule.baseline = fromWire(reader.readU8(), {Rule::EMA, Rule::WINDOW, Rule::NONE}, "baseline");
  rule.baseline_decay = reader.readDouble();
  rule.baseline_window = reader.readInt();
  rule.learning_rate_decay_steps = readOptional(reader);
  rule.max_update = readOptional(reader);
  rule.decorrelation = readOptional(reader);
  return rule;
}

double maxAbs(const Eigen::MatrixXd &matrix) { return matrix.size() == 0 ? 0.0 : matrix.cwiseAbs().maxCoeff(); }

} // namespace

void RewardModulatedHebbianRule::validate() const {
  // Range checks are written so that NaN fails them; one-sided bounds also need isfinite.
  if (!(std::isfinite(learning_rate) && learning_rate > 0.0)) {
    throw std::invalid_argument("learning_rate must be finite and positive.");
  }
  if (noise_distribution != UNIFORM && noise_distribution != GAUSSIAN) {
    throw std::invalid_argument("noise_distribution must be UNIFORM or GAUSSIAN.");
  }
  if (!(std::isfinite(noise_amplitude) && noise_amplitude >= 0.0)) {
    throw std::invalid_argument("noise_amplitude must be finite and non-negative.");
  }
  if (!(noise_probability > 0.0 && noise_probability <= 1.0)) {
    throw std::invalid_argument("noise_probability must be in (0, 1].");
  }
  if (post != NOISE && post != HIGH_PASS && post != POTENTIAL) {
    throw std::invalid_argument("post must be NOISE, HIGH_PASS or POTENTIAL.");
  }
  if (!inUnitInterval(post_decay, false)) {
    throw std::invalid_argument("post_decay must be in [0, 1).");
  }
  if (!(std::isfinite(eligibility_power) && eligibility_power > 0.0)) {
    throw std::invalid_argument("eligibility_power must be finite and positive.");
  }
  if (update != ONLINE && update != TRIAL) {
    throw std::invalid_argument("update must be ONLINE or TRIAL.");
  }
  if (!inUnitInterval(trace_decay, true)) {
    throw std::invalid_argument("trace_decay must be in [0, 1].");
  }
  if (update == TRIAL && trace_decay != 0.0) {
    throw std::invalid_argument("trace_decay applies to ONLINE updates only; TRIAL eligibility never decays.");
  }
  if (modulator != DIFFERENCE && modulator != HEAVISIDE) {
    throw std::invalid_argument("modulator must be DIFFERENCE or HEAVISIDE.");
  }
  if (baseline != EMA && baseline != WINDOW && baseline != NONE) {
    throw std::invalid_argument("baseline must be EMA, WINDOW or NONE.");
  }
  if (!inUnitInterval(baseline_decay, false)) {
    throw std::invalid_argument("baseline_decay must be in [0, 1).");
  }
  if (baseline_window < 1 || baseline_window > max_baseline_window) {
    throw std::invalid_argument("baseline_window must be in [1, " + std::to_string(max_baseline_window) + "].");
  }
  requirePositive(learning_rate_decay_steps, "learning_rate_decay_steps");
  requirePositive(max_update, "max_update");
  requirePositive(decorrelation, "decorrelation");
  if (decorrelation && update != TRIAL) {
    throw std::invalid_argument("decorrelation requires TRIAL updates.");
  }
  if (decorrelation && eligibility_power != 1.0) {
    throw std::invalid_argument("decorrelation requires an eligibility_power of 1.");
  }
}

std::uint64_t SplitMix64::next() {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

double SplitMix64::uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

RewardModulatedHebbian::RewardModulatedHebbian(const RewardModulatedHebbianRule &rule, std::uint64_t stream_salt,
                                               std::string context)
    : rule(rule), context(std::move(context)) {
  rule.validate();
  // Mix the seed first, so that nearby seeds and salts start far apart in the sequence.
  SplitMix64 seeder(rule.noise_seed);
  rng.setState(seeder.next() ^ stream_salt);
}

void RewardModulatedHebbian::initialize(Eigen::Index n_pre, Eigen::Index n_post) {
  if (initialized) {
    throw std::logic_error(context + ": the learning state is already initialized.");
  }
  if (n_pre < 1 || n_post < 1) {
    throw std::invalid_argument(context + ": the weights need at least one row and one column.");
  }
  eligibility = Eigen::MatrixXd::Zero(n_pre, n_post);
  if (rule.decorrelation) {
    covariance = Eigen::MatrixXd::Zero(n_pre, n_pre);
  }
  post_average = Eigen::RowVectorXd::Zero(n_post);
  initialized = true;
}

Eigen::RowVectorXd RewardModulatedHebbian::sampleNoise() {
  if (!initialized) {
    throw std::logic_error(context + ": the learning state is not initialized.");
  }
  Eigen::RowVectorXd noise(getNPost());
  for (Eigen::Index i = 0; i < noise.size(); ++i) {
    const bool perturbed = rng.uniform() < rule.noise_probability;
    double value = 0.0;
    if (rule.noise_distribution == RewardModulatedHebbianRule::UNIFORM) {
      value = rule.noise_amplitude * (2.0 * rng.uniform() - 1.0);
    } else {
      // Box-Muller without a cached second value; 1 - u1 is in (0, 1], so the log is finite.
      const double u1 = rng.uniform();
      const double u2 = rng.uniform();
      value = rule.noise_amplitude * std::sqrt(-2.0 * std::log(1.0 - u1)) * std::cos(2.0 * pi * u2);
    }
    noise(i) = perturbed ? value : 0.0;
  }
  return noise;
}

double RewardModulatedHebbian::eligibilityFunction(double y, double power) {
  return power == 1.0 ? y : std::copysign(std::pow(std::abs(y), power), y);
}

void RewardModulatedHebbian::fail(const std::string &what) const { throw std::runtime_error(context + ": " + what); }

void RewardModulatedHebbian::stageRow(const Eigen::RowVectorXd &pre, const Eigen::RowVectorXd &potential,
                                      const Eigen::RowVectorXd &noise) {
  staged_row.reset();
  if (!initialized) {
    throw std::logic_error(context + ": the learning state is not initialized.");
  }
  if (pre.size() != getNPre() || potential.size() != getNPost() || noise.size() != getNPost()) {
    throw std::invalid_argument(context + ": the row does not match the shape of the weights.");
  }
  if (!pre.allFinite()) {
    fail("the pre-synaptic activity is not finite.");
  }
  if (!potential.allFinite()) {
    fail("the potential is not finite.");
  }

  StagedRow row;
  Eigen::RowVectorXd post;
  switch (rule.post) {
    case RewardModulatedHebbianRule::NOISE:
      post = noise;
      row.post_average = post_average;
      break;
    case RewardModulatedHebbianRule::POTENTIAL:
      post = potential;
      row.post_average = post_average;
      break;
    case RewardModulatedHebbianRule::HIGH_PASS: {
      // The average before this row; an uninitialized average starts at this row's potential.
      const Eigen::RowVectorXd &previous = filter_initialized ? post_average : potential;
      post = potential - previous;
      row.post_average = rule.post_decay * previous + (1.0 - rule.post_decay) * potential;
      break;
    }
  }
  if (!post.allFinite() || !row.post_average.allFinite()) {
    fail("the post-synaptic signal is not finite.");
  }

  const double power = rule.eligibility_power;
  row.s_pre = pre.unaryExpr([power](double y) { return eligibilityFunction(y, power); });
  row.s_post = post.unaryExpr([power](double y) { return eligibilityFunction(y, power); });
  if (!row.s_pre.allFinite() || !row.s_post.allFinite()) {
    fail("the eligibility terms are not finite.");
  }
  // Every entry of S(pre)^T S(post) is at most term_bound in magnitude, so the new eligibility
  // stays finite when its bound does. TRIAL eligibility accumulates without decay.
  const double term_bound = maxAbs(row.s_pre) * maxAbs(row.s_post);
  const double decay = rule.update == RewardModulatedHebbianRule::ONLINE ? rule.trace_decay : 1.0;
  row.eligibility_bound = decay * eligibility_bound + term_bound;
  if (!std::isfinite(row.eligibility_bound)) {
    fail("the eligibility would overflow.");
  }
  row.covariance_bound = covariance_bound;
  if (rule.decorrelation) {
    const double pre_max = maxAbs(pre);
    row.covariance_bound = covariance_bound + pre_max * pre_max;
    if (!std::isfinite(row.covariance_bound)) {
      fail("the pre-synaptic covariance would overflow.");
    }
  }
  row.pre = pre;
  staged_row = std::move(row);
}

void RewardModulatedHebbian::commitRow() {
  if (!staged_row) {
    throw std::logic_error(context + ": no row is staged.");
  }
  StagedRow &row = *staged_row;
  if (rule.update == RewardModulatedHebbianRule::ONLINE && rule.trace_decay == 0.0) {
    eligibility.noalias() = row.s_pre.transpose() * row.s_post;
  } else {
    if (rule.update == RewardModulatedHebbianRule::ONLINE) {
      eligibility *= rule.trace_decay;
    }
    eligibility.noalias() += row.s_pre.transpose() * row.s_post;
  }
  if (rule.decorrelation) {
    covariance.noalias() += row.pre.transpose() * row.pre;
  }
  eligibility_bound = row.eligibility_bound;
  covariance_bound = row.covariance_bound;
  if (rule.post == RewardModulatedHebbianRule::HIGH_PASS) {
    post_average = std::move(row.post_average);
    filter_initialized = true;
  }
  has_eligibility = true;
  staged_row.reset();
}

void RewardModulatedHebbian::checkReady() const {
  if (!has_eligibility) {
    throw std::logic_error(context + ": reinforce and modulate need an exploring act since the last update.");
  }
}

std::optional<double> RewardModulatedHebbian::getBaseline(int key) const {
  if (rule.baseline == RewardModulatedHebbianRule::NONE) {
    return 0.0;
  }
  const auto found = baselines.find(key);
  if (found == baselines.end() || found->second.empty()) {
    return std::nullopt;
  }
  const std::vector<double> &history = found->second;
  if (rule.baseline == RewardModulatedHebbianRule::EMA) {
    return history.front();
  }
  // Divide before adding, so the mean of finite rewards cannot overflow.
  double mean = 0.0;
  const double count = static_cast<double>(history.size());
  for (double reward : history) {
    mean += reward / count;
  }
  return mean;
}

double RewardModulatedHebbian::getLearningRate() const {
  if (!rule.learning_rate_decay_steps) {
    return rule.learning_rate;
  }
  return rule.learning_rate / (1.0 + static_cast<double>(update_count) / *rule.learning_rate_decay_steps);
}

void RewardModulatedHebbian::prepareReinforce(double reward, int key, std::optional<double> baseline) {
  prepared_update.reset();
  if (!std::isfinite(reward)) {
    throw std::invalid_argument("reward must be finite.");
  }
  if (baseline && !std::isfinite(*baseline)) {
    throw std::invalid_argument("baseline must be finite.");
  }
  checkReady();

  // The baseline before this reward; the first reward of a key with an internal baseline
  // becomes its own baseline, so its modulator is zero.
  double expected = 0.0;
  std::optional<int> updated_key;
  std::vector<double> history;
  if (baseline) {
    expected = *baseline; // the internal baseline of the key is neither read nor updated
  } else if (rule.baseline != RewardModulatedHebbianRule::NONE) {
    expected = getBaseline(key).value_or(reward);
    updated_key = key;
    const auto found = baselines.find(key);
    if (rule.baseline == RewardModulatedHebbianRule::EMA) {
      const double previous = found == baselines.end() ? reward : found->second.front();
      history = {rule.baseline_decay * previous + (1.0 - rule.baseline_decay) * reward};
    } else {
      if (found != baselines.end()) {
        history = found->second;
      }
      history.push_back(reward);
      if (history.size() > static_cast<std::size_t>(rule.baseline_window)) {
        history.erase(history.begin());
      }
    }
  }

  double modulator = 0.0;
  if (rule.modulator == RewardModulatedHebbianRule::DIFFERENCE) {
    modulator = reward - expected;
    if (!std::isfinite(modulator)) {
      fail("the difference between the reward and its baseline overflows.");
    }
  } else {
    modulator = reward > expected ? 1.0 : 0.0;
  }
  prepare(modulator, updated_key, std::move(history));
}

void RewardModulatedHebbian::prepareModulate(double modulator) {
  prepared_update.reset();
  if (!std::isfinite(modulator)) {
    throw std::invalid_argument("modulator must be finite.");
  }
  checkReady();
  prepare(modulator, std::nullopt, {});
}

void RewardModulatedHebbian::prepare(double modulator, std::optional<int> key, std::vector<double> history) {
  PreparedUpdate update;
  update.key = key;
  update.baseline_history = std::move(history);
  if (modulator != 0.0) {
    Eigen::MatrixXd direction;
    if (rule.decorrelation) {
      const Eigen::Index n = covariance.rows();
      const Eigen::LDLT<Eigen::MatrixXd> solver(covariance + *rule.decorrelation * Eigen::MatrixXd::Identity(n, n));
      if (solver.info() != Eigen::Success) {
        fail("the decorrelation solve failed.");
      }
      direction = solver.solve(eligibility);
      if (!direction.allFinite()) {
        fail("the decorrelation solve failed.");
      }
    } else {
      direction = eligibility;
    }
    Eigen::MatrixXd delta = (getLearningRate() * modulator) * direction;
    // Overflow is a failure even where clipping would hide it.
    if (!delta.allFinite()) {
      fail("the weight change is not finite.");
    }
    if (rule.max_update) {
      delta = delta.cwiseMax(-*rule.max_update).cwiseMin(*rule.max_update);
    }
    update.delta = std::move(delta);
  }
  prepared_update = std::move(update);
}

const Eigen::MatrixXd *RewardModulatedHebbian::preparedDelta() const {
  if (!prepared_update) {
    throw std::logic_error(context + ": no update is prepared.");
  }
  return prepared_update->delta ? &*prepared_update->delta : nullptr;
}

void RewardModulatedHebbian::commitUpdate() {
  if (!prepared_update) {
    throw std::logic_error(context + ": no update is prepared.");
  }
  if (prepared_update->key) {
    baselines[*prepared_update->key] = std::move(prepared_update->baseline_history);
  }
  ++update_count;
  if (rule.update == RewardModulatedHebbianRule::TRIAL) {
    eligibility.setZero();
    covariance.setZero();
    eligibility_bound = 0.0;
    covariance_bound = 0.0;
  }
  has_eligibility = false;
  prepared_update.reset();
}

void RewardModulatedHebbian::clearEligibility() {
  eligibility.setZero();
  covariance.setZero();
  eligibility_bound = 0.0;
  covariance_bound = 0.0;
  has_eligibility = false;
}

void RewardModulatedHebbian::checkConsistency() const {
  translateSerializationErrors(context.c_str(), [&] {
    rule.validate();
    const auto fail_consistency = [&](const std::string &what) { throw SerializationError(context + ": " + what); };
    if (!initialized) {
      if (eligibility.size() != 0 || covariance.size() != 0 || post_average.size() != 0 || has_eligibility ||
          filter_initialized || eligibility_bound != 0.0 || covariance_bound != 0.0) {
        fail_consistency("an uninitialized learning state holds activity.");
      }
    } else {
      const Eigen::Index n_pre = eligibility.rows();
      const Eigen::Index n_post = eligibility.cols();
      if (n_pre < 1 || n_post < 1) {
        fail_consistency("the eligibility needs at least one row and one column.");
      }
      const Eigen::Index covariance_size = rule.decorrelation ? n_pre : 0;
      if (covariance.rows() != covariance_size || covariance.cols() != covariance_size) {
        fail_consistency("the covariance does not match the eligibility and the rule.");
      }
      if (post_average.size() != n_post) {
        fail_consistency("the post-synaptic average does not match the eligibility.");
      }
      // The bounds guard against overflow, so the matrices they bound must stay within them.
      if (!eligibility.allFinite() || !(std::isfinite(eligibility_bound) && eligibility_bound >= maxAbs(eligibility))) {
        fail_consistency("the eligibility is not finite or exceeds its bound.");
      }
      if (!covariance.allFinite() || !(std::isfinite(covariance_bound) && covariance_bound >= maxAbs(covariance))) {
        fail_consistency("the covariance is not finite or exceeds its bound.");
      }
      if (!post_average.allFinite()) {
        fail_consistency("the post-synaptic average is not finite.");
      }
    }
    for (const auto &[key, history] : baselines) {
      const bool fits = rule.baseline == Rule::EMA ? history.size() == 1
                        : rule.baseline == Rule::WINDOW
                            ? !history.empty() && history.size() <= static_cast<std::size_t>(rule.baseline_window)
                            : false;
      if (!fits) {
        fail_consistency("the baseline of key " + std::to_string(key) + " does not fit the rule.");
      }
      for (double reward : history) {
        if (!std::isfinite(reward)) {
          fail_consistency("the baseline of key " + std::to_string(key) + " is not finite.");
        }
      }
    }
  });
}

void RewardModulatedHebbian::save(BinaryWriter &writer) const {
  translateSerializationErrors(context.c_str(), [&] {
    checkConsistency();
    writeRule(writer, rule);
    writer.writeU64(rng.getState());
    writer.writeU64(update_count);
    writer.writeBool(initialized);
    if (initialized) {
      writer.writeMatrix(eligibility);
      writer.writeMatrix(covariance);
      writer.writeDouble(eligibility_bound);
      writer.writeDouble(covariance_bound);
      writer.writeBool(filter_initialized);
      writer.writeMatrix(post_average);
      writer.writeBool(has_eligibility);
    }
    writer.writeUInt(static_cast<std::uint32_t>(baselines.size()));
    for (const auto &[key, history] : baselines) {
      writer.writeInt(key);
      writer.writeUInt(static_cast<std::uint32_t>(history.size()));
      for (double reward : history) {
        writer.writeDouble(reward);
      }
    }
  });
}

RewardModulatedHebbian RewardModulatedHebbian::load(BinaryReader &reader, std::uint64_t stream_salt,
                                                    std::string context) {
  const std::string name = context;
  return translateSerializationErrors(name.c_str(), [&] {
    RewardModulatedHebbian engine(readRule(reader), stream_salt, std::move(context)); // validates the rule
    engine.rng.setState(reader.readU64());
    engine.update_count = reader.readU64();
    if (reader.readBool()) {
      engine.eligibility = reader.readMatrix();
      engine.covariance = reader.readMatrix();
      engine.eligibility_bound = reader.readDouble();
      engine.covariance_bound = reader.readDouble();
      engine.filter_initialized = reader.readBool();
      const Eigen::MatrixXd average = reader.readMatrix();
      if (average.rows() != 1) {
        throw SerializationError(name + ": the post-synaptic average must be one row.");
      }
      engine.post_average = average.row(0);
      engine.has_eligibility = reader.readBool();
      engine.initialized = true;
    }
    // Each key consumes input, so a bogus count fails at the end of the data.
    const std::uint32_t n_keys = reader.readUInt();
    for (std::uint32_t k = 0; k < n_keys; ++k) {
      const int key = reader.readInt();
      if (engine.baselines.count(key) != 0) {
        throw SerializationError(name + ": the baseline of key " + std::to_string(key) + " appears twice.");
      }
      const std::uint32_t count = reader.readUInt();
      if (count > static_cast<std::uint32_t>(engine.rule.baseline_window) || count * 8ULL > reader.remainingBytes()) {
        throw SerializationError(name + ": the baseline of key " + std::to_string(key) + " is too long.");
      }
      std::vector<double> &history = engine.baselines[key];
      for (std::uint32_t i = 0; i < count; ++i) {
        history.push_back(reader.readDouble());
      }
    }
    engine.checkConsistency();
    return engine;
  });
}
