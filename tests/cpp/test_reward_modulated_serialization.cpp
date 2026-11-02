#include "rclib/Model.h"
#include "rclib/Serialization.h"
#include "rclib/readouts/RewardModulatedReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/reservoirs/NvarReservoir.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"
#include "rclib/reservoirs/RewardModulatedReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;

namespace {

using Rule = RewardModulatedHebbianRule;
using Method = RandomSparseReservoir::SpectralRadiusMethod;

Eigen::MatrixXd ramp(Eigen::Index rows, Eigen::Index cols, double phase) {
  Eigen::MatrixXd values(rows, cols);
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    values(i) = std::sin(0.37 * static_cast<double>(i) + phase);
  }
  return values;
}

std::string toBytes(const Model &model) {
  std::ostringstream output;
  model.save(output);
  return output.str();
}

Model fromBytes(const std::string &bytes) {
  std::istringstream input(bytes);
  return Model::load(input);
}

Rule rule(Rule::PostSignal post, Rule::Update update, unsigned int seed) {
  Rule result;
  result.post = post;
  result.update = update;
  result.noise_seed = seed;
  result.learning_rate = 0.05;
  return result;
}

std::shared_ptr<RewardModulatedReservoir> plasticReservoir(int n, unsigned int seed, const Rule &learning_rule,
                                                           std::optional<std::vector<int>> pre = std::nullopt,
                                                           std::optional<std::vector<int>> post = std::nullopt) {
  return std::make_shared<RewardModulatedReservoir>(n, 0.9, 0.5, 0.4, 1.0, true, seed, Method::POWER_ITERATION, 1.0,
                                                    learning_rule, std::move(pre), std::move(post));
}

// Steps, rewards and noise-free steps that exercise the noise, the eligibility, the baselines
// and the update count.
Eigen::MatrixXd exercise(Model &model, int input_width) {
  Eigen::MatrixXd outputs = model.act(ramp(3, input_width, 0.1));
  model.reinforce(-0.5, 1);
  model.act(ramp(2, input_width, 0.7));
  model.reinforce(0.25, 1);
  model.act(ramp(2, input_width, 1.3));
  model.reinforce(0.75, 2, 0.1);
  const Eigen::MatrixXd last = model.act(ramp(2, input_width, 2.1));
  const Eigen::MatrixXd noise_free = model.act(ramp(2, input_width, 2.9), false);
  Eigen::MatrixXd all(outputs.rows() + last.rows() + noise_free.rows(), outputs.cols());
  all << outputs, last, noise_free;
  return all;
}

} // namespace

TEST_CASE("Reward-modulated serialization - a model saved mid-trial continues bit for bit", "[serialization][reward]") {
  Model model;
  int input_width = 1;
  const int which = GENERATE(0, 1, 2);
  if (which == 0) { // serial: an online plastic reservoir and an online Heaviside readout
    model.addReservoir(plasticReservoir(12, 3, rule(Rule::HIGH_PASS, Rule::ONLINE, 5)));
    Rule readout_rule = rule(Rule::HIGH_PASS, Rule::ONLINE, 6);
    readout_rule.modulator = Rule::HEAVISIDE;
    readout_rule.learning_rate_decay_steps = 10.0;
    model.setReadout(std::make_shared<RewardModulatedReadout>(2, readout_rule, true));
  } else if (which == 1) { // parallel and readout-less: trial cube and decorrelated windowed trials
    Rule cube = rule(Rule::HIGH_PASS, Rule::TRIAL, 7);
    cube.eligibility_power = 3.0;
    cube.noise_probability = 0.3;
    cube.max_update = 0.01;
    Rule decorrelated = rule(Rule::NOISE, Rule::TRIAL, 8);
    decorrelated.decorrelation = 1.0;
    decorrelated.baseline = Rule::WINDOW;
    decorrelated.baseline_window = 2;
    decorrelated.noise_distribution = Rule::GAUSSIAN;
    model.addReservoir(plasticReservoir(10, 4, cube), "parallel");
    model.addReservoir(plasticReservoir(8, 5, decorrelated, std::vector<int>{0, 2, 4}, std::vector<int>{1, 3, 5, 7}),
                       "parallel");
    input_width = 2;
  } else { // an identity NVAR and a decorrelated readout with Gaussian noise
    model.addReservoir(std::make_shared<NvarReservoir>(1, 1));
    Rule decorrelated = rule(Rule::NOISE, Rule::TRIAL, 9);
    decorrelated.decorrelation = 0.5;
    decorrelated.noise_distribution = Rule::GAUSSIAN;
    model.setReadout(std::make_shared<RewardModulatedReadout>(1, decorrelated, false));
    input_width = 3;
  }
  model.act(ramp(4, input_width, 0.0));
  model.reinforce(1.0, 1);
  model.act(ramp(3, input_width, 0.4)); // mid-trial: the eligibility is pending

  const std::string saved = toBytes(model);
  Model restored = fromBytes(saved);
  REQUIRE(toBytes(restored) == saved);
  REQUIRE(restored.hasReadout() == model.hasReadout());
  const Eigen::MatrixXd original_outputs = exercise(model, input_width);
  const Eigen::MatrixXd restored_outputs = exercise(restored, input_width);
  REQUIRE(restored_outputs == original_outputs);
  REQUIRE(toBytes(restored) == toBytes(model));
}

TEST_CASE("Reward-modulated serialization - restored overflow bounds accept and reject the same rows",
          "[serialization][reward]") {
  // Rows peaking on different features make the tracked bound larger than the largest entry,
  // so recomputing the bound from the matrix on load would accept a row the original rejects.
  Rule learning_rule = rule(Rule::NOISE, Rule::TRIAL, 1);
  Eigen::MatrixXd first(1, 2);
  Eigen::MatrixXd second(1, 2);
  Eigen::MatrixXd noise(1, 1);
  const bool covariance = GENERATE(false, true);
  if (covariance) { // pre^2 = 0.7e308 on each feature, with a tiny eligibility
    learning_rule.decorrelation = 1.0;
    const double a = std::sqrt(0.7e308);
    first << a, 0.0;
    second << 0.0, a;
    noise << 1e-200;
  } else { // S(pre) S(post) = 0.81e308 on each feature
    first << 0.9e154, 0.0;
    second << 0.0, 0.9e154;
    noise << 0.9e154;
  }
  auto readout = std::make_shared<RewardModulatedReadout>(1, learning_rule, false);
  readout->act(first, noise);
  readout->act(second, noise);
  Model model;
  model.addReservoir(std::make_shared<NvarReservoir>(1, 1));
  model.setReadout(readout);
  Model restored = fromBytes(toBytes(model));
  auto restored_readout = std::dynamic_pointer_cast<RewardModulatedReadout>(restored.getReadout());
  REQUIRE(restored_readout->getLearning().getEligibilityBound() == readout->getLearning().getEligibilityBound());
  REQUIRE(restored_readout->getLearning().getCovarianceBound() == readout->getLearning().getCovarianceBound());
  REQUIRE_THROWS_AS(readout->act(first, noise), std::runtime_error);
  REQUIRE_THROWS_AS(restored_readout->act(first, noise), std::runtime_error);
}

namespace {

// A hand-written RewardModulatedReadout payload with an uninitialized learning state.
struct ReadoutPayload {
  int distribution = 0;
  int post = 1;
  int update = 0;
  int modulator = 0;
  int baseline = 0;
  double learning_rate = 0.1;
  int window = 3;
  std::vector<std::pair<int, std::vector<double>>> histories;
};

std::string writeReadoutModel(const ReadoutPayload &payload) {
  std::ostringstream buffer;
  BinaryWriter writer(buffer);
  writer.writeHeader();
  writer.writeU8(0); // serial
  writer.writeUInt(1);
  writer.writeString("NvarReservoir");
  NvarReservoir(1, 1).save(writer);
  writer.writeBool(true);
  writer.writeString("RewardModulatedReadout");
  writer.writeInt(1);      // n_outputs
  writer.writeBool(false); // include_bias
  writer.writeDouble(payload.learning_rate);
  writer.writeU8(static_cast<std::uint8_t>(payload.distribution));
  writer.writeDouble(0.1); // noise_amplitude
  writer.writeDouble(1.0); // noise_probability
  writer.writeUInt(0);     // noise_seed
  writer.writeU8(static_cast<std::uint8_t>(payload.post));
  writer.writeDouble(0.0); // post_decay
  writer.writeDouble(1.0); // eligibility_power
  writer.writeU8(static_cast<std::uint8_t>(payload.update));
  writer.writeDouble(0.0); // trace_decay
  writer.writeU8(static_cast<std::uint8_t>(payload.modulator));
  writer.writeU8(static_cast<std::uint8_t>(payload.baseline));
  writer.writeDouble(0.8); // baseline_decay
  writer.writeInt(payload.window);
  for (int i = 0; i < 3; ++i) {
    writer.writeBool(false); // learning_rate_decay_steps, max_update, decorrelation
  }
  writer.writeU64(12345);  // noise state
  writer.writeU64(2);      // update count
  writer.writeBool(false); // learning state not initialized
  writer.writeUInt(static_cast<std::uint32_t>(payload.histories.size()));
  for (const auto &[key, history] : payload.histories) {
    writer.writeInt(key);
    writer.writeUInt(static_cast<std::uint32_t>(history.size()));
    for (double reward : history) {
      writer.writeDouble(reward);
    }
  }
  return buffer.str();
}

} // namespace

TEST_CASE("Reward-modulated serialization - a hand-written payload loads", "[serialization][reward]") {
  ReadoutPayload payload;
  payload.baseline = 1; // WINDOW
  payload.histories = {{0, {1.0, 2.0}}, {-4, {3.0}}};
  const Model model = fromBytes(writeReadoutModel(payload));
  const auto readout = std::dynamic_pointer_cast<RewardModulatedReadout>(model.getReadout());
  REQUIRE(readout->getRule().baseline == Rule::WINDOW);
  REQUIRE(readout->getLearning().getBaseline(0) == 1.5);
  REQUIRE(readout->getLearning().getBaseline(-4) == 3.0);
  REQUIRE(readout->getLearning().getUpdateCount() == 2);
  REQUIRE(readout->getLearning().getNoiseState() == 12345);
}

TEST_CASE("Reward-modulated serialization - invalid payloads are rejected", "[serialization][reward]") {
  ReadoutPayload payload;
  std::string expected_message;
  SECTION("Unknown codes") {
    const int which = GENERATE(0, 1, 2, 3, 4);
    (which == 0   ? payload.distribution
     : which == 1 ? payload.post
     : which == 2 ? payload.update
     : which == 3 ? payload.modulator
                  : payload.baseline) = 9;
    expected_message = "code 9";
  }
  SECTION("An invalid rule") {
    payload.learning_rate = -1.0;
    expected_message = "learning_rate";
  }
  SECTION("An EMA baseline with a history") {
    payload.histories = {{0, {1.0, 2.0}}};
    expected_message = "does not fit the rule";
  }
  SECTION("A window longer than the rule's") {
    payload.baseline = 1;
    payload.histories = {{0, {1.0, 2.0, 3.0, 4.0}}};
    expected_message = "too long";
  }
  SECTION("A baseline with no baseline kind") {
    payload.baseline = 2;
    payload.histories = {{0, {1.0}}};
    expected_message = "does not fit the rule";
  }
  SECTION("A key twice") {
    payload.histories = {{5, {1.0}}, {5, {2.0}}};
    expected_message = "appears twice";
  }
  SECTION("A non-finite reward") {
    payload.histories = {{0, {std::numeric_limits<double>::infinity()}}};
    expected_message = "not finite";
  }
  REQUIRE_THROWS_MATCHES(fromBytes(writeReadoutModel(payload)), SerializationError,
                         MessageMatches(ContainsSubstring(expected_message)));
}

TEST_CASE("Reward-modulated serialization - a bound below its matrix is rejected", "[serialization][reward]") {
  auto readout = std::make_shared<RewardModulatedReadout>(1, rule(Rule::NOISE, Rule::TRIAL, 2), false);
  readout->act(Eigen::MatrixXd::Constant(1, 2, 2.0), Eigen::MatrixXd::Constant(1, 1, 3.0));
  Model model;
  model.addReservoir(std::make_shared<NvarReservoir>(1, 1));
  model.setReadout(readout);
  std::string bytes = toBytes(model);
  // The eligibility (2 x 1, all 6) is followed by the empty covariance and the eligibility bound.
  const double six = 6.0;
  const std::string eligibility_bound(reinterpret_cast<const char *>(&six), 8);
  const std::size_t at = bytes.rfind(eligibility_bound);
  REQUIRE(at != std::string::npos);
  const double lowered = 5.0;
  bytes.replace(at, 8, reinterpret_cast<const char *>(&lowered), 8);
  REQUIRE_THROWS_MATCHES(fromBytes(bytes), SerializationError, MessageMatches(ContainsSubstring("exceeds its bound")));
}

TEST_CASE("Reward-modulated serialization - version 3 files know no reward-modulated components",
          "[serialization][reward]") {
  const auto craft = [](const std::function<void(BinaryWriter &)> &body) {
    std::ostringstream buffer;
    BinaryWriter writer(buffer);
    for (char c : std::string("RCLIBMDL")) {
      writer.writeU8(static_cast<std::uint8_t>(c));
    }
    writer.writeUInt(3);
    body(writer);
    return buffer.str();
  };
  SECTION("Reservoir") {
    const std::string bytes = craft([](BinaryWriter &writer) {
      writer.writeU8(0);
      writer.writeUInt(1);
      writer.writeString("RewardModulatedReservoir");
    });
    REQUIRE_THROWS_MATCHES(fromBytes(bytes), SerializationError,
                           MessageMatches(ContainsSubstring("unknown reservoir type tag")));
  }
  SECTION("Readout, with no readout flag before it") {
    const std::string bytes = craft([](BinaryWriter &writer) {
      writer.writeU8(0);
      writer.writeUInt(1);
      writer.writeString("NvarReservoir");
      NvarReservoir(1, 1).save(writer);
      writer.writeString("RewardModulatedReadout");
    });
    REQUIRE_THROWS_MATCHES(fromBytes(bytes), SerializationError,
                           MessageMatches(ContainsSubstring("unknown readout type tag")));
  }
}

TEST_CASE("Reward-modulated serialization - a model without a readout round-trips", "[serialization][reward]") {
  Model model;
  model.addReservoir(std::make_shared<RandomSparseReservoir>(6, 0.9, 0.5, 0.5, 1.0, true, 3));
  model.act(ramp(2, 1, 0.0));
  const std::string saved = toBytes(model);
  Model restored = fromBytes(saved);
  REQUIRE_FALSE(restored.hasReadout());
  REQUIRE(restored.act(ramp(3, 1, 0.5), false) == model.act(ramp(3, 1, 0.5), false));
  REQUIRE_THROWS_AS(restored.predict(ramp(1, 1, 0.0)), std::runtime_error);
}
