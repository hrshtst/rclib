#include "rclib/reservoirs/RandomSparseReservoir.h"
#include "rclib/reservoirs/RewardModulatedReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using Rule = RewardModulatedHebbianRule;
using Method = RandomSparseReservoir::SpectralRadiusMethod;

Eigen::MatrixXd randomMatrix(Eigen::Index rows, Eigen::Index cols, unsigned int seed, double scale = 1.0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dis(-scale, scale);
  return Eigen::MatrixXd::NullaryExpr(rows, cols, [&]() { return dis(gen); });
}

RewardModulatedReservoir makeReservoir(const Rule &rule, int n = 6, double leak = 0.3, double sparsity = 0.6,
                                       std::optional<std::vector<int>> pre = std::nullopt,
                                       std::optional<std::vector<int>> post = std::nullopt) {
  return RewardModulatedReservoir(n, 0.9, sparsity, leak, 1.0, true, 5, Method::POWER_ITERATION, 1.0, rule,
                                  std::move(pre), std::move(post));
}

// Everything the reservoir learns, for checking that a rejected call changed nothing.
struct Snapshot {
  Eigen::MatrixXd weights;
  Eigen::MatrixXd eligibility;
  double eligibility_bound;
  Eigen::RowVectorXd post_average;
  bool post_average_initialized;
  bool has_eligibility;
  std::uint64_t update_count;
  std::map<int, std::vector<double>> baselines;

  explicit Snapshot(const RewardModulatedReservoir &res)
      : weights(res.getRecurrentWeights()), eligibility(res.getLearning().getEligibility()),
        eligibility_bound(res.getLearning().getEligibilityBound()), post_average(res.getLearning().getPostAverage()),
        post_average_initialized(res.getLearning().isPostAverageInitialized()),
        has_eligibility(res.getLearning().hasEligibility()), update_count(res.getLearning().getUpdateCount()),
        baselines(res.getLearning().getBaselines()) {}

  bool operator==(const Snapshot &other) const {
    return weights == other.weights && eligibility == other.eligibility &&
           eligibility_bound == other.eligibility_bound && post_average == other.post_average &&
           post_average_initialized == other.post_average_initialized && has_eligibility == other.has_eligibility &&
           update_count == other.update_count && baselines == other.baselines;
  }
};

} // namespace

TEST_CASE("RewardModulatedReservoir - constructor validation", "[RewardModulatedReservoir]") {
  const Rule rule;
  REQUIRE_THROWS_AS(RewardModulatedReservoir(0, 0.9, 0.5, 1.0, 1.0, false, 1, Method::POWER_ITERATION, 1.0, rule),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(RewardModulatedReservoir(5, 0.9, 0.5, 0.0, 1.0, false, 1, Method::POWER_ITERATION, 1.0, rule),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(RewardModulatedReservoir(5, 0.9, 0.5, 1.0, 1.0, false, 1, static_cast<Method>(5), 1.0, rule),
                    std::invalid_argument);
  Rule bad;
  bad.learning_rate = -1.0;
  REQUIRE_THROWS_AS(RewardModulatedReservoir(5, 0.9, 0.5, 1.0, 1.0, false, 1, Method::POWER_ITERATION, 1.0, bad),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(makeReservoir(rule, 6, 0.3, 0.6, std::vector<int>{0, 6}), std::invalid_argument);
  REQUIRE_THROWS_AS(makeReservoir(rule, 6, 0.3, 0.6, std::nullopt, std::vector<int>{-1}), std::invalid_argument);
  REQUIRE_THROWS_AS(makeReservoir(rule, 6, 0.3, 0.6, std::vector<int>{2, 2}), std::invalid_argument);
  const RewardModulatedReservoir sorted = makeReservoir(rule, 6, 0.3, 0.6, std::vector<int>{4, 1}, std::vector<int>{});
  REQUIRE(sorted.getPlasticPresynaptic() == std::vector<int>{1, 4});
  REQUIRE(sorted.getPlasticPostsynaptic() == std::vector<int>{});
  REQUIRE_THROWS_AS(sorted.getInputWeights(), std::runtime_error);
}

TEST_CASE("RewardModulatedReservoir - equals RandomSparseReservoir at leak rate 1 without noise",
          "[RewardModulatedReservoir]") {
  struct Config {
    int n;
    double sparsity;
    bool bias;
    unsigned int seed;
    Method method;
    int width;
  };
  const Config config =
      GENERATE(Config{7, 0.3, true, 42, Method::POWER_ITERATION, 3}, Config{50, 0.1, false, 0, Method::DENSE, 1},
               Config{1001, 0.01, true, 4294967295u, Method::POWER_ITERATION, 2});
  RandomSparseReservoir reference(config.n, 0.95, config.sparsity, 1.0, 0.8, config.bias, config.seed, config.method,
                                  1.5);
  RewardModulatedReservoir reservoir(config.n, 0.95, config.sparsity, 1.0, 0.8, config.bias, config.seed, config.method,
                                     1.5, Rule{});
  REQUIRE(Eigen::MatrixXd(reservoir.getRecurrentWeights()) == Eigen::MatrixXd(reference.getRecurrentWeights()));
  REQUIRE(reservoir.getBias() == reference.getBias());
  const Eigen::MatrixXd inputs = randomMatrix(20, config.width, 3);
  for (Eigen::Index t = 0; t < inputs.rows(); ++t) {
    REQUIRE(reservoir.advance(inputs.row(t)) == reference.advance(inputs.row(t)));
  }
  REQUIRE(reservoir.getInputWeights() == reference.getInputWeights());
  // A noise-free act over several rows continues the same sequence.
  const Eigen::MatrixXd more = randomMatrix(5, config.width, 4);
  const Eigen::MatrixXd rates = reservoir.act(more, false);
  for (Eigen::Index t = 0; t < more.rows(); ++t) {
    REQUIRE(Eigen::MatrixXd(rates.row(t)) == reference.advance(more.row(t)));
  }
}

TEST_CASE("RewardModulatedReservoir - the potential leaks and the noise enters outside the leak",
          "[RewardModulatedReservoir]") {
  Rule rule;
  rule.post = Rule::NOISE;
  RewardModulatedReservoir reservoir = makeReservoir(rule);
  const Eigen::MatrixXd inputs = randomMatrix(2, 2, 4);
  const Eigen::MatrixXd noise = randomMatrix(2, 6, 5, 0.2);
  const Eigen::MatrixXd rates = reservoir.act(inputs, noise);

  const Eigen::MatrixXd W_in = reservoir.getInputWeights();
  const Eigen::MatrixXd W = reservoir.getRecurrentWeights();
  const Eigen::RowVectorXd b = reservoir.getBias();
  const Eigen::RowVectorXd v1 = 0.3 * (inputs.row(0) * W_in + b) + noise.row(0);
  const Eigen::RowVectorXd r1 = v1.array().tanh();
  const Eigen::RowVectorXd v2 = 0.7 * v1 + 0.3 * (inputs.row(1) * W_in + b + r1 * W) + noise.row(1);
  REQUIRE(rates.row(0).isApprox(r1, 1e-14));
  REQUIRE(reservoir.getPotential().isApprox(v2, 1e-14));
  REQUIRE(rates.row(1).isApprox(Eigen::RowVectorXd(v2.array().tanh()), 1e-14));
  // pre = the rates before each step, post = the noise.
  REQUIRE(reservoir.getLearning().getEligibility().isApprox(r1.transpose() * noise.row(1), 1e-14));
}

TEST_CASE("RewardModulatedReservoir - updates change only existing plastic synapses", "[RewardModulatedReservoir]") {
  Rule rule;
  rule.post = Rule::NOISE;
  rule.update = Rule::TRIAL;
  rule.baseline = Rule::NONE;
  rule.learning_rate = 0.5;
  RewardModulatedReservoir reservoir =
      makeReservoir(rule, 6, 0.3, 0.6, std::vector<int>{0, 1, 2, 4}, std::vector<int>{1, 3, 4, 5});
  const Eigen::SparseMatrix<double> before = reservoir.getRecurrentWeights();
  const Eigen::MatrixXd inputs = randomMatrix(4, 2, 6);
  reservoir.act(inputs, randomMatrix(4, 6, 7, 0.2));
  const Eigen::MatrixXd eligibility = reservoir.getLearning().getEligibility();
  reservoir.reinforce(2.0);
  const Eigen::SparseMatrix<double> &after = reservoir.getRecurrentWeights();

  REQUIRE(after.nonZeros() == before.nonZeros());
  int changed = 0;
  for (int j = 0; j < 6; ++j) {
    Eigen::SparseMatrix<double>::InnerIterator it_after(after, j);
    for (Eigen::SparseMatrix<double>::InnerIterator it(before, j); it; ++it, ++it_after) {
      REQUIRE(it_after.row() == it.row());
      const bool plastic = (it.row() == 0 || it.row() == 1 || it.row() == 2 || it.row() == 4) && j != 0 && j != 2;
      if (plastic) {
        REQUIRE_THAT(it_after.value(),
                     Catch::Matchers::WithinAbs(it.value() + 0.5 * 2.0 * eligibility(it.row(), j), 1e-14));
        ++changed;
      } else {
        REQUIRE(it_after.value() == it.value());
      }
    }
  }
  REQUIRE(changed > 0);
}

TEST_CASE("RewardModulatedReservoir - lifecycle", "[RewardModulatedReservoir]") {
  Rule rule;
  rule.post = Rule::HIGH_PASS;
  rule.trace_decay = 0.5;
  rule.noise_seed = 9;

  SECTION("resetState clears the activity and the average but keeps the eligibility") {
    RewardModulatedReservoir reservoir = makeReservoir(rule);
    reservoir.act(randomMatrix(3, 2, 8));
    const Eigen::MatrixXd eligibility = reservoir.getLearning().getEligibility();
    reservoir.resetState();
    REQUIRE(reservoir.getState().isZero(0.0));
    REQUIRE(reservoir.getPotential().isZero(0.0));
    REQUIRE_FALSE(reservoir.getLearning().isPostAverageInitialized());
    REQUIRE(reservoir.getLearning().getEligibility() == eligibility);
    REQUIRE(reservoir.getLearning().hasEligibility());
  }
  SECTION("several rows in one act equal consecutive single-row acts") {
    RewardModulatedReservoir batched = makeReservoir(rule);
    RewardModulatedReservoir stepped = makeReservoir(rule);
    const Eigen::MatrixXd inputs = randomMatrix(4, 2, 10);
    const Eigen::MatrixXd rates = batched.act(inputs);
    for (Eigen::Index t = 0; t < inputs.rows(); ++t) {
      REQUIRE(stepped.act(inputs.row(t)) == rates.row(t));
    }
    REQUIRE(Snapshot(batched) == Snapshot(stepped));
  }
  SECTION("a noise-free step changes no learning state and leaves the generator alone") {
    RewardModulatedReservoir reservoir = makeReservoir(rule);
    reservoir.act(randomMatrix(2, 2, 11));
    const Snapshot before(reservoir);
    const std::uint64_t noise_state = reservoir.getLearning().getNoiseState();
    reservoir.act(randomMatrix(3, 2, 12), false);
    reservoir.advance(randomMatrix(1, 2, 13));
    REQUIRE(Snapshot(reservoir) == before);
    REQUIRE(reservoir.getLearning().getNoiseState() == noise_state);
  }
}

TEST_CASE("RewardModulatedReservoir - rejected calls change no learning state", "[RewardModulatedReservoir]") {
  Rule rule;
  rule.post = Rule::NOISE;
  rule.baseline = Rule::NONE;

  SECTION("non-finite or mismatched inputs") {
    RewardModulatedReservoir reservoir = makeReservoir(rule);
    reservoir.act(randomMatrix(1, 2, 14));
    const Snapshot before(reservoir);
    const Eigen::MatrixXd state = reservoir.getState();
    Eigen::MatrixXd inputs = randomMatrix(2, 2, 15);
    inputs(1, 1) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(reservoir.act(inputs), std::invalid_argument);
    REQUIRE_THROWS_AS(reservoir.act(randomMatrix(1, 3, 15)), std::invalid_argument);
    REQUIRE(Snapshot(reservoir) == before);
    REQUIRE(reservoir.getState() == state);
  }
  SECTION("a potential that overflows, exploring or not") {
    RewardModulatedReservoir reservoir(6, 0.9, 0.6, 1.0, 1e10, true, 5, Method::POWER_ITERATION, 1.0, rule);
    reservoir.act(randomMatrix(1, 2, 16));
    const Snapshot before(reservoir);
    const bool explore = GENERATE(true, false);
    REQUIRE_THROWS_AS(reservoir.act(Eigen::MatrixXd::Constant(1, 2, 1e300), explore), std::runtime_error);
    REQUIRE(Snapshot(reservoir) == before);
  }
  SECTION("new weights that overflow") {
    rule.learning_rate = 1e308;
    RewardModulatedReservoir reservoir = makeReservoir(rule);
    reservoir.act(randomMatrix(2, 2, 17), randomMatrix(2, 6, 18));
    const Snapshot before(reservoir);
    REQUIRE_THROWS_AS(reservoir.reinforce(10.0), std::runtime_error);
    REQUIRE(Snapshot(reservoir) == before);
    REQUIRE_THROWS_AS(reservoir.commitUpdate(), std::logic_error);
  }
}

TEST_CASE("RewardModulatedReservoir - recurrent weights learn a delayed target from trial rewards",
          "[RewardModulatedReservoir]") {
  // Two trial types, a constant input of +1 or -1 for 15 steps; neuron 0 should end at +0.6 or
  // -0.6. Each trial gives one reward, -|error| at its end.
  Rule rule;
  rule.update = Rule::TRIAL;
  rule.noise_amplitude = 0.3;
  rule.noise_probability = 0.2;
  rule.baseline_decay = 0.33;
  rule.max_update = 0.01;
  const bool supralinear = GENERATE(false, true);
  if (supralinear) { // Miconi's rule: the cube of the high-passed potential
    rule.post = Rule::HIGH_PASS;
    rule.post_decay = 0.05;
    rule.eligibility_power = 3.0;
    rule.learning_rate = 20.0;
  } else { // node perturbation with the known noise
    rule.post = Rule::NOISE;
    rule.learning_rate = 1.0;
  }
  RewardModulatedReservoir reservoir(30, 0.9, 0.5, 0.3, 1.0, true, 10, Method::POWER_ITERATION, 1.0, rule);
  const auto trialInput = [](int type) { return Eigen::MatrixXd::Constant(15, 1, type == 0 ? 1.0 : -1.0); };
  const auto target = [](int type) { return type == 0 ? 0.6 : -0.6; };
  const auto error = [&]() {
    double sum = 0.0;
    for (int type = 0; type < 2; ++type) {
      reservoir.resetState();
      sum += std::abs(reservoir.act(trialInput(type), false)(14, 0) - target(type));
    }
    return sum / 2.0;
  };
  const double initial = error();
  for (int trial = 0; trial < 600; ++trial) {
    const int type = trial % 2;
    reservoir.resetState();
    const Eigen::MatrixXd rates = reservoir.act(trialInput(type));
    reservoir.reinforce(-std::abs(rates(14, 0) - target(type)), type);
  }
  REQUIRE(error() < initial / 4.0);
}
