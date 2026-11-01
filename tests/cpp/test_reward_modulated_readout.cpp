#include "rclib/readouts/RewardModulatedReadout.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using Rule = RewardModulatedHebbianRule;

// Everything the readout learns, for checking that a rejected call changed nothing.
struct Snapshot {
  Eigen::MatrixXd weights;
  Eigen::MatrixXd eligibility;
  Eigen::MatrixXd covariance;
  double eligibility_bound;
  double covariance_bound;
  Eigen::RowVectorXd post_average;
  bool post_average_initialized;
  bool has_eligibility;
  std::uint64_t update_count;
  std::map<int, std::vector<double>> baselines;

  explicit Snapshot(const RewardModulatedReadout &readout)
      : weights(readout.getWeights()), eligibility(readout.getLearning().getEligibility()),
        covariance(readout.getLearning().getCovariance()),
        eligibility_bound(readout.getLearning().getEligibilityBound()),
        covariance_bound(readout.getLearning().getCovarianceBound()),
        post_average(readout.getLearning().getPostAverage()),
        post_average_initialized(readout.getLearning().isPostAverageInitialized()),
        has_eligibility(readout.getLearning().hasEligibility()), update_count(readout.getLearning().getUpdateCount()),
        baselines(readout.getLearning().getBaselines()) {}

  bool operator==(const Snapshot &other) const {
    return weights == other.weights && eligibility == other.eligibility && covariance == other.covariance &&
           eligibility_bound == other.eligibility_bound && covariance_bound == other.covariance_bound &&
           post_average == other.post_average && post_average_initialized == other.post_average_initialized &&
           has_eligibility == other.has_eligibility && update_count == other.update_count &&
           baselines == other.baselines;
  }
};

Eigen::MatrixXd randomMatrix(Eigen::Index rows, Eigen::Index cols, unsigned int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  return Eigen::MatrixXd::NullaryExpr(rows, cols, [&]() { return dis(gen); });
}

Eigen::RowVectorXd withBias(const Eigen::RowVectorXd &state) {
  Eigen::RowVectorXd pre(state.size() + 1);
  pre << state, 1.0;
  return pre;
}

} // namespace

TEST_CASE("RewardModulatedReadout - constructor validation", "[RewardModulatedReadout]") {
  REQUIRE_THROWS_AS(RewardModulatedReadout(0, Rule{}, true), std::invalid_argument);
  Rule bad;
  bad.learning_rate = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(RewardModulatedReadout(1, bad, true), std::invalid_argument);
}

TEST_CASE("RewardModulatedReadout - weights, prediction and noise-free steps", "[RewardModulatedReadout]") {
  Rule rule;
  rule.noise_amplitude = 0.3;
  RewardModulatedReadout readout(2, rule, true);
  REQUIRE(readout.getInputDim() == 0);
  REQUIRE_THROWS_AS(readout.getWeights(), std::runtime_error);
  REQUIRE_THROWS_AS(readout.predict(Eigen::MatrixXd::Ones(1, 3)), std::runtime_error);
  REQUIRE_THROWS_AS(readout.fit(Eigen::MatrixXd::Ones(4, 3), Eigen::MatrixXd::Ones(4, 2)), std::logic_error);
  REQUIRE_THROWS_AS(readout.partialFit(Eigen::MatrixXd::Ones(1, 3), Eigen::MatrixXd::Ones(1, 2)), std::logic_error);

  SECTION("the first act starts from zero weights, so the output is the noise") {
    const Eigen::MatrixXd output = readout.act(Eigen::MatrixXd::Ones(1, 3));
    REQUIRE(readout.getInputDim() == 3);
    REQUIRE(readout.getWeights() == Eigen::MatrixXd::Zero(4, 2));
    REQUIRE(output.cwiseAbs().maxCoeff() <= 0.3);
    REQUIRE(output.cwiseAbs().maxCoeff() > 0.0);
  }
  SECTION("setWeights fixes the width; predict and noise-free steps give x W") {
    REQUIRE_THROWS_AS(readout.setWeights(Eigen::MatrixXd::Zero(1, 2)), std::invalid_argument); // bias row only
    REQUIRE_THROWS_AS(readout.setWeights(Eigen::MatrixXd::Zero(4, 3)), std::invalid_argument);
    const Eigen::MatrixXd W = randomMatrix(4, 2, 1);
    readout.setWeights(W);
    REQUIRE(readout.getInputDim() == 3);
    REQUIRE_THROWS_AS(readout.setWeights(Eigen::MatrixXd::Zero(5, 2)), std::invalid_argument);
    Eigen::MatrixXd bad = W;
    bad(0, 0) = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(readout.setWeights(bad), std::invalid_argument);
    REQUIRE(readout.getWeights() == W);

    const Eigen::MatrixXd states = randomMatrix(5, 3, 2);
    const Eigen::MatrixXd expected = states * W.topRows(3) + Eigen::VectorXd::Ones(5) * W.row(3);
    REQUIRE(readout.predict(states).isApprox(expected, 1e-15));

    const std::uint64_t noise_state = readout.getLearning().getNoiseState();
    const Snapshot before(readout);
    REQUIRE(readout.act(states, false).isApprox(expected, 1e-15));
    REQUIRE(Snapshot(readout) == before);
    REQUIRE(readout.getLearning().getNoiseState() == noise_state);
    REQUIRE_THROWS_AS(readout.act(Eigen::MatrixXd::Ones(1, 4)), std::invalid_argument);
  }
  SECTION("zero amplitude explores without changing the output") {
    rule.noise_amplitude = 0.0;
    RewardModulatedReadout silent(2, rule, false);
    const Eigen::MatrixXd W = randomMatrix(3, 2, 3);
    silent.setWeights(W);
    const Eigen::MatrixXd states = randomMatrix(4, 3, 4);
    REQUIRE(silent.act(states) == silent.predict(states));
    REQUIRE(silent.getLearning().hasEligibility());
  }
}

TEST_CASE("RewardModulatedReadout - one online step matches the rule", "[RewardModulatedReadout]") {
  Rule rule;
  rule.post = Rule::HIGH_PASS;
  rule.post_decay = 0.0;
  rule.learning_rate = 0.1;
  rule.baseline = Rule::EMA;
  rule.baseline_decay = 0.8;
  RewardModulatedReadout readout(2, rule, true);
  const Eigen::MatrixXd W0 = randomMatrix(4, 2, 5);
  readout.setWeights(W0);
  const Eigen::MatrixXd x = randomMatrix(3, 3, 6);
  const Eigen::MatrixXd xi = 0.1 * randomMatrix(3, 2, 7);
  const double R1 = -0.4;
  const double R2 = -0.1;
  const double R3 = -0.3;

  readout.act(x.row(0), xi.row(0));
  readout.reinforce(R1); // the first reward of the key: zero modulator, the baseline becomes R1
  readout.act(x.row(1), xi.row(1));
  readout.reinforce(R2);
  readout.act(x.row(2), xi.row(2));
  readout.reinforce(R3);

  const Eigen::RowVectorXd v1 = withBias(x.row(0)) * W0 + xi.row(0);
  const Eigen::RowVectorXd v2 = withBias(x.row(1)) * W0 + xi.row(1);
  const Eigen::MatrixXd W1 = W0 + 0.1 * (R2 - R1) * withBias(x.row(1)).transpose() * (v2 - v1);
  const Eigen::RowVectorXd v3 = withBias(x.row(2)) * W1 + xi.row(2);
  const Eigen::MatrixXd W2 = W1 + 0.1 * (R3 - (0.8 * R1 + 0.2 * R2)) * withBias(x.row(2)).transpose() * (v3 - v2);
  REQUIRE(readout.getWeights().isApprox(W2, 1e-12));
  REQUIRE(readout.getLearning().getUpdateCount() == 3);
}

TEST_CASE("RewardModulatedReadout - trial episodes match the rule", "[RewardModulatedReadout]") {
  Rule rule;
  rule.post = Rule::NOISE;
  rule.update = Rule::TRIAL;
  rule.baseline = Rule::NONE;
  rule.learning_rate = 0.05;
  const Eigen::MatrixXd x = randomMatrix(4, 3, 8);
  const Eigen::MatrixXd xi = 0.2 * randomMatrix(4, 2, 9);
  const Eigen::MatrixXd W0 = randomMatrix(4, 2, 10);
  const double reward = 0.7;

  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(4, 2);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(4, 4);

  SECTION("with a supralinear eligibility") {
    rule.eligibility_power = 2.0;
    RewardModulatedReadout readout(2, rule, true);
    readout.setWeights(W0);
    readout.act(x, xi);
    readout.reinforce(reward);
    for (Eigen::Index t = 0; t < x.rows(); ++t) {
      const Eigen::RowVectorXd pre = withBias(x.row(t));
      E += (pre.array() * pre.array().abs()).matrix().transpose() *
           (xi.row(t).array() * xi.row(t).array().abs()).matrix();
    }
    REQUIRE(readout.getWeights().isApprox(W0 + 0.05 * reward * E, 1e-12));
    REQUIRE(readout.getLearning().getEligibility().isZero(0.0));
  }
  SECTION("decorrelated") {
    rule.decorrelation = 0.7;
    RewardModulatedReadout readout(2, rule, true);
    readout.setWeights(W0);
    readout.act(x, xi);
    readout.reinforce(reward);
    for (Eigen::Index t = 0; t < x.rows(); ++t) {
      const Eigen::RowVectorXd pre = withBias(x.row(t));
      E += pre.transpose() * xi.row(t);
      C += pre.transpose() * pre;
    }
    const Eigen::MatrixXd expected = W0 + 0.05 * reward * (C + 0.7 * Eigen::MatrixXd::Identity(4, 4)).inverse() * E;
    REQUIRE(readout.getWeights().isApprox(expected, 1e-12));
  }
}

TEST_CASE("RewardModulatedReadout - several rows in one act equal consecutive single-row acts",
          "[RewardModulatedReadout]") {
  Rule rule;
  rule.post = Rule::HIGH_PASS;
  rule.post_decay = 0.3;
  rule.trace_decay = GENERATE(0.0, 0.6);
  rule.noise_seed = 11;
  RewardModulatedReadout batched(2, rule, true);
  RewardModulatedReadout stepped(2, rule, true);
  const Eigen::MatrixXd W0 = randomMatrix(4, 2, 12);
  batched.setWeights(W0);
  stepped.setWeights(W0);
  const Eigen::MatrixXd states = randomMatrix(5, 3, 13);

  const Eigen::MatrixXd outputs = batched.act(states);
  for (Eigen::Index t = 0; t < states.rows(); ++t) {
    REQUIRE(stepped.act(states.row(t)) == outputs.row(t));
  }
  REQUIRE(Snapshot(batched) == Snapshot(stepped));
  batched.reinforce(1.0, 0, 0.0);
  stepped.reinforce(1.0, 0, 0.0);
  REQUIRE(batched.getWeights() == stepped.getWeights());
}

TEST_CASE("RewardModulatedReadout - resetActivity restarts the post-synaptic average between episodes",
          "[RewardModulatedReadout]") {
  Rule rule;
  rule.post = Rule::HIGH_PASS;
  RewardModulatedReadout readout(1, rule, false);
  readout.setWeights(randomMatrix(2, 1, 14));
  readout.act(randomMatrix(3, 2, 15));
  REQUIRE(readout.getLearning().isPostAverageInitialized());
  readout.resetActivity();
  REQUIRE_FALSE(readout.getLearning().isPostAverageInitialized());
  const Eigen::MatrixXd output = readout.act(randomMatrix(1, 2, 16));
  REQUIRE(readout.getLearning().getEligibility().isZero(0.0));
  REQUIRE(readout.getLearning().getPostAverage() == output.row(0));
}

TEST_CASE("RewardModulatedReadout - rejected calls change no learning state", "[RewardModulatedReadout]") {
  Rule rule;
  rule.post = Rule::NOISE;
  rule.baseline = Rule::NONE;
  RewardModulatedReadout readout(1, rule, true);
  readout.setWeights(randomMatrix(3, 1, 17));
  readout.act(randomMatrix(1, 2, 18));
  const Snapshot before(readout);
  const std::uint64_t noise_state = readout.getLearning().getNoiseState();

  SECTION("non-finite input") {
    Eigen::MatrixXd states = randomMatrix(2, 2, 19);
    states(1, 0) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(readout.act(states), std::invalid_argument);
    REQUIRE(readout.getLearning().getNoiseState() == noise_state);
  }
  SECTION("non-finite explicit noise") {
    REQUIRE_THROWS_AS(
        readout.act(randomMatrix(1, 2, 20), Eigen::MatrixXd::Constant(1, 1, std::numeric_limits<double>::infinity())),
        std::invalid_argument);
    REQUIRE_THROWS_AS(readout.act(randomMatrix(1, 2, 20), Eigen::MatrixXd::Zero(2, 1)), std::invalid_argument);
  }
  SECTION("a reward with no exploring act since the last update") {
    readout.reinforce(1.0);
    const Snapshot updated(readout);
    REQUIRE_THROWS_AS(readout.reinforce(1.0), std::logic_error);
    REQUIRE_THROWS_AS(readout.modulate(1.0), std::logic_error);
    REQUIRE(Snapshot(readout) == updated);
    return;
  }
  SECTION("a non-finite reward") {
    REQUIRE_THROWS_AS(readout.reinforce(std::numeric_limits<double>::infinity()), std::invalid_argument);
  }
  REQUIRE(Snapshot(readout) == before);
}

TEST_CASE("RewardModulatedReadout - non-finite outputs and weights are rejected", "[RewardModulatedReadout]") {
  Rule rule;
  rule.post = Rule::NOISE;
  rule.baseline = Rule::NONE;

  SECTION("an output that overflows, exploring or not") {
    RewardModulatedReadout readout(1, rule, false);
    readout.setWeights(Eigen::MatrixXd::Constant(2, 1, 1e308));
    readout.act(Eigen::MatrixXd::Constant(1, 2, 1e-10));
    const Snapshot before(readout);
    const bool explore = GENERATE(true, false);
    REQUIRE_THROWS_AS(readout.act(Eigen::MatrixXd::Constant(1, 2, 10.0), explore), std::runtime_error);
    REQUIRE(Snapshot(readout) == before);
  }
  SECTION("new weights that overflow") {
    rule.learning_rate = 1e308;
    rule.noise_amplitude = 0.5;
    RewardModulatedReadout readout(1, rule, true);
    readout.setWeights(Eigen::MatrixXd::Constant(2, 1, 1.7e308));
    readout.act(Eigen::MatrixXd::Constant(1, 1, 1e-10), Eigen::MatrixXd::Constant(1, 1, 0.5));
    const Snapshot before(readout);
    REQUIRE_THROWS_AS(readout.reinforce(1.0), std::runtime_error);
    REQUIRE(Snapshot(readout) == before);
    // The rejected update left nothing prepared.
    REQUIRE_THROWS_AS(readout.commitUpdate(), std::logic_error);
  }
}

TEST_CASE("RewardModulatedReadout - learns a linear map from rewards alone", "[RewardModulatedReadout]") {
  // Node perturbation: the reward of the noisy output against that of the noise-free output.
  Rule rule;
  rule.post = Rule::NOISE;
  rule.noise_amplitude = 0.1;
  rule.learning_rate = 7.5;
  rule.noise_seed = 3;
  RewardModulatedReadout readout(2, rule, true);
  const Eigen::MatrixXd A = randomMatrix(5, 2, 21);
  std::mt19937 gen(22);
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  auto error = [&](const Eigen::MatrixXd &states) {
    return (readout.predict(states) - states * A).squaredNorm() / static_cast<double>(states.rows());
  };
  const Eigen::MatrixXd probe = randomMatrix(200, 5, 23);
  readout.setWeights(Eigen::MatrixXd::Zero(6, 2));
  const double initial = error(probe);
  for (int step = 0; step < 3000; ++step) {
    const Eigen::MatrixXd x = Eigen::MatrixXd::NullaryExpr(1, 5, [&]() { return dis(gen); });
    const Eigen::MatrixXd target = x * A;
    const double clean = -(readout.predict(x) - target).squaredNorm();
    const double reward = -(readout.act(x) - target).squaredNorm();
    readout.reinforce(reward, 0, clean);
  }
  REQUIRE(error(probe) < initial / 10.0);
}

TEST_CASE("RewardModulatedReadout - a failed first step leaves the readout unset", "[RewardModulatedReadout]") {
  Rule rule;
  rule.post = Rule::NOISE;
  rule.eligibility_power = 3.0; // 1e200 cubed overflows
  RewardModulatedReadout readout(1, rule, true);
  const std::uint64_t noise_state = readout.getLearning().getNoiseState();
  const bool explicit_noise = GENERATE(false, true);
  if (explicit_noise) {
    REQUIRE_THROWS_AS(readout.act(Eigen::MatrixXd::Constant(1, 3, 1e200), Eigen::MatrixXd::Constant(1, 1, 0.1)),
                      std::runtime_error);
  } else {
    REQUIRE_THROWS_AS(readout.act(Eigen::MatrixXd::Constant(1, 3, 1e200)), std::runtime_error);
  }
  REQUIRE(readout.getInputDim() == 0);
  REQUIRE_FALSE(readout.getLearning().isInitialized());
  REQUIRE(readout.getLearning().getEligibility().size() == 0);
  REQUIRE(readout.getLearning().getNoiseState() == noise_state);
  REQUIRE_THROWS_AS(readout.getWeights(), std::runtime_error);
  // The feature width is still open.
  readout.act(Eigen::MatrixXd::Ones(1, 2));
  REQUIRE(readout.getInputDim() == 2);
}

TEST_CASE("RewardModulatedReadout - a noise-free step before the weights exist changes nothing",
          "[RewardModulatedReadout]") {
  RewardModulatedReadout readout(2, Rule{}, true);
  const std::uint64_t noise_state = readout.getLearning().getNoiseState();
  REQUIRE(readout.act(Eigen::MatrixXd::Ones(3, 4), false) == Eigen::MatrixXd::Zero(3, 2));
  REQUIRE(readout.getInputDim() == 0);
  REQUIRE_FALSE(readout.getLearning().isInitialized());
  REQUIRE(readout.getLearning().getNoiseState() == noise_state);
}
