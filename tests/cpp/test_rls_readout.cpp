#include "rclib/readouts/RlsReadout.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

TEST_CASE("RlsReadout - fit and predict", "[RlsReadout]") {
  int n_samples = 100;
  int n_features = 10;
  int n_targets = 2;

  Eigen::MatrixXd states = Eigen::MatrixXd::Random(n_samples, n_features);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(n_samples, n_targets);

  SECTION("Without bias") {
    RlsReadout readout(0.99, 1.0, false);
    readout.fit(states, targets);
    Eigen::MatrixXd predictions = readout.predict(states);

    REQUIRE(predictions.rows() == n_samples);
    REQUIRE(predictions.cols() == n_targets);
    // Check if the prediction error is smaller than the original error
    double prediction_error = (predictions - targets).squaredNorm();
    double original_error = targets.squaredNorm();
    REQUIRE(prediction_error < original_error);
  }

  SECTION("With bias") {
    RlsReadout readout(0.99, 1.0, true);
    readout.fit(states, targets);
    Eigen::MatrixXd predictions = readout.predict(states);

    REQUIRE(predictions.rows() == n_samples);
    REQUIRE(predictions.cols() == n_targets);
    // Check if the prediction error is smaller than the original error
    double prediction_error = (predictions - targets).squaredNorm();
    double original_error = targets.squaredNorm();
    REQUIRE(prediction_error < original_error);
  }
}

TEST_CASE("RlsReadout - partialFit", "[RlsReadout]") {
  int n_features = 5;
  int n_targets = 1;
  RlsReadout readout(0.99, 1.0, false);

  Eigen::MatrixXd state1 = Eigen::MatrixXd::Random(1, n_features);
  Eigen::MatrixXd target1 = Eigen::MatrixXd::Random(1, n_targets);

  readout.partialFit(state1, target1);
  Eigen::MatrixXd predictions1 = readout.predict(state1);
  REQUIRE(predictions1.rows() == 1);
  REQUIRE(predictions1.cols() == n_targets);

  Eigen::MatrixXd state2 = Eigen::MatrixXd::Random(1, n_features);
  Eigen::MatrixXd target2 = Eigen::MatrixXd::Random(1, n_targets);
  readout.partialFit(state2, target2);
  Eigen::MatrixXd predictions2 = readout.predict(state2);
  REQUIRE(predictions2.rows() == 1);
  REQUIRE(predictions2.cols() == n_targets);
}

TEST_CASE("RlsReadout - rejects non-finite hyperparameters", "[RlsReadout]") {
  const double bad = GENERATE(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity());
  REQUIRE_THROWS_AS(RlsReadout(bad, 1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(RlsReadout(0.99, bad), std::invalid_argument);
}

TEST_CASE("RlsReadout - configuration getters and input width", "[RlsReadout]") {
  RlsReadout readout(0.95, 2.0, true, RlsReadout::RANK_K_UPDATE);
  REQUIRE(readout.getLambda() == 0.95);
  REQUIRE(readout.getDelta() == 2.0);
  REQUIRE(readout.getIncludeBias());
  REQUIRE(readout.getSolver() == RlsReadout::RANK_K_UPDATE);

  REQUIRE(readout.getInputDim() == 0);
  readout.partialFit(Eigen::MatrixXd::Random(3, 5), Eigen::MatrixXd::Random(3, 1));
  REQUIRE(readout.getInputDim() == 5);

  // A failed fit leaves the readout unfitted even though the previous weights
  // are still allocated; getInputDim must follow the initialized flag.
  REQUIRE_THROWS(readout.fit(Eigen::MatrixXd(0, 5), Eigen::MatrixXd(0, 1)));
  REQUIRE(readout.getInputDim() == 0);
  REQUIRE_THROWS(readout.predict(Eigen::MatrixXd::Random(1, 5)));

  // The next update starts afresh and may use a different width.
  readout.partialFit(Eigen::MatrixXd::Random(1, 4), Eigen::MatrixXd::Random(1, 1));
  REQUIRE(readout.getInputDim() == 4);
}

TEST_CASE("RlsReadout - the weights are readable", "[RlsReadout]") {
  const int n_samples = 40;
  const int n_features = 6;
  const int n_targets = 2;
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(n_samples, n_features);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(n_samples, n_targets);

  SECTION("Before the first update") {
    RlsReadout readout(0.99, 1.0, true);
    REQUIRE_THROWS_AS(readout.getWeights(), std::runtime_error);
  }

  SECTION("With bias the last row is the bias term, after every update") {
    RlsReadout readout(0.99, 1.0, true);
    for (int i = 0; i < n_samples; ++i) {
      readout.partialFit(states.row(i), targets.row(i));
      const Eigen::MatrixXd &weights = readout.getWeights();
      REQUIRE(weights.rows() == n_features + 1);
      REQUIRE(weights.cols() == n_targets);
      Eigen::MatrixXd expected = states * weights.topRows(n_features);
      expected.rowwise() += weights.row(n_features);
      REQUIRE((readout.predict(states) - expected).norm() < 1e-12);
    }
  }

  SECTION("Without bias the matrix is the plain map") {
    RlsReadout readout(1.0, 1.0, false, RlsReadout::RANK_K_UPDATE);
    readout.fit(states, targets);
    const Eigen::MatrixXd &weights = readout.getWeights();
    REQUIRE(weights.rows() == n_features);
    REQUIRE(weights.cols() == n_targets);
    REQUIRE((readout.predict(states) - states * weights).norm() < 1e-12);
  }

  SECTION("The first update from zero weights") {
    // With P = I / delta, one rank-1 update gives W = x^T y / (delta * lambda + x x^T).
    const double lambda = 0.9;
    const double delta = 2.0;
    RlsReadout readout(lambda, delta, false);
    const Eigen::MatrixXd x = states.topRows(1);
    const Eigen::MatrixXd y = targets.topRows(1);
    readout.partialFit(x, y);
    const Eigen::MatrixXd expected = x.transpose() * y / (delta * lambda + x.squaredNorm());
    REQUIRE((readout.getWeights() - expected).norm() < 1e-12);
  }

  SECTION("A failed fit leaves no weights to read") {
    RlsReadout readout(0.99, 1.0, true);
    readout.partialFit(states, targets);
    REQUIRE_THROWS(readout.fit(Eigen::MatrixXd(0, n_features), Eigen::MatrixXd(0, n_targets)));
    REQUIRE_THROWS_AS(readout.getWeights(), std::runtime_error);
  }
}

namespace {

bool sameBits(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.data(), b.data(), sizeof(double) * static_cast<std::size_t>(a.size())) == 0;
}

// Inputs that excite two of four directions exactly: the first two features are
// equal, and so are the last two, so [1, -1, 0, 0] and [0, 0, 1, -1] see nothing.
Eigen::MatrixXd confinedInputs(int n_samples) {
  Eigen::MatrixXd states(n_samples, 4);
  for (int t = 0; t < n_samples; ++t) {
    const double a = std::sin(0.3 * t);
    const double b = 0.5 * std::cos(0.7 * t) + 0.2;
    states.row(t) << a, a, b, b;
  }
  return states;
}

} // namespace

TEST_CASE("RlsReadout - forgetting method", "[RlsReadout]") {
  REQUIRE(RlsReadout(0.99, 1.0).getForgettingMethod() == RlsReadout::EXPONENTIAL);
  REQUIRE(RlsReadout(0.99, 1.0, true, RlsReadout::RANK1_UPDATE, RlsReadout::DIRECTIONAL).getForgettingMethod() ==
          RlsReadout::DIRECTIONAL);
  REQUIRE_THROWS_AS(RlsReadout(0.99, 1.0, true, RlsReadout::RANK1_UPDATE, static_cast<RlsReadout::ForgettingMethod>(2)),
                    std::invalid_argument);
}

TEST_CASE("RlsReadout - directional forgetting is exponential without forgetting", "[RlsReadout]") {
  // At lambda = 1 nothing is forgotten, so the two methods give the same bits.
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(30, 5);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(30, 2);
  const auto solver = GENERATE(RlsReadout::RANK1_UPDATE, RlsReadout::RANK_K_UPDATE);
  RlsReadout exponential(1.0, 0.5, true, solver, RlsReadout::EXPONENTIAL);
  RlsReadout directional(1.0, 0.5, true, solver, RlsReadout::DIRECTIONAL);
  for (Eigen::Index start = 0; start < 30; start += 3) {
    exponential.partialFit(states.middleRows(start, 3), targets.middleRows(start, 3));
    directional.partialFit(states.middleRows(start, 3), targets.middleRows(start, 3));
    REQUIRE(sameBits(directional.getWeights(), exponential.getWeights()));
  }
}

TEST_CASE("RlsReadout - directional forgetting with one feature is exponential", "[RlsReadout]") {
  // With one direction, forgetting along the input is forgetting everywhere.
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(50, 1);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(50, 2);
  RlsReadout exponential(0.9, 2.0, false, RlsReadout::RANK1_UPDATE, RlsReadout::EXPONENTIAL);
  RlsReadout directional(0.9, 2.0, false, RlsReadout::RANK1_UPDATE, RlsReadout::DIRECTIONAL);
  for (Eigen::Index i = 0; i < states.rows(); ++i) {
    exponential.partialFit(states.row(i), targets.row(i));
    directional.partialFit(states.row(i), targets.row(i));
    REQUIRE((directional.getWeights() - exponential.getWeights()).norm() <=
            1e-12 * (1.0 + exponential.getWeights().norm()));
  }
}

TEST_CASE("RlsReadout - directional forgetting follows its information form", "[RlsReadout]") {
  // Each sample takes (1 - lambda) x x^T / (x^T P x) from the information
  // matrix R = P^-1, then adds x x^T; the weights move by R^-1 x times the error.
  const double lambda = 0.9;
  const double delta = 0.5;
  const int n_features = 4;
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(60, n_features);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(60, 2);
  RlsReadout readout(lambda, delta, true, RlsReadout::RANK1_UPDATE, RlsReadout::DIRECTIONAL);

  Eigen::MatrixXd R = delta * Eigen::MatrixXd::Identity(n_features + 1, n_features + 1);
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n_features + 1, 2);
  for (Eigen::Index i = 0; i < states.rows(); ++i) {
    Eigen::VectorXd x(n_features + 1);
    x << states.row(i).transpose(), 1.0;
    const double xPx = x.dot(R.ldlt().solve(x));
    R += (1.0 - (1.0 - lambda) / xPx) * x * x.transpose();
    W += R.ldlt().solve(x) * (targets.row(i) - x.transpose() * W);

    readout.partialFit(states.row(i), targets.row(i));
    REQUIRE((readout.getWeights() - W).norm() <= 1e-9 * (1.0 + W.norm()));
  }
}

TEST_CASE("RlsReadout - directional forgetting does not wind up", "[RlsReadout]") {
  // The inputs excite two of four directions. Exponential forgetting discounts the
  // other two as well: P grows there as lambda^-n, overflows within about 3,200
  // samples at lambda = 0.8, and the weights are lost. Directional forgetting
  // discounts only the excited directions, so it still forgets and follows a
  // target that changes halfway.
  const int n_samples = 4000;
  const Eigen::MatrixXd states = confinedInputs(n_samples);
  Eigen::MatrixXd targets(n_samples, 1);
  for (int t = 0; t < n_samples; ++t) {
    const double a = states(t, 0);
    const double b = states(t, 2);
    targets(t, 0) = t < n_samples / 2 ? 2.0 * a - b : -a + 3.0 * b;
  }
  const Eigen::MatrixXd tail_states = states.bottomRows(100);
  const Eigen::MatrixXd tail_targets = targets.bottomRows(100);

  RlsReadout exponential(0.8, 1.0, false, RlsReadout::RANK1_UPDATE, RlsReadout::EXPONENTIAL);
  RlsReadout directional(0.8, 1.0, false, RlsReadout::RANK1_UPDATE, RlsReadout::DIRECTIONAL);
  for (int t = 0; t < n_samples; ++t) {
    exponential.partialFit(states.row(t), targets.row(t));
    directional.partialFit(states.row(t), targets.row(t));
  }

  const Eigen::MatrixXd lost = exponential.predict(tail_states);
  REQUIRE_FALSE((lost.allFinite() && (lost - tail_targets).norm() < 1.0));

  REQUIRE((directional.predict(tail_states) - tail_targets).norm() < 1e-8);
  // The unexcited directions keep the prior's zero weights.
  const Eigen::MatrixXd &weights = directional.getWeights();
  REQUIRE(std::abs(weights(0, 0) - weights(1, 0)) < 1e-8);
  REQUIRE(std::abs(weights(2, 0) - weights(3, 0)) < 1e-8);
}

TEST_CASE("RlsReadout - directional forgetting in mini-batches is sequential", "[RlsReadout]") {
  // Below lambda = 1 the rank-k solver falls back to sequential rank-1 updates.
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(40, 3);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(40, 2);
  RlsReadout batched(0.95, 1.0, true, RlsReadout::RANK_K_UPDATE, RlsReadout::DIRECTIONAL);
  RlsReadout sequential(0.95, 1.0, true, RlsReadout::RANK1_UPDATE, RlsReadout::DIRECTIONAL);
  for (Eigen::Index start = 0; start < 40; start += 8) {
    batched.partialFit(states.middleRows(start, 8), targets.middleRows(start, 8));
    for (Eigen::Index i = start; i < start + 8; ++i) {
      sequential.partialFit(states.row(i), targets.row(i));
    }
  }
  REQUIRE(sameBits(batched.getWeights(), sequential.getWeights()));
}

TEST_CASE("RlsReadout - directional forgetting skips a zero input", "[RlsReadout]") {
  // A zero input carries no information: the weights stay, and nothing is forgotten.
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(10, 3);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 1);
  RlsReadout with_zero(0.9, 1.0, false, RlsReadout::RANK1_UPDATE, RlsReadout::DIRECTIONAL);
  RlsReadout without(0.9, 1.0, false, RlsReadout::RANK1_UPDATE, RlsReadout::DIRECTIONAL);
  with_zero.partialFit(states.topRows(5), targets.topRows(5));
  without.partialFit(states.topRows(5), targets.topRows(5));
  with_zero.partialFit(Eigen::MatrixXd::Zero(1, 3), Eigen::MatrixXd::Ones(1, 1));
  REQUIRE(sameBits(with_zero.getWeights(), without.getWeights()));
  with_zero.partialFit(states.bottomRows(5), targets.bottomRows(5));
  without.partialFit(states.bottomRows(5), targets.bottomRows(5));
  REQUIRE(sameBits(with_zero.getWeights(), without.getWeights()));
}
