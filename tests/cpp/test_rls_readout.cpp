#include "rclib/readouts/RlsReadout.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
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
