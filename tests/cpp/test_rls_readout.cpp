#include "rclib/readouts/RlsReadout.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

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
