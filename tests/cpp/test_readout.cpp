#include "rclib/readouts/RidgeReadout.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

TEST_CASE("RidgeReadout - fit and predict", "[RidgeReadout]") {
  int n_samples = 100;
  int n_features = 10;
  int n_targets = 2;

  Eigen::MatrixXd states = Eigen::MatrixXd::Random(n_samples, n_features);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(n_samples, n_targets);

  SECTION("Without bias") {
    RidgeReadout readout(0.1, false);
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
    RidgeReadout readout(0.1, true);
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

TEST_CASE("RidgeReadout - partialFit throws error", "[RidgeReadout]") {
  RidgeReadout readout;
  Eigen::MatrixXd state = Eigen::MatrixXd::Random(1, 10);
  Eigen::MatrixXd target = Eigen::MatrixXd::Random(1, 2);

  REQUIRE_THROWS_AS(readout.partialFit(state, target), std::logic_error);
}
