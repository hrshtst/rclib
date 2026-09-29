#include "rclib/readouts/LmsReadout.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <limits>
#include <stdexcept>

TEST_CASE("LmsReadout - fit and predict", "[LmsReadout]") {
  int n_samples = 100;
  int n_features = 10;
  int n_targets = 2;

  Eigen::MatrixXd states = Eigen::MatrixXd::Random(n_samples, n_features);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(n_samples, n_targets);

  SECTION("Without bias") {
    LmsReadout readout(0.01, false);
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
    LmsReadout readout(0.01, true);
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

TEST_CASE("LmsReadout - partialFit", "[LmsReadout]") {
  int n_features = 5;
  int n_targets = 1;
  LmsReadout readout(0.01, false);

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

TEST_CASE("LmsReadout - rejects non-finite hyperparameters", "[LmsReadout]") {
  const double bad = GENERATE(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity());
  REQUIRE_THROWS_AS(LmsReadout(bad), std::invalid_argument);
}

TEST_CASE("LmsReadout - configuration getters and input width", "[LmsReadout]") {
  LmsReadout readout(0.05, false);
  REQUIRE(readout.getLearningRate() == 0.05);
  REQUIRE_FALSE(readout.getIncludeBias());

  REQUIRE(readout.getInputDim() == 0);
  readout.partialFit(Eigen::MatrixXd::Random(1, 6), Eigen::MatrixXd::Random(1, 2));
  REQUIRE(readout.getInputDim() == 6);
}

TEST_CASE("LmsReadout - fit rejects invalid input and keeps the fitted state", "[LmsReadout]") {
  LmsReadout readout(0.05, true);
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(5, 4);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(5, 2);
  readout.fit(states, targets);
  const Eigen::MatrixXd before = readout.predict(states);

  Eigen::MatrixXd bad_states = states;
  Eigen::MatrixXd bad_targets = targets;
  SECTION("No rows") {
    bad_states = Eigen::MatrixXd(0, 4);
    bad_targets = Eigen::MatrixXd(0, 2);
  }
  SECTION("No state columns") { bad_states = Eigen::MatrixXd(5, 0); }
  SECTION("Fewer target rows than states") { bad_targets = targets.topRows(3); } // used to read past the end
  SECTION("More target rows than states") { bad_targets = Eigen::MatrixXd::Random(7, 2); }
  SECTION("No target columns") { bad_targets = Eigen::MatrixXd(5, 0); }

  REQUIRE_THROWS_AS(readout.fit(bad_states, bad_targets), std::invalid_argument);
  REQUIRE(readout.getInputDim() == 4);
  REQUIRE(readout.predict(states) == before);
}

TEST_CASE("LmsReadout - fit applies one update per sample", "[LmsReadout]") {
  // fit is sequential LMS, not one averaged batch update.
  const Eigen::MatrixXd states = Eigen::MatrixXd::Random(6, 3);
  const Eigen::MatrixXd targets = Eigen::MatrixXd::Random(6, 1);
  LmsReadout fitted(0.1, true);
  LmsReadout sequential(0.1, true);
  fitted.fit(states, targets);
  for (Eigen::Index i = 0; i < states.rows(); ++i) {
    sequential.partialFit(states.row(i), targets.row(i));
  }
  REQUIRE(fitted.predict(states) == sequential.predict(states));
}
