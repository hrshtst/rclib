#include "rclib/readouts/RidgeReadout.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

TEST_CASE("RidgeReadout - fit and predict", "[RidgeReadout]") {
  int n_samples = 100;
  int n_features = 10;
  int n_targets = 2;

  Eigen::MatrixXd states = Eigen::MatrixXd::Random(n_samples, n_features);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(n_samples, n_targets);

  SECTION("Without bias - CHOLESKY") {
    RidgeReadout readout(0.1, false, RidgeReadout::CHOLESKY);
    readout.fit(states, targets);
    Eigen::MatrixXd predictions = readout.predict(states);

    REQUIRE(predictions.rows() == n_samples);
    REQUIRE(predictions.cols() == n_targets);
    // Check if the prediction error is smaller than the original error
    double prediction_error = (predictions - targets).squaredNorm();
    double original_error = targets.squaredNorm();
    REQUIRE(prediction_error < original_error);
  }

  SECTION("Without bias - DUAL_CHOLESKY") {
    // Force dual even if N < T to test it
    RidgeReadout readout(0.1, false, RidgeReadout::DUAL_CHOLESKY);
    readout.fit(states, targets);
    Eigen::MatrixXd predictions = readout.predict(states);

    REQUIRE(predictions.rows() == n_samples);
    REQUIRE(predictions.cols() == n_targets);
    double prediction_error = (predictions - targets).squaredNorm();
    double original_error = targets.squaredNorm();
    REQUIRE(prediction_error < original_error);
  }

  SECTION("With bias - DUAL_CHOLESKY") {
    RidgeReadout readout(0.1, true, RidgeReadout::DUAL_CHOLESKY);
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

TEST_CASE("RidgeReadout Solver Consistency", "[RidgeReadout]") {
  int n_samples = 50;
  int n_features = 80;
  Eigen::MatrixXd states = Eigen::MatrixXd::Random(n_samples, n_features);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(n_samples, 1);

  RidgeReadout primal(0.1, true, RidgeReadout::CHOLESKY);
  primal.fit(states, targets);
  Eigen::MatrixXd pred_primal = primal.predict(states);

  RidgeReadout dual(0.1, true, RidgeReadout::DUAL_CHOLESKY);
  dual.fit(states, targets);
  Eigen::MatrixXd pred_dual = dual.predict(states);

  RidgeReadout implicit(0.1, true, RidgeReadout::CONJUGATE_GRADIENT_IMPLICIT, 1e-12);
  implicit.fit(states, targets);
  Eigen::MatrixXd pred_implicit = implicit.predict(states);

  REQUIRE(pred_primal.isApprox(pred_dual, 1e-6));
  REQUIRE(pred_primal.isApprox(pred_implicit, 1e-6));
}

TEST_CASE("RidgeReadout Adaptive Solver Selection", "[readout][ridge][.slow]") {
  SECTION("Small problem (N <= T) uses CHOLESKY") {
    RidgeReadout readout(1e-8, true, RidgeReadout::AUTO);
    Eigen::MatrixXd states = Eigen::MatrixXd::Random(200, 100);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(200, 1);

    readout.fit(states, targets);

    CHECK(readout.getSolver() == RidgeReadout::AUTO);
    CHECK(readout.getEffectiveSolver() == RidgeReadout::CHOLESKY);
  }

  SECTION("Problem with N > T uses DUAL_CHOLESKY") {
    RidgeReadout readout(1e-8, true, RidgeReadout::AUTO);
    Eigen::MatrixXd states = Eigen::MatrixXd::Random(100, 1000);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(100, 1);

    readout.fit(states, targets);

    CHECK(readout.getSolver() == RidgeReadout::AUTO);
    CHECK(readout.getEffectiveSolver() == RidgeReadout::DUAL_CHOLESKY);
  }

  SECTION("Large problem uses CONJUGATE_GRADIENT_IMPLICIT") {
    RidgeReadout readout(1e-8, true, RidgeReadout::AUTO);
    Eigen::MatrixXd states = Eigen::MatrixXd::Random(10, 8000);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 1);

    readout.fit(states, targets);

    CHECK(readout.getSolver() == RidgeReadout::AUTO);
    CHECK(readout.getEffectiveSolver() == RidgeReadout::CONJUGATE_GRADIENT_IMPLICIT);
  }
}
