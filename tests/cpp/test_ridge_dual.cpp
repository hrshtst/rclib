#include "rclib/readouts/RidgeReadout.h"

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("RidgeReadout Dual vs Primal Consistency", "[readout][ridge]") {
  int n_samples = 100;
  int n_features = 200; // N > T, triggers Dual if using AUTO, but we force them

  Eigen::MatrixXd states = Eigen::MatrixXd::Random(n_samples, n_features);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(n_samples, 2);
  double alpha = 0.1;

  SECTION("Without Bias") {
    RidgeReadout primal(alpha, false, RidgeReadout::CHOLESKY);
    primal.fit(states, targets);
    Eigen::MatrixXd W_primal = primal.predict(states);

    RidgeReadout dual(alpha, false, RidgeReadout::DUAL_CHOLESKY);
    dual.fit(states, targets);
    Eigen::MatrixXd W_dual = dual.predict(states);

    REQUIRE(W_primal.isApprox(W_dual, 1e-10));
  }

  SECTION("With Bias") {
    RidgeReadout primal(alpha, true, RidgeReadout::CHOLESKY);
    primal.fit(states, targets);
    Eigen::MatrixXd W_primal = primal.predict(states);

    RidgeReadout dual(alpha, true, RidgeReadout::DUAL_CHOLESKY);
    dual.fit(states, targets);
    Eigen::MatrixXd W_dual = dual.predict(states);

    // This specifically tests the K.array() += 1.0 logic vs Primal aug
    REQUIRE(W_primal.isApprox(W_dual, 1e-10));
  }
}

TEST_CASE("RidgeReadout Numerical Stability at Scale", "[readout][ridge][.slow]") {
  // Test a larger problem to exercise the parallel GEMM paths
  int n_samples = 1000;
  int n_features = 1200;

  Eigen::MatrixXd states = Eigen::MatrixXd::Random(n_samples, n_features);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(n_samples, 1);

  RidgeReadout readout(1e-6, true, RidgeReadout::AUTO);
  readout.fit(states, targets);

  Eigen::MatrixXd pred = readout.predict(states);

  // Simple sanity check: MSE on training data should be reasonably low for random data
  double mse = (pred - targets).squaredNorm() / n_samples;
  CHECK(mse < 1.0);
}
