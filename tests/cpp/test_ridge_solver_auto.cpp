#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <rclib/readouts/RidgeReadout.h>

TEST_CASE("RidgeReadout Adaptive Solver Selection", "[readout][ridge]") {
  SECTION("Small problem uses CHOLESKY") {
    RidgeReadout readout(1e-8, true, RidgeReadout::AUTO);
    Eigen::MatrixXd states = Eigen::MatrixXd::Random(100, 1000);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(100, 1);

    readout.fit(states, targets);

    CHECK(readout.getSolver() == RidgeReadout::AUTO);
    CHECK(readout.getEffectiveSolver() == RidgeReadout::CHOLESKY);
  }

  SECTION("Large problem uses CONJUGATE_GRADIENT_IMPLICIT") {
    RidgeReadout readout(1e-8, true, RidgeReadout::AUTO);
    // N >= 8000 triggers IMPLICIT
    Eigen::MatrixXd states = Eigen::MatrixXd::Random(10, 8000);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 1);

    readout.fit(states, targets);

    CHECK(readout.getSolver() == RidgeReadout::AUTO);
    CHECK(readout.getEffectiveSolver() == RidgeReadout::CONJUGATE_GRADIENT_IMPLICIT);
  }

  SECTION("Explicit choice overrides AUTO") {
    RidgeReadout readout(1e-8, true, RidgeReadout::CHOLESKY);
    Eigen::MatrixXd states = Eigen::MatrixXd::Random(10, 8000);
    Eigen::MatrixXd targets = Eigen::MatrixXd::Random(10, 1);

    readout.fit(states, targets);

    CHECK(readout.getSolver() == RidgeReadout::CHOLESKY);
    CHECK(readout.getEffectiveSolver() == RidgeReadout::CHOLESKY);
  }
}
