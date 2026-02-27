#include "rclib/readouts/RlsReadout.h"

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("RlsReadout Mini-batch Consistency", "[readout][rls]") {
  int n_samples = 16;
  int n_features = 20;
  int n_targets = 2;

  Eigen::MatrixXd states = Eigen::MatrixXd::Random(n_samples, n_features);
  Eigen::MatrixXd targets = Eigen::MatrixXd::Random(n_samples, n_targets);

  double delta = 1.0;

  SECTION("rank1_update vs rank_k_update (No Bias, lambda=1.0 - optimized path)") {
    double lambda = 1.0;
    RlsReadout rls1(lambda, delta, false, RlsReadout::RANK1_UPDATE);
    rls1.partialFit(states, targets);
    Eigen::MatrixXd pred1 = rls1.predict(states);

    RlsReadout rlsK(lambda, delta, false, RlsReadout::RANK_K_UPDATE);
    rlsK.partialFit(states, targets);
    Eigen::MatrixXd predK = rlsK.predict(states);

    REQUIRE(pred1.isApprox(predK, 1e-10));
  }

  SECTION("rank1_update vs rank_k_update (With Bias, lambda=0.99 - fallback path)") {
    double lambda = 0.99;
    RlsReadout rls1(lambda, delta, true, RlsReadout::RANK1_UPDATE);
    rls1.partialFit(states, targets);
    Eigen::MatrixXd pred1 = rls1.predict(states);

    RlsReadout rlsK(lambda, delta, true, RlsReadout::RANK_K_UPDATE);
    rlsK.partialFit(states, targets);
    Eigen::MatrixXd predK = rlsK.predict(states);

    // Fallback path should also be equivalent since it just loops
    REQUIRE(pred1.isApprox(predK, 1e-10));
  }
}
