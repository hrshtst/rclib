#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdlib>
#include <rclib/reservoirs/RandomSparseReservoir.h>

TEST_CASE("RandomSparseReservoir - Seed Consistency", "[unit][reservoir][seed]") {
  int n_neurons = 50;
  double spectral_radius = 0.9;
  double sparsity = 0.1;
  double leak_rate = 0.5;
  double input_scaling = 1.0;
  bool include_bias = true;
  unsigned int seed = 123;

  RandomSparseReservoir res1(n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, include_bias, seed);
  RandomSparseReservoir res2(n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, include_bias, seed);
  RandomSparseReservoir res3(n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, include_bias, 456);

  Eigen::MatrixXd input = Eigen::MatrixXd::Ones(1, 5);

  SECTION("Same seed produces identical state advancement") {
    Eigen::MatrixXd state1 = res1.advance(input);
    Eigen::MatrixXd state2 = res2.advance(input);
    REQUIRE(state1.isApprox(state2, 1e-12));
  }

  SECTION("Different seeds produce different state advancement") {
    Eigen::MatrixXd state1 = res1.advance(input);
    Eigen::MatrixXd state3 = res3.advance(input);
    REQUIRE_FALSE(state1.isApprox(state3, 1e-12));
  }
}

namespace {

// Reservoir states for a fixed input sequence, one row per step. The state starts at
// zero, so W_res only contributes from the second step on.
Eigen::MatrixXd runSequence(RandomSparseReservoir &res) {
  const int steps = 30;
  Eigen::MatrixXd states(steps, res.getNNeurons());
  for (int t = 0; t < steps; ++t) {
    states.row(t) = res.advance(Eigen::MatrixXd::Constant(1, 1, std::sin(0.3 * t)));
  }
  return states;
}

} // namespace

TEST_CASE("RandomSparseReservoir - Same seed reproduces states after other random draws", "[unit][reservoir][seed]") {
  int n_neurons = 300;
  double spectral_radius = 0.5;
  double sparsity = 0.1;
  double leak_rate = 0.3;
  double input_scaling = 1.0;
  bool include_bias = true;
  unsigned int seed = 0;

  RandomSparseReservoir res1(n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, include_bias, seed);

  SECTION("After std::rand() is used") {
    for (int i = 0; i < 10; ++i) {
      static_cast<void>(std::rand());
    }
  }

  SECTION("After another reservoir is constructed") {
    RandomSparseReservoir other(n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, include_bias, 1);
  }

  RandomSparseReservoir res2(n_neurons, spectral_radius, sparsity, leak_rate, input_scaling, include_bias, seed);
  const Eigen::MatrixXd states1 = runSequence(res1);
  const Eigen::MatrixXd states2 = runSequence(res2);
  REQUIRE((states1 - states2).cwiseAbs().maxCoeff() == 0.0); // bitwise, reported as one number
}
