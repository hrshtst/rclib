#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
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
