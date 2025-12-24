#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

TEST_CASE("RandomSparseReservoir - Constructor and Initialization", "[RandomSparseReservoir]") {
  int n_neurons = 10;
  double spectral_radius = 0.9;
  double sparsity = 0.5;
  double leak_rate = 0.1;
  bool include_bias = true;

  RandomSparseReservoir res(n_neurons, spectral_radius, sparsity, leak_rate, include_bias);

  SECTION("State is initialized to zeros") {
    REQUIRE(res.getState().rows() == 1);
    REQUIRE(res.getState().cols() == n_neurons);
    REQUIRE(res.getState().isZero(0));
  }
}

TEST_CASE("RandomSparseReservoir - State Advancement", "[RandomSparseReservoir]") {
  int n_neurons = 10;
  double spectral_radius = 0.9;
  double sparsity = 0.5;
  double leak_rate = 0.1;
  bool include_bias = true;

  RandomSparseReservoir res(n_neurons, spectral_radius, sparsity, leak_rate, include_bias);

  Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, 5); // Assuming input_dim = 5 for now

  // Advance state multiple times
  for (int i = 0; i < 10; ++i) {
    res.advance(input);
    // Check if state dimensions remain correct
    REQUIRE(res.getState().rows() == 1);
    REQUIRE(res.getState().cols() == n_neurons);
    // Check if state is not all zeros (it should change)
    REQUIRE_FALSE(res.getState().isZero(0));
  }
}

TEST_CASE("RandomSparseReservoir - State Reset", "[RandomSparseReservoir]") {
  int n_neurons = 10;
  double spectral_radius = 0.9;
  double sparsity = 0.5;
  double leak_rate = 0.1;
  bool include_bias = true;

  RandomSparseReservoir res(n_neurons, spectral_radius, sparsity, leak_rate, include_bias);

  Eigen::MatrixXd input = Eigen::MatrixXd::Random(1, 5); // Assuming input_dim = 5 for now

  // Advance state to make it non-zero
  res.advance(input);
  REQUIRE_FALSE(res.getState().isZero(0));

  // Reset state
  res.resetState();

  // Check if state is reset to zeros
  REQUIRE(res.getState().rows() == 1);
  REQUIRE(res.getState().cols() == n_neurons);
  REQUIRE(res.getState().isZero(0));
}
