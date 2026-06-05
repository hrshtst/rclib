#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

class MinimalReservoir : public Reservoir {
public:
  const Eigen::MatrixXd &advance(const Eigen::MatrixXd & /*input*/) override { return state; }
  void resetState() override { state.setZero(); }
  const Eigen::MatrixXd &getState() const override { return state; }

private:
  Eigen::MatrixXd state = Eigen::MatrixXd::Zero(1, 3);
};

TEST_CASE("Reservoir - default output dimension", "[Reservoir]") {
  MinimalReservoir res;
  REQUIRE(res.getOutputDim(1) == 3);
}

TEST_CASE("RandomSparseReservoir - Constructor and Initialization", "[RandomSparseReservoir]") {
  int n_neurons = 10;
  double spectral_radius = 0.9;
  double sparsity = 0.5;
  double leak_rate = 0.1;
  bool include_bias = true;

  RandomSparseReservoir res(n_neurons, spectral_radius, sparsity, leak_rate, 1.0, include_bias);

  SECTION("State is initialized to zeros") {
    REQUIRE(res.getState().rows() == 1);
    REQUIRE(res.getState().cols() == n_neurons);
    REQUIRE(res.getState().isZero(0));
  }
}

TEST_CASE("RandomSparseReservoir - large n_neurons does not overflow entry count", "[RandomSparseReservoir]") {
  // n_neurons^2 exceeds INT_MAX here (46341^2 > 2^31); the non-zero entry count
  // must be computed in 64-bit, otherwise it wraps negative and construction fails.
  // spectral_radius = 0 skips power iteration so the test stays fast.
  REQUIRE_NOTHROW(RandomSparseReservoir(46341, 0.0, 0.0001, 0.5, 1.0, false, 42));
}

TEST_CASE("RandomSparseReservoir - State Advancement", "[RandomSparseReservoir]") {
  int n_neurons = 10;
  double spectral_radius = 0.9;
  double sparsity = 0.5;
  double leak_rate = 0.1;
  bool include_bias = true;

  RandomSparseReservoir res(n_neurons, spectral_radius, sparsity, leak_rate, 1.0, include_bias);

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

  RandomSparseReservoir res(n_neurons, spectral_radius, sparsity, leak_rate, 1.0, include_bias);

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
