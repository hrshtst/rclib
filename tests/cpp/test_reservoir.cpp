#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "rclib/Serialization.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <catch2/catch_all.hpp>
#include <limits>
#include <sstream>
#include <stdexcept>

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

TEST_CASE("RandomSparseReservoir - rejects input width changes after initialization", "[RandomSparseReservoir]") {
  RandomSparseReservoir res(10, 0.9, 0.5, 0.1, 1.0, true);

  // The first input locks W_in to three columns.
  res.advance(Eigen::MatrixXd::Random(1, 3));
  const Eigen::MatrixXd state_before = res.getState();

  REQUIRE_THROWS_AS(res.advance(Eigen::MatrixXd::Random(1, 4)), std::invalid_argument);
  REQUIRE_THROWS_AS(res.advance(Eigen::MatrixXd::Random(1, 2)), std::invalid_argument);
  // A rejected input leaves the state untouched.
  REQUIRE(res.getState() == state_before);
  REQUIRE_NOTHROW(res.advance(Eigen::MatrixXd::Random(1, 3)));
}

TEST_CASE("RandomSparseReservoir - rejects non-finite hyperparameters", "[RandomSparseReservoir]") {
  const double bad = GENERATE(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity());
  REQUIRE_THROWS_AS(RandomSparseReservoir(10, bad, 0.5, 0.5, 1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(RandomSparseReservoir(10, 0.9, bad, 0.5, 1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(RandomSparseReservoir(10, 0.9, 0.5, bad, 1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(RandomSparseReservoir(10, 0.9, 0.5, 0.5, bad), std::invalid_argument);
}

TEST_CASE("RandomSparseReservoir - rejects an unknown spectral_radius_method", "[RandomSparseReservoir]") {
  const auto unknown = static_cast<RandomSparseReservoir::SpectralRadiusMethod>(2);
  REQUIRE_THROWS_AS(RandomSparseReservoir(10, 0.9, 0.5, 0.5, 1.0, false, 42, unknown), std::invalid_argument);
}

TEST_CASE("RandomSparseReservoir - configuration getters and input width", "[RandomSparseReservoir]") {
  RandomSparseReservoir res(10, 0.9, 0.5, 0.25, 2.0, true, 7);
  REQUIRE(res.getNNeurons() == 10);
  REQUIRE(res.getSpectralRadius() == 0.9);
  REQUIRE(res.getSparsity() == 0.5);
  REQUIRE(res.getLeakRate() == 0.25);
  REQUIRE(res.getInputScaling() == 2.0);
  REQUIRE(res.getIncludeBias());
  REQUIRE(res.getSeed() == 7U);
  REQUIRE(res.getSpectralRadiusMethod() == RandomSparseReservoir::POWER_ITERATION);
  REQUIRE(
      RandomSparseReservoir(10, 0.9, 0.5, 0.25, 2.0, true, 7, RandomSparseReservoir::DENSE).getSpectralRadiusMethod() ==
      RandomSparseReservoir::DENSE);

  REQUIRE(res.getInputDim() == 0);
  res.advance(Eigen::MatrixXd::Random(1, 3));
  REQUIRE(res.getInputDim() == 3);
  res.resetState();
  REQUIRE(res.getInputDim() == 3);
}

namespace {

// The scaled W_res, read back from the reservoir's payload, where it follows i32 n_neurons, four f64
// hyperparameters, bool include_bias, u32 seed and u8 spectral_radius_method
// (docs/development/model_file_format.md).
Eigen::SparseMatrix<double> reservoirWeights(const RandomSparseReservoir &res) {
  std::stringstream buffer;
  BinaryWriter writer(buffer);
  res.save(writer);
  BinaryReader reader(buffer);
  reader.readInt();
  for (int i = 0; i < 4; ++i) {
    reader.readDouble();
  }
  reader.readBool();
  reader.readUInt();
  reader.readU8();
  return reader.readSparse();
}

double denseSpectralRadius(const Eigen::SparseMatrix<double> &matrix) {
  const Eigen::EigenSolver<Eigen::MatrixXd> solver(Eigen::MatrixXd(matrix), /*computeEigenvectors=*/false);
  return solver.eigenvalues().cwiseAbs().maxCoeff();
}

} // namespace

TEST_CASE("RandomSparseReservoir - W_res has the requested spectral radius", "[RandomSparseReservoir]") {
  // Most of these matrices have a complex dominant pair or near ties in modulus, where the norm after
  // plain power iteration missed the target by up to 6% at these seeds.
  const int n_neurons = GENERATE(100, 300);
  const unsigned int seed = GENERATE(range(0U, 5U));
  const double spectral_radius = 0.9;

  RandomSparseReservoir res(n_neurons, spectral_radius, 0.1, 0.5, 1.0, true, seed);
  CAPTURE(n_neurons, seed);
  REQUIRE_THAT(denseSpectralRadius(reservoirWeights(res)), Catch::Matchers::WithinRel(spectral_radius, 1e-2));
}

TEST_CASE("RandomSparseReservoir - DENSE gives W_res the requested spectral radius exactly",
          "[RandomSparseReservoir]") {
  const int n_neurons = GENERATE(100, 300);
  const unsigned int seed = GENERATE(range(0U, 5U));
  const double spectral_radius = 0.9;

  RandomSparseReservoir res(n_neurons, spectral_radius, 0.1, 0.5, 1.0, true, seed, RandomSparseReservoir::DENSE);
  CAPTURE(n_neurons, seed);
  REQUIRE_THAT(denseSpectralRadius(reservoirWeights(res)), Catch::Matchers::WithinRel(spectral_radius, 1e-10));
}

TEST_CASE("RandomSparseReservoir - W_res without nonzero eigenvalues is left unscaled", "[RandomSparseReservoir]") {
  const auto method = GENERATE(RandomSparseReservoir::POWER_ITERATION, RandomSparseReservoir::DENSE);
  Eigen::SparseMatrix<double> W_res;

  SECTION("Empty W_res") {
    W_res = reservoirWeights(RandomSparseReservoir(10, 0.9, 0.0, 0.5, 1.0, true, 1, method));
    REQUIRE(W_res.nonZeros() == 0);
  }

  SECTION("Nilpotent W_res, where the power iterate collapses to zero") {
    // A single off-diagonal entry: W_res * W_res = 0.
    W_res = reservoirWeights(RandomSparseReservoir(10, 0.9, 0.01, 0.5, 1.0, true, 1, method));
    REQUIRE(W_res.nonZeros() == 1);
    REQUIRE(Eigen::MatrixXd(W_res * W_res).isZero(0));
  }

  // Unscaled entries keep their draws from [-1, 1].
  REQUIRE(Eigen::MatrixXd(W_res).allFinite());
  REQUIRE(Eigen::MatrixXd(W_res).cwiseAbs().maxCoeff() <= 1.0);
}
