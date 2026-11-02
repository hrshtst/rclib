#include "RandomWeights.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace random_weights {

namespace {

// Helper function to generate a sparse random matrix
Eigen::SparseMatrix<double> generate_sparse_random_matrix(int size, double sparsity, std::mt19937 &gen) {
  std::vector<Eigen::Triplet<double>> triplets;
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  std::uniform_int_distribution<> pos_dis(0, size - 1);

  // size * size can exceed INT_MAX for large reservoirs (n_neurons > ~46340),
  // so compute the entry count in 64-bit to avoid wrapping to a negative value.
  const long long total_cells = static_cast<long long>(size) * size;
  const long long num_non_zero = static_cast<long long>(static_cast<double>(total_cells) * sparsity);
  triplets.reserve(static_cast<size_t>(num_non_zero));

  for (long long k = 0; k < num_non_zero; ++k) {
    triplets.push_back(Eigen::Triplet<double>(pos_dis(gen), pos_dis(gen), dis(gen)));
  }

  Eigen::SparseMatrix<double> mat(size, size);
  mat.setFromTriplets(triplets.begin(), triplets.end());
  return mat;
}

// Function to find the spectral radius (largest eigenvalue modulus) of a sparse matrix using power iteration.
// Random sparse matrices often have a complex-conjugate dominant pair or near ties in modulus, where the norm
// after the last step does not converge. The geometric mean of the per-step growth ||A b_k|| over the second
// half of the iterations does: the first half lets the dominant eigenvalues take over.
// The start vector is drawn from gen, not Eigen's Random(), which reads the global std::rand() state.
double largest_eigenvalue(const Eigen::SparseMatrix<double> &mat, std::mt19937 &gen, int iterations = 1000) {
  if (mat.rows() == 0) {
    return 0.0;
  }
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  Eigen::VectorXd b_k = Eigen::VectorXd::NullaryExpr(mat.rows(), [&]() { return dis(gen); });
  b_k.normalize(); // leaves a zero vector unchanged
  const int burn_in = iterations / 2;
  double log_growth_sum = 0.0;
  for (int i = 0; i < iterations; ++i) {
    Eigen::VectorXd b_k1 = mat * b_k;
    const double growth = b_k1.norm();
    if (growth < 1e-9) {
      return 0.0; // Matrix is likely zero or the iterate collapsed; also keeps log() away from zero
    }
    if (i >= burn_in) {
      log_growth_sum += std::log(growth);
    }
    b_k = b_k1 / growth;
  }
  return std::exp(log_growth_sum / (iterations - burn_in));
}

// Function to find the exact spectral radius of a sparse matrix from the eigenvalues of its dense copy.
// Costs O(n^3) time and O(n^2) memory.
double dense_spectral_radius(const Eigen::SparseMatrix<double> &mat, const char *context) {
  if (mat.rows() == 0) {
    return 0.0;
  }
  const Eigen::EigenSolver<Eigen::MatrixXd> solver(Eigen::MatrixXd(mat), /*computeEigenvectors=*/false);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(std::string(context) + ": the dense eigenvalue solver did not converge.");
  }
  return solver.eigenvalues().cwiseAbs().maxCoeff();
}

} // namespace

RecurrentWeights generateRecurrentWeights(int n_neurons, double spectral_radius, double sparsity, bool include_bias,
                                          unsigned int seed, RandomSparseReservoir::SpectralRadiusMethod method,
                                          double bias_scaling, const char *context) {
  RecurrentWeights weights;
  std::mt19937 gen(seed);
  weights.W_res = generate_sparse_random_matrix(n_neurons, sparsity, gen);

  if (spectral_radius > 0) {
    double max_eigenvalue = 0.0;
    if (method == RandomSparseReservoir::DENSE) {
      max_eigenvalue = dense_spectral_radius(weights.W_res, context);
    } else {
      // Its own generator: the start vector depends only on the seed, and W_res and bias keep their draws from gen.
      std::mt19937 power_iteration_gen(seed + 2);
      max_eigenvalue = largest_eigenvalue(weights.W_res, power_iteration_gen);
    }
    if (max_eigenvalue > 1e-9) {
      weights.W_res = weights.W_res * (spectral_radius / max_eigenvalue);
    }
  }
  weights.W_res.makeCompressed();

  if (include_bias) {
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    weights.bias = Eigen::RowVectorXd::NullaryExpr(n_neurons, [&]() { return dis(gen); });
    // Scaled after the draws, which a bias_scaling of 1 leaves bit for bit.
    weights.bias *= bias_scaling;
  } else {
    weights.bias = Eigen::RowVectorXd::Zero(n_neurons);
  }
  return weights;
}

Eigen::MatrixXd generateInputWeights(int input_dim, int n_neurons, double input_scaling, unsigned int seed) {
  // A seed sequence of its own, so W_in does not depend on how many draws W_res and the bias took.
  std::mt19937 gen(seed + 1);
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  return Eigen::MatrixXd::NullaryExpr(input_dim, n_neurons, [&]() { return dis(gen); }) * input_scaling;
}

} // namespace random_weights
