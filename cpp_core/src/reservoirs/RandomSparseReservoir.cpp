#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <vector>

// Helper function to generate a sparse random matrix
Eigen::SparseMatrix<double> generate_sparse_random_matrix(int size, double sparsity, std::mt19937 &gen) {
  std::vector<Eigen::Triplet<double>> triplets;
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  std::uniform_int_distribution<> pos_dis(0, size - 1);

  int num_non_zero = static_cast<int>(size * size * sparsity);
  triplets.reserve(num_non_zero);

  for (int k = 0; k < num_non_zero; ++k) {
    triplets.push_back(Eigen::Triplet<double>(pos_dis(gen), pos_dis(gen), dis(gen)));
  }

  Eigen::SparseMatrix<double> mat(size, size);
  mat.setFromTriplets(triplets.begin(), triplets.end());
  return mat;
}

// Function to find the largest eigenvalue of a sparse matrix using power iteration
double largest_eigenvalue(const Eigen::SparseMatrix<double> &mat, int iterations = 100) {
  if (mat.rows() == 0) {
    return 0.0;
  }
  Eigen::VectorXd b_k = Eigen::VectorXd::Random(mat.rows());
  for (int i = 0; i < iterations; ++i) {
    Eigen::VectorXd b_k1 = mat * b_k;
    if (b_k1.norm() < 1e-9) {
      return 0.0; // Matrix is likely zero
    }
    b_k = b_k1.normalized();
  }
  return (mat * b_k).norm();
}

RandomSparseReservoir::RandomSparseReservoir(int n_neurons, double spectral_radius, double sparsity, double leak_rate,
                                             double input_scaling, bool include_bias, unsigned int seed)
    : n_neurons(n_neurons), spectral_radius(spectral_radius), sparsity(sparsity), leak_rate(leak_rate),
      input_scaling(input_scaling), include_bias(include_bias), W_in_initialized(false) {

  state = Eigen::MatrixXd::Zero(1, n_neurons);

  std::mt19937 gen(seed);
  W_res = generate_sparse_random_matrix(n_neurons, sparsity, gen);

  if (spectral_radius > 0) {
    double max_eigenvalue = largest_eigenvalue(W_res);
    if (max_eigenvalue > 1e-9) {
      W_res = W_res * (spectral_radius / max_eigenvalue);
    }
  }

  if (include_bias) {
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    bias = Eigen::RowVectorXd::NullaryExpr(n_neurons, [&]() { return dis(gen); });
  } else {
    bias = Eigen::RowVectorXd::Zero(n_neurons);
  }

  // Store the generator state or re-seed later for W_in if needed.
  // For now, let's re-create the generator with the same seed + offset for W_in to keep it deterministic.
  this->seed = seed;
}

void RandomSparseReservoir::initialize_W_in(int input_dim) {
  // Use a different seed sequence for W_in based on the original seed
  std::mt19937 gen(seed + 1);
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  W_in = Eigen::MatrixXd::NullaryExpr(input_dim, n_neurons, [&]() { return dis(gen); }) * input_scaling;
  W_in_initialized = true;
}

Eigen::MatrixXd RandomSparseReservoir::advance(const Eigen::MatrixXd &input) {
  if (!W_in_initialized) {
    initialize_W_in(input.cols());
  }

  state = (1 - leak_rate) * state + leak_rate * (input * W_in + state * W_res + bias).array().tanh().matrix();
  return state;
}

void RandomSparseReservoir::resetState() { state.setZero(); }

const Eigen::MatrixXd &RandomSparseReservoir::getState() const { return state; }
