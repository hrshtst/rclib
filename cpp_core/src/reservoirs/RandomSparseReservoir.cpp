#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <vector>

// Helper function to generate a sparse random matrix
Eigen::SparseMatrix<double> generate_sparse_random_matrix(int size, double sparsity) {
  std::vector<Eigen::Triplet<double>> triplets;
  std::random_device rd;
  std::mt19937 gen(rd());
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
                                             double input_scaling, bool include_bias)
    : n_neurons(n_neurons), spectral_radius(spectral_radius), sparsity(sparsity), leak_rate(leak_rate),
      input_scaling(input_scaling), include_bias(include_bias), W_in_initialized(false) {

  state = Eigen::MatrixXd::Zero(1, n_neurons);

  W_res = generate_sparse_random_matrix(n_neurons, sparsity);

  if (spectral_radius > 0) {
    double max_eigenvalue = largest_eigenvalue(W_res);
    if (max_eigenvalue > 1e-9) {
      W_res = W_res * (spectral_radius / max_eigenvalue);
    }
  }

  if (include_bias) {
    bias = Eigen::RowVectorXd::Random(n_neurons);
  } else {
    bias = Eigen::RowVectorXd::Zero(n_neurons);
  }
}

void RandomSparseReservoir::initialize_W_in(int input_dim) {
  W_in = Eigen::MatrixXd::Random(input_dim, n_neurons) * input_scaling;
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
