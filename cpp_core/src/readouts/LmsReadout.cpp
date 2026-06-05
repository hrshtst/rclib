#include "rclib/readouts/LmsReadout.h"

#include <stdexcept>

LmsReadout::LmsReadout(double learning_rate, bool include_bias)
    : learning_rate(learning_rate), include_bias(include_bias), initialized(false) {
  if (learning_rate <= 0.0) {
    throw std::invalid_argument("learning_rate must be positive.");
  }
}

void LmsReadout::fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) {
  // LMS is an online algorithm, so fit will call partialFit repeatedly.
  // Reset the state before fitting.
  initialized = false; // Force re-initialization in partialFit

  for (int i = 0; i < states.rows(); ++i) {
    partialFit(states.row(i), targets.row(i));
  }
}

void LmsReadout::partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) {
  if (state.rows() == 0 || state.cols() == 0) {
    throw std::invalid_argument("state must be a non-empty 2D matrix.");
  }
  if (target.rows() != state.rows()) {
    throw std::invalid_argument("target must have the same number of rows as state.");
  }
  if (target.cols() == 0) {
    throw std::invalid_argument("target must have at least one column.");
  }

  int n_samples = state.rows();
  int n_in = state.cols();
  int n_features = n_in + (include_bias ? 1 : 0);

  if (!initialized) {
    // Initialize W_out
    int n_targets = target.cols();
    W_out = Eigen::MatrixXd::Zero(n_features, n_targets);
    initialized = true;
  } else if (W_out.rows() != n_features || W_out.cols() != target.cols()) {
    throw std::invalid_argument("state or target dimensions changed after LmsReadout initialization.");
  }

  Eigen::MatrixXd x;
  if (include_bias) {
    x.resize(n_samples, n_features);
    x.leftCols(n_in) = state;
    x.rightCols(1).setOnes();
  } else {
    x = state; // Shallow copy if possible, but scikit-build/eigen handles this
  }

  // LMS update equations (GEMM)
  Eigen::MatrixXd y_hat = x * W_out;
  Eigen::MatrixXd error = target - y_hat;

  // Batch update: W = W + eta * X^T * E / n_samples
  W_out.noalias() += (learning_rate / n_samples) * x.transpose() * error;
}

Eigen::MatrixXd LmsReadout::predict(const Eigen::MatrixXd &states) {
  if (!initialized) {
    throw std::runtime_error("LmsReadout must be fit before predict.");
  }
  if (states.rows() == 0 || states.cols() == 0) {
    throw std::invalid_argument("states must be a non-empty 2D matrix.");
  }
  if (states.cols() != W_out.rows() - (include_bias ? 1 : 0)) {
    throw std::invalid_argument("states column count does not match initialized LmsReadout.");
  }

  Eigen::MatrixXd X = states;
  if (include_bias) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    X.col(X.cols() - 1) = Eigen::VectorXd::Ones(X.rows());
  }
  return X * W_out;
}
