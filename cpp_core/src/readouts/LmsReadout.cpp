#include "rclib/readouts/LmsReadout.h"

#include <stdexcept>

LmsReadout::LmsReadout(double learning_rate, bool include_bias)
    : learning_rate(learning_rate), include_bias(include_bias), initialized(false) {}

void LmsReadout::fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) {
  // LMS is an online algorithm, so fit will call partialFit repeatedly.
  // Reset the state before fitting.
  initialized = false; // Force re-initialization in partialFit

  for (int i = 0; i < states.rows(); ++i) {
    partialFit(states.row(i), targets.row(i));
  }
}

void LmsReadout::partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) {
  int n_samples = state.rows();
  int n_in = state.cols();
  int n_features = n_in + (include_bias ? 1 : 0);

  if (!initialized) {
    // Initialize W_out
    int n_targets = target.cols();
    W_out = Eigen::MatrixXd::Zero(n_features, n_targets);
    initialized = true;
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
  Eigen::MatrixXd X = states;
  if (include_bias) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    X.col(X.cols() - 1) = Eigen::VectorXd::Ones(X.rows());
  }
  return X * W_out;
}
