#include "rclib/readouts/RlsReadout.h"

#include <stdexcept>

RlsReadout::RlsReadout(double lambda, double delta, bool include_bias)
    : lambda(lambda), delta(delta), include_bias(include_bias), initialized(false) {}

void RlsReadout::fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) {
  // RLS is an online algorithm, so fit will call partialFit repeatedly.
  // Reset the state before fitting.
  initialized = false; // Force re-initialization in partialFit

  for (int i = 0; i < states.rows(); ++i) {
    partialFit(states.row(i), targets.row(i));
  }
}

void RlsReadout::partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) {
  int n_in = state.cols();
  int n_features = n_in + (include_bias ? 1 : 0);

  if (!initialized) {
    // Initialize W_out and P
    int n_targets = target.cols();

    W_out = Eigen::MatrixXd::Zero(n_features, n_targets);
    P = (1.0 / delta) * Eigen::MatrixXd::Identity(n_features, n_features);

    // Pre-allocate buffers
    x_aug.resize(n_features);
    k.resize(n_features);
    Px.resize(n_features);

    initialized = true;
  }

  // Prepare input vector x_aug
  x_aug.head(n_in) = state.row(0).transpose();
  if (include_bias) {
    x_aug(n_in) = 1.0;
  }

  // RLS update equations

  // 1. Compute Px = P * x using symmetry (Upper triangle)
  // P is symmetric, so we use selfadjointView to optimize multiplication
  Px.noalias() = P.selfadjointView<Eigen::Upper>() * x_aug;

  // 2. Compute denominator = lambda + x^T * Px
  double denominator = lambda + x_aug.dot(Px);

  // 3. Compute Kalman gain vector k = Px / denominator
  k = Px / denominator;

  // 4. Compute prediction y_hat = x^T * W_out and error
  Eigen::MatrixXd error = target - (x_aug.transpose() * W_out);

  // 5. Update weights: W_out = W_out + k * error
  W_out.noalias() += k * error;

  // 6. Update P: P = (1/lambda) * (P - (Px * Px^T) / denominator)
  // We exploit symmetry and rank-1 update structure.
  // First scale P (only upper triangle needed)
  P.triangularView<Eigen::Upper>() *= (1.0 / lambda);

  // Then apply rank-1 update: P -= alpha * v * v^T
  // alpha = 1 / (lambda * denominator)
  // v = Px
  P.selfadjointView<Eigen::Upper>().rankUpdate(Px, -1.0 / (lambda * denominator));
}

Eigen::MatrixXd RlsReadout::predict(const Eigen::MatrixXd &states) {
  Eigen::MatrixXd X = states;
  if (include_bias) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    X.col(X.cols() - 1) = Eigen::VectorXd::Ones(X.rows());
  }
  return X * W_out;
}
