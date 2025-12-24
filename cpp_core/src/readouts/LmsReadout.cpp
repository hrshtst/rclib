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
  Eigen::MatrixXd x = state;
  if (include_bias) {
    x.conservativeResize(1, x.cols() + 1);
    x(0, x.cols() - 1) = 1.0;
  }

  if (!initialized) {
    // Initialize W_out
    int n_features = x.cols();
    int n_targets = target.cols();

    W_out = Eigen::MatrixXd::Zero(n_features, n_targets);
    initialized = true;
  }

  // LMS update equations
  Eigen::MatrixXd y_hat = x * W_out;
  Eigen::MatrixXd error = target - y_hat;

  W_out = W_out + learning_rate * x.transpose() * error;
}

Eigen::MatrixXd LmsReadout::predict(const Eigen::MatrixXd &states) {
  Eigen::MatrixXd X = states;
  if (include_bias) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    X.col(X.cols() - 1) = Eigen::VectorXd::Ones(X.rows());
  }
  return X * W_out;
}
