#include "rclib/readouts/RidgeReadout.h"

#include <stdexcept>

RidgeReadout::RidgeReadout(double alpha, bool include_bias) : alpha(alpha), include_bias(include_bias) {}

void RidgeReadout::fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) {
  Eigen::MatrixXd X = states;
  if (include_bias) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    X.col(X.cols() - 1) = Eigen::VectorXd::Ones(X.rows());
  }

  Eigen::MatrixXd XtX = X.transpose() * X;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());
  Eigen::MatrixXd XtY = X.transpose() * targets;

  W_out = (XtX + alpha * I).ldlt().solve(XtY);
}

void RidgeReadout::partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) {
  // RidgeReadout is a batch-trained method, so partialFit is not applicable.
  // We could throw an error or do nothing.
  // For now, let's throw an error.
  throw std::logic_error("partialFit is not implemented for RidgeReadout");
}

Eigen::MatrixXd RidgeReadout::predict(const Eigen::MatrixXd &states) {
  Eigen::MatrixXd X = states;
  if (include_bias) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    X.col(X.cols() - 1) = Eigen::VectorXd::Ones(X.rows());
  }
  return X * W_out;
}
