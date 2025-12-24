#pragma once

#include "rclib/Readout.h"

#include <Eigen/Dense>

class RlsReadout : public Readout {
public:
  RlsReadout(double lambda = 0.99, double delta = 1.0, bool include_bias = true);

  void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) override;
  void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) override;
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override;

private:
  double lambda;
  double delta;
  bool include_bias;

  Eigen::MatrixXd W_out; // Weight matrix
  Eigen::MatrixXd P;     // Inverse covariance matrix
  bool initialized;

  // Pre-allocated temporaries to avoid reallocation in partialFit
  Eigen::VectorXd x_aug;
  Eigen::VectorXd k;
  Eigen::VectorXd Px;
  Eigen::RowVectorXd xP;
};
