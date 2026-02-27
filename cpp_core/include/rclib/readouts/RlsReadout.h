#pragma once

#include "rclib/Readout.h"

#include <Eigen/Dense>

class RlsReadout : public Readout {
public:
  enum Solver { RANK1_UPDATE, RANK_K_UPDATE };

  RlsReadout(double lambda = 0.99, double delta = 1.0, bool include_bias = true, Solver solver = RANK1_UPDATE);

  void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) override;
  void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) override;
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override;

  Solver getSolver() const { return solver; }

private:
  double lambda;
  double delta;
  bool include_bias;
  Solver solver;

  Eigen::MatrixXd W_out; // Weight matrix
  Eigen::MatrixXd P;     // Inverse covariance matrix
  bool initialized;

  // Pre-allocated temporaries to avoid reallocation in partialFit
  Eigen::VectorXd x_aug;
  Eigen::VectorXd k;
  Eigen::VectorXd Px;
  Eigen::RowVectorXd xP;
};
