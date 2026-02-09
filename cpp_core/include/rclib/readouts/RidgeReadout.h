#pragma once

#include "rclib/Readout.h"

class RidgeReadout : public Readout {
public:
  enum Solver { CHOLESKY, CONJUGATE_GRADIENT, CONJUGATE_GRADIENT_IMPLICIT };

  RidgeReadout(double alpha = 1e-8, bool include_bias = true, Solver solver = CONJUGATE_GRADIENT);

  void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) override;
  void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) override;
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override;

private:
  double alpha;
  bool include_bias;
  Solver solver;
  Eigen::MatrixXd W_out;
};
