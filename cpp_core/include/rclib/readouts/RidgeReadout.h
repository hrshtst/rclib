#pragma once

#include "rclib/Readout.h"

class RidgeReadout : public Readout {
public:
  RidgeReadout(double alpha = 1e-8, bool include_bias = true);

  void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) override;
  void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) override;
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override;

private:
  double alpha;
  bool include_bias;
  Eigen::MatrixXd W_out;
};
