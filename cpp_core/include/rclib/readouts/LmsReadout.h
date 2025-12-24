#pragma once

#include "rclib/Readout.h"

#include <Eigen/Dense>

class LmsReadout : public Readout {
public:
  LmsReadout(double learning_rate = 0.01, bool include_bias = true);

  void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) override;
  void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) override;
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override;

private:
  double learning_rate;
  bool include_bias;

  Eigen::MatrixXd W_out; // Weight matrix
  bool initialized;
};
