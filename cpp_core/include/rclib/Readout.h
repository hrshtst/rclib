#pragma once

#include <Eigen/Dense>

class Readout {
public:
  virtual ~Readout() = default;
  virtual void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) = 0;
  virtual void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) = 0;
  virtual Eigen::MatrixXd predict(const Eigen::MatrixXd &states) = 0;
};
