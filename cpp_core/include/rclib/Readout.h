#pragma once

#include <Eigen/Dense>

class Readout {
public:
  virtual ~Readout() = default;
  virtual void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) = 0;
  virtual void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) = 0;
  virtual Eigen::MatrixXd predict(const Eigen::MatrixXd &states) = 0;
  /// Number of state features the readout is fitted to; 0 while unfitted (or
  /// when the type does not track it).
  virtual int getInputDim() const { return 0; }
};
