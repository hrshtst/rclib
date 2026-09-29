#pragma once

#include <Eigen/Dense>

class Reservoir {
public:
  virtual ~Reservoir() = default;
  virtual const Eigen::MatrixXd &advance(const Eigen::MatrixXd &input) = 0;
  virtual void resetState() = 0;
  virtual const Eigen::MatrixXd &getState() const = 0;
  virtual int getOutputDim(int /*input_dim*/) const { return static_cast<int>(getState().cols()); }
  /// Input width the reservoir is locked to after its first advance(); 0 while
  /// any width is still accepted (or when the type does not track it).
  virtual int getInputDim() const { return 0; }
};
