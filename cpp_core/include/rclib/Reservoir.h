#pragma once

#include <Eigen/Dense>

class Reservoir {
public:
  virtual ~Reservoir() = default;
  virtual Eigen::MatrixXd advance(const Eigen::MatrixXd &input) = 0;
  virtual void resetState() = 0;
  virtual const Eigen::MatrixXd &getState() const = 0;
};
