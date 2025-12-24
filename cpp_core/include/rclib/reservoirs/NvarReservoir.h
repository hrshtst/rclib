#pragma once

#include "rclib/Reservoir.h"

#include <Eigen/Dense>

class NvarReservoir : public Reservoir {
public:
  NvarReservoir(int num_lags);

  Eigen::MatrixXd advance(const Eigen::MatrixXd &input) override;
  void resetState() override;
  const Eigen::MatrixXd &getState() const override;

private:
  void initialize(int input_dim);

  int num_lags;
  int input_dim;
  bool initialized;
  Eigen::MatrixXd state;
  Eigen::MatrixXd past_inputs;
};
