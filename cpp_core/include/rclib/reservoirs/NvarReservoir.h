#pragma once

#include "rclib/Reservoir.h"

#include <Eigen/Dense>

class NvarReservoir : public Reservoir {
public:
  NvarReservoir(int num_lags, int polynomial_order = 1);

  const Eigen::MatrixXd &advance(const Eigen::MatrixXd &input) override;
  void resetState() override;
  const Eigen::MatrixXd &getState() const override;
  int getOutputDim(int input_dim) const override;

private:
  void initialize(int input_dim);
  void appendMonomials(const Eigen::RowVectorXd &delayed, int start_index, int remaining_degree, double current_value,
                       std::vector<double> &features) const;
  static int countMonomials(int n_variables, int degree);

  int num_lags;
  int polynomial_order;
  int input_dim;
  bool initialized;
  Eigen::MatrixXd state;
  Eigen::MatrixXd past_inputs;
};
