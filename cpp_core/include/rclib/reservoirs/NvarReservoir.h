#pragma once

#include "rclib/Reservoir.h"

#include <Eigen/Dense>
#include <vector>

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
  static constexpr int max_feature_count = 1000000;
  // Monomial generation recurses to a depth of polynomial_order; bound it so a
  // pathological value cannot overflow the stack. NVAR orders are small (~2-3)
  // in practice, so this ceiling is generous.
  static constexpr int max_polynomial_order = 32;

  int num_lags;
  int polynomial_order;
  int input_dim;
  bool initialized;
  Eigen::MatrixXd state;
  Eigen::MatrixXd past_inputs;
};
