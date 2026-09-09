#pragma once

#include "rclib/Readout.h"

class RidgeReadout : public Readout {
public:
  enum Solver { AUTO, CHOLESKY, DUAL_CHOLESKY, CONJUGATE_GRADIENT, CONJUGATE_GRADIENT_IMPLICIT };

  RidgeReadout(double alpha = 1e-8, bool include_bias = true, Solver solver = AUTO, double tolerance = 1e-6);

  void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) override;
  void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) override;
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override;

  Solver getSolver() const { return solver; }
  Solver getEffectiveSolver() const { return effective_solver; }
  bool getIncludeBias() const { return include_bias; }

  /// The fitted readout weights: a (n_features [+ 1], n_outputs) matrix whose
  /// last row is the bias term when include_bias is true. Read-only; throws
  /// before fit.
  const Eigen::MatrixXd &getWeights() const;

private:
  double alpha;
  bool include_bias;
  Solver solver;
  Solver effective_solver;
  double tolerance;
  Eigen::MatrixXd W_out;
};
