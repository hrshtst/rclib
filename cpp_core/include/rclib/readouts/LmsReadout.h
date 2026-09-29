#pragma once

#include "rclib/Readout.h"

#include <Eigen/Dense>

class LmsReadout : public Readout {
public:
  LmsReadout(double learning_rate = 0.01, bool include_bias = true);

  void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) override;
  void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) override;
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override;
  // Keyed on `initialized`, not on W_out: a failed fit() clears the flag but
  // leaves the previous matrix in place, and the readout is then unfitted.
  int getInputDim() const override { return initialized ? static_cast<int>(W_out.rows()) - (include_bias ? 1 : 0) : 0; }

  double getLearningRate() const { return learning_rate; }
  bool getIncludeBias() const { return include_bias; }

private:
  double learning_rate;
  bool include_bias;

  Eigen::MatrixXd W_out; // Weight matrix
  bool initialized;
};
