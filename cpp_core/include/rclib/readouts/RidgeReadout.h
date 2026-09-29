#pragma once

#include "rclib/Readout.h"

#include <memory>

class BinaryReader;
class BinaryWriter;

class RidgeReadout : public Readout {
public:
  enum Solver { AUTO, CHOLESKY, DUAL_CHOLESKY, CONJUGATE_GRADIENT, CONJUGATE_GRADIENT_IMPLICIT };

  RidgeReadout(double alpha = 1e-8, bool include_bias = true, Solver solver = AUTO, double tolerance = 1e-6);

  void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) override;
  void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) override;
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override;

  int getInputDim() const override {
    return W_out.size() == 0 ? 0 : static_cast<int>(W_out.rows()) - (include_bias ? 1 : 0);
  }

  double getAlpha() const { return alpha; }
  double getTolerance() const { return tolerance; }
  Solver getSolver() const { return solver; }
  Solver getEffectiveSolver() const { return effective_solver; }
  bool getIncludeBias() const { return include_bias; }

  /// The fitted readout weights: a (n_features [+ 1], n_outputs) matrix whose
  /// last row is the bias term when include_bias is true. Read-only; throws
  /// before fit.
  const Eigen::MatrixXd &getWeights() const;

  /// Writes the hyperparameters and the fitted state in the model file format;
  /// the type tag is written by Model. Throws SerializationError.
  void save(BinaryWriter &writer) const;
  /// Reads a payload written by save(). Throws SerializationError on invalid input.
  static std::shared_ptr<RidgeReadout> load(BinaryReader &reader);
  /// Checks that the stored matrices have consistent shapes. Saving and loading
  /// both run it. Throws SerializationError.
  void checkConsistency() const;

private:
  double alpha;
  bool include_bias;
  Solver solver;
  Solver effective_solver;
  double tolerance;
  Eigen::MatrixXd W_out;
};
