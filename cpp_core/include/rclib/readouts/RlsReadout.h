#pragma once

#include "rclib/Readout.h"

#include <Eigen/Dense>
#include <memory>

class BinaryReader;
class BinaryWriter;

class RlsReadout : public Readout {
public:
  enum Solver { RANK1_UPDATE, RANK_K_UPDATE };

  RlsReadout(double lambda = 0.99, double delta = 1.0, bool include_bias = true, Solver solver = RANK1_UPDATE);

  void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) override;
  void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) override;
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override;

  // Keyed on `initialized`, not on W_out: a failed fit() clears the flag but
  // leaves the previous matrices in place, and the readout is then unfitted.
  int getInputDim() const override { return initialized ? static_cast<int>(W_out.rows()) - (include_bias ? 1 : 0) : 0; }

  double getLambda() const { return lambda; }
  double getDelta() const { return delta; }
  bool getIncludeBias() const { return include_bias; }
  Solver getSolver() const { return solver; }

  /// Writes the hyperparameters and the fitted state in the model file format;
  /// the type tag is written by Model. Throws SerializationError.
  void save(BinaryWriter &writer) const;
  /// Reads a payload written by save(). Throws SerializationError on invalid input.
  static std::shared_ptr<RlsReadout> load(BinaryReader &reader);
  /// Checks that the stored matrices have consistent shapes. Saving and loading
  /// both run it. Throws SerializationError.
  void checkConsistency() const;

private:
  double lambda;
  double delta;
  bool include_bias;
  Solver solver;

  Eigen::MatrixXd W_out; // Weight matrix
  Eigen::MatrixXd P;     // Inverse covariance matrix
  bool initialized;

  // Pre-allocated temporaries to avoid reallocation in partialFit
  Eigen::VectorXd x_aug;
  Eigen::VectorXd k;
  Eigen::VectorXd Px;
  Eigen::RowVectorXd xP;
};
