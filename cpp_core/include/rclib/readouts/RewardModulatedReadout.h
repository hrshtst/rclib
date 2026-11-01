#pragma once

#include "rclib/Readout.h"
#include "rclib/learning/RewardModulatedHebbian.h"
#include "rclib/learning/RewardModulatedLearner.h"

#include <Eigen/Dense>
#include <optional>

/// A linear readout that learns from rewards instead of targets. Each exploring step adds
/// exploration noise to the output, z = x W (+ bias row) + noise, and records the eligibility of
/// the rule; reinforce() then changes W in proportion to the eligibility and the reward
/// modulator. W has shape (n_features [+ 1], n_outputs) with the bias row last, and starts at
/// zero at the first act(), which fixes the feature width, unless setWeights() sets it first.
class RewardModulatedReadout : public Readout, public RewardModulatedLearner {
public:
  /// Throws std::invalid_argument if n_outputs is not positive or the rule is invalid.
  RewardModulatedReadout(int n_outputs, const RewardModulatedHebbianRule &rule, bool include_bias = true);

  /// Throws std::logic_error: this readout learns from rewards, through act() and reinforce().
  void fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) override;
  /// Throws std::logic_error, as fit() does.
  void partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) override;
  /// The noise-free output x W. Throws std::runtime_error before the weights exist.
  Eigen::MatrixXd predict(const Eigen::MatrixXd &states) override;
  int getInputDim() const override;

  /// Computes the output of each row in order. With `explore`, adds exploration noise and
  /// records each row's eligibility; without it, the output is noise-free and no learning state
  /// changes. Rows must be finite (std::invalid_argument, nothing changes). A row whose output
  /// or eligibility is not finite throws std::runtime_error; earlier rows stay applied and the
  /// failing row changes no learning state.
  Eigen::MatrixXd act(const Eigen::MatrixXd &states, bool explore = true);
  /// Explores with the given noise, (rows, n_outputs), instead of drawing it.
  Eigen::MatrixXd act(const Eigen::MatrixXd &states, const Eigen::MatrixXd &noise);

  /// The output of one row, with the row staged for commitRow() when exploring. Used by Model.
  Eigen::RowVectorXd stageAct(const Eigen::RowVectorXd &state, bool explore);
  Eigen::RowVectorXd stageAct(const Eigen::RowVectorXd &state, const Eigen::RowVectorXd &noise);

  /// The weights, (n_features [+ 1], n_outputs), bias row last. Throws std::runtime_error before
  /// the first act() or setWeights().
  const Eigen::MatrixXd &getWeights() const;
  /// Replaces the weights, for example with a random or supervised start. Before the first act()
  /// it fixes the feature width; afterwards the shape must stay. Throws std::invalid_argument for
  /// a wrong shape or non-finite values.
  void setWeights(const Eigen::MatrixXd &weights);

  int getNOutputs() const { return n_outputs; }
  bool getIncludeBias() const { return include_bias; }
  const RewardModulatedHebbianRule &getRule() const { return learning.getRule(); }
  /// The learning state: eligibility, baselines, update count and more.
  const RewardModulatedHebbian &getLearning() const { return learning; }

  void commitRow() override { learning.commitRow(); }
  void discardRow() override { learning.discardRow(); }
  void checkReady() const override { learning.checkReady(); }
  void prepareReinforce(double reward, int key, std::optional<double> baseline) override;
  void prepareModulate(double modulator) override;
  void commitUpdate() override;
  void discardUpdate() override;
  void clearEligibility() override { learning.clearEligibility(); }
  void resetActivity() override { learning.resetActivity(); }

private:
  bool isInitialized() const { return learning.isInitialized(); }
  void initialize(Eigen::Index n_features);
  void checkStates(const Eigen::MatrixXd &states) const;
  Eigen::RowVectorXd withBias(const Eigen::RowVectorXd &state) const;
  Eigen::RowVectorXd stage(const Eigen::RowVectorXd &state, const Eigen::RowVectorXd &noise, bool explore);
  void prepareWeights();

  int n_outputs;
  bool include_bias;
  RewardModulatedHebbian learning;
  Eigen::MatrixXd W; // (n_features [+ 1], n_outputs) once initialized
  std::optional<Eigen::MatrixXd> prepared_weights;
};
