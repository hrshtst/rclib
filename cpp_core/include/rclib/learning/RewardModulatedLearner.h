#pragma once

#include <optional>

/// A component whose weights learn from rewards: a reward-modulated readout or reservoir.
/// Model drives the two phases of its rows and updates through this interface, so that a row
/// or an update applies to every learner of a model or to none.
class RewardModulatedLearner {
public:
  virtual ~RewardModulatedLearner() = default;

  /// Applies or drops the row staged by the component's last exploring step.
  virtual void commitRow() = 0;
  virtual void discardRow() = 0;

  /// Throws std::logic_error unless an exploring row was committed since the last update.
  virtual void checkReady() const = 0;
  /// Prepares the weight change for a reward (see RewardModulatedHebbian::prepareReinforce),
  /// including the check that the new weights are finite. Changes no learning state.
  virtual void prepareReinforce(double reward, int key, std::optional<double> baseline) = 0;
  virtual void prepareModulate(double modulator) = 0;
  /// Applies the prepared update to the weights and the learning state.
  virtual void commitUpdate() = 0;
  virtual void discardUpdate() = 0;

  /// Zeroes the eligibility; the weights, baselines and update count stay.
  virtual void clearEligibility() = 0;
  /// Forgets the activity the learning rule filters, such as the post-synaptic average.
  virtual void resetActivity() = 0;

  /// Prepares and commits the update for a reward. Throws, changing nothing, as
  /// prepareReinforce does.
  void reinforce(double reward, int key = 0, std::optional<double> baseline = std::nullopt) {
    prepareReinforce(reward, key, baseline);
    commitUpdate();
  }
  /// Prepares and commits an update with the modulator given directly.
  void modulate(double modulator) {
    prepareModulate(modulator);
    commitUpdate();
  }
};
