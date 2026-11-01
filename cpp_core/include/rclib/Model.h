#pragma once

#include "Readout.h"
#include "Reservoir.h"

#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class RewardModulatedLearner;

class Model {
public:
  void addReservoir(std::shared_ptr<Reservoir> res, std::string connection_type = "serial");
  void setReadout(std::shared_ptr<Readout> readout);
  /// Resets the reservoirs, runs them through `inputs`, drops the first `washout_len`
  /// states and fits the readout on the rest. Input of a width the reservoirs are not
  /// locked to by earlier input is rejected before they are reset.
  void fit(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, int washout_len = 0);
  /// Fits the readout once on several independent sequences, such as episodes. Each
  /// sequence starts from reset reservoirs and loses its own first `washout_len` states;
  /// the remaining states and targets of all sequences are stacked for a single
  /// readout fit, so nothing carries over from one sequence to the next. With one
  /// sequence this equals fit(). Every sequence is checked as fit() checks its input,
  /// including the input width the reservoirs are locked to, and all must share the
  /// input and target widths; these checks run before the reservoirs change, and their
  /// errors name the sequence index. Afterwards the reservoirs hold their states at the
  /// end of the last sequence.
  void fitSequences(const std::vector<Eigen::MatrixXd> &inputs, const std::vector<Eigen::MatrixXd> &targets,
                    int washout_len = 0);
  void partialFit(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target);
  Eigen::MatrixXd predict(const Eigen::MatrixXd &inputs, bool reset_state_before_predict = true);
  Eigen::MatrixXd predictOnline(const Eigen::MatrixXd &input);
  /// Advances the reservoirs through `prime_inputs` (if any), then generates `n_steps`
  /// outputs, feeding each output back as the next input. Every generated output,
  /// including the last, advances the reservoirs, so a following call with empty
  /// `prime_inputs` continues the sequence: generating a steps and then b steps
  /// equals generating a + b steps. With n_steps == 0 no output is fed back. Feeding
  /// back an output whose width differs from the input width the reservoirs are
  /// locked to throws std::invalid_argument, also for n_steps == 1.
  Eigen::MatrixXd predictGenerative(const Eigen::MatrixXd &prime_inputs, int n_steps);
  /// Resets the reservoir states and the activity the reward-modulated components filter,
  /// such as their post-synaptic averages. Weights, eligibility and baselines stay.
  void resetReservoirs();

  /// Steps the model through `inputs` row by row to learn from rewards, and returns each row's
  /// output: the readout's output, or the reservoir states when the model has no readout. With
  /// `explore`, reward-modulated components add exploration noise and record each row's
  /// eligibility; without it every step is noise-free and no learning state changes, which for
  /// a model with a readout equals predictOnline(). Everything is checked before the model
  /// changes: the inputs must be finite and fit the widths the reservoirs and the readout are
  /// locked to (std::invalid_argument), and no reservoir object may be listed twice
  /// (std::logic_error). A row whose output, from any readout or from the reservoirs, or
  /// eligibility is not finite throws std::runtime_error, exploring or not: earlier rows stay
  /// applied, the failing row changes no component's learning state, and the reservoir states
  /// are unspecified until resetReservoirs().
  Eigen::MatrixXd act(const Eigen::MatrixXd &inputs, bool explore = true);
  /// Updates every reward-modulated component for a reward: each compares it with its baseline
  /// for `key`, or with `baseline` when given (see RewardModulatedHebbianRule). The update
  /// applies to all components or to none: std::invalid_argument for a non-finite reward or
  /// baseline, std::logic_error if the model has no reward-modulated component, lists one twice,
  /// or one has not explored since its last update, and std::runtime_error if a modulator or new
  /// weights would not be finite.
  void reinforce(double reward, int key = 0, std::optional<double> baseline = std::nullopt);
  /// Updates every reward-modulated component with the modulator given directly, all or none.
  void modulate(double modulator);
  /// Zeroes the eligibility of every reward-modulated component.
  void clearEligibility();

  std::shared_ptr<Reservoir> getReservoir(size_t index) const;
  std::shared_ptr<Readout> getReadout() const;
  bool hasReadout() const { return readout != nullptr; }
  size_t getNumReservoirs() const { return reservoirs.size(); }
  const std::string &getConnectionType() const { return connection_type; }

  /// Saves the configuration, weights, current reservoir states and the learning state of
  /// reward-modulated components in the rclib model format, so that a loaded model continues
  /// exactly. A model without a readout is saved too. Throws SerializationError, before writing
  /// anything, if the model lacks a reservoir, holds the same reservoir object twice, contains a
  /// component type other than the built-in ones, or is internally inconsistent.
  void save(std::ostream &os) const;
  /// Saves to a file. The model is written to a new temporary file next to `path`,
  /// which then replaces `path` in one step, so an existing file there is never left
  /// damaged: after a failure it is unchanged and the temporary file is removed.
  void save(const std::string &path) const;
  /// Loads a model written by save(). Reads from the current position of a
  /// seekable stream; the model must extend to the end of the stream. Throws
  /// SerializationError (or std::bad_alloc) on invalid input.
  static Model load(std::istream &is);
  /// Loads a model file written by save(). Throws SerializationError on failure.
  static Model load(const std::string &path);

private:
  /// How collectStates() steps each reservoir: advance(), or stageAct() exploring or not.
  enum class Step { Advance, Explore, NoiseFree };

  Eigen::MatrixXd collectStates(const Eigen::MatrixXd &inputs, Step step = Step::Advance);
  /// The reward-modulated components, after checking that none is listed twice
  /// (std::logic_error, naming `operation`).
  std::vector<RewardModulatedLearner *> learners(const char *operation) const;
  /// Throws std::logic_error if the readout learns from rewards, which `operation` cannot train.
  void checkSupervisedReadout(const char *operation) const;
  /// Prepares an update on every learner and commits it on all, or discards it on all.
  template <typename Prepare> void update(const std::vector<RewardModulatedLearner *> &targets, Prepare &&prepare);
  /// Throws std::invalid_argument, prefixed with `context`, if input `input_width`
  /// columns wide does not fit the widths the reservoirs are locked to by earlier input.
  void checkInputWidth(Eigen::Index input_width, const std::string &context) const;
  /// Resets the reservoirs, collects the states for `inputs` and drops the first `washout_len`.
  Eigen::MatrixXd collectStatesAfterWashout(const Eigen::MatrixXd &inputs, int washout_len);
  Eigen::MatrixXd collectCurrentStates() const;

  std::vector<std::shared_ptr<Reservoir>> reservoirs;
  std::shared_ptr<Readout> readout;
  std::string connection_type = "serial";
};
