#pragma once

#include "Readout.h"
#include "Reservoir.h"

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

class Model {
public:
  void addReservoir(std::shared_ptr<Reservoir> res, std::string connection_type = "serial");
  void setReadout(std::shared_ptr<Readout> readout);
  /// Resets the reservoirs, runs them through `inputs`, drops the first `washout_len`
  /// states and fits the readout on the rest.
  void fit(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, int washout_len = 0);
  /// Fits the readout once on several independent sequences, such as episodes. Each
  /// sequence starts from reset reservoirs and loses its own first `washout_len` states;
  /// the remaining states and targets of all sequences are stacked for a single
  /// readout fit, so nothing carries over from one sequence to the next. With one
  /// sequence this equals fit(). Every sequence is checked as fit() checks its input,
  /// and all must share the input and target widths; these checks run before the
  /// reservoirs change, and their errors name the sequence index. Afterwards the
  /// reservoirs hold their states at the end of the last sequence.
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
  void resetReservoirs();

  std::shared_ptr<Reservoir> getReservoir(size_t index) const;
  std::shared_ptr<Readout> getReadout() const;
  size_t getNumReservoirs() const { return reservoirs.size(); }
  const std::string &getConnectionType() const { return connection_type; }

  /// Saves the configuration, weights and current reservoir states in the rclib
  /// model format. Throws SerializationError, before writing anything, if the model
  /// lacks a reservoir or readout, holds the same reservoir object twice, contains a
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
  Eigen::MatrixXd collectStates(const Eigen::MatrixXd &inputs);
  /// Resets the reservoirs, collects the states for `inputs` and drops the first `washout_len`.
  Eigen::MatrixXd collectStatesAfterWashout(const Eigen::MatrixXd &inputs, int washout_len);
  Eigen::MatrixXd collectCurrentStates() const;

  std::vector<std::shared_ptr<Reservoir>> reservoirs;
  std::shared_ptr<Readout> readout;
  std::string connection_type = "serial";
};
