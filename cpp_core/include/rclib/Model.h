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
  void fit(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, int washout_len = 0);
  void partialFit(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target);
  Eigen::MatrixXd predict(const Eigen::MatrixXd &inputs, bool reset_state_before_predict = true);
  Eigen::MatrixXd predictOnline(const Eigen::MatrixXd &input);
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
  Eigen::MatrixXd collectCurrentStates() const;

  std::vector<std::shared_ptr<Reservoir>> reservoirs;
  std::shared_ptr<Readout> readout;
  std::string connection_type = "serial";
};
