#include "rclib/Model.h"

#ifdef RCLIB_USE_OPENMP
#  include <omp.h>
#endif
#include <stdexcept>

void Model::addReservoir(std::shared_ptr<Reservoir> res, std::string connection_type) {
  if (!res) {
    throw std::invalid_argument("Reservoir must not be null.");
  }
  if (connection_type != "serial" && connection_type != "parallel") {
    throw std::invalid_argument("connection_type must be 'serial' or 'parallel'.");
  }
  if (!reservoirs.empty() && this->connection_type != connection_type) {
    throw std::invalid_argument("All reservoirs in a model must use the same connection_type.");
  }
  reservoirs.push_back(res);
  this->connection_type = connection_type;
}

void Model::setReadout(std::shared_ptr<Readout> readout) {
  if (!readout) {
    throw std::invalid_argument("Readout must not be null.");
  }
  this->readout = readout;
}

void Model::fit(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, int washout_len) {
  if (reservoirs.empty() || !readout) {
    throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
  }
  if (inputs.rows() == 0 || inputs.cols() == 0) {
    throw std::invalid_argument("inputs must be a non-empty 2D matrix.");
  }
  if (targets.rows() != inputs.rows()) {
    throw std::invalid_argument("targets must have the same number of rows as inputs.");
  }
  if (targets.cols() == 0) {
    throw std::invalid_argument("targets must have at least one column.");
  }

  if (washout_len < 0 || washout_len >= inputs.rows()) {
    throw std::out_of_range("washout_len must be non-negative and less than the number of input rows.");
  }

  resetReservoirs();

  Eigen::MatrixXd all_states_full = collectStates(inputs);

  // Apply washout period
  Eigen::MatrixXd fit_states = all_states_full.bottomRows(all_states_full.rows() - washout_len);
  Eigen::MatrixXd fit_targets = targets.bottomRows(targets.rows() - washout_len);

  readout->fit(fit_states, fit_targets);
}

void Model::partialFit(const Eigen::MatrixXd &input, const Eigen::MatrixXd &target) {
  if (reservoirs.empty() || !readout) {
    throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
  }
  if (input.rows() == 0 || input.cols() == 0) {
    throw std::invalid_argument("input must be a non-empty 2D matrix.");
  }
  if (target.rows() != input.rows()) {
    throw std::invalid_argument("target must have the same number of rows as input.");
  }
  if (target.cols() == 0) {
    throw std::invalid_argument("target must have at least one column.");
  }

  Eigen::MatrixXd all_states = collectStates(input);
  readout->partialFit(all_states, target);
}

Eigen::MatrixXd Model::predict(const Eigen::MatrixXd &inputs, bool reset_state_before_predict) {
  if (reservoirs.empty() || !readout) {
    throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
  }

  if (reset_state_before_predict) {
    resetReservoirs();
  }
  if (inputs.rows() == 0 || inputs.cols() == 0) {
    throw std::invalid_argument("inputs must be a non-empty 2D matrix.");
  }

  Eigen::MatrixXd all_states = collectStates(inputs);
  return readout->predict(all_states);
}

Eigen::MatrixXd Model::predictOnline(const Eigen::MatrixXd &input) {
  if (reservoirs.empty() || !readout) {
    throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
  }
  if (input.rows() == 0 || input.cols() == 0) {
    throw std::invalid_argument("input must be a non-empty 2D matrix.");
  }

  Eigen::MatrixXd all_states = collectStates(input);
  return readout->predict(all_states);
}

std::shared_ptr<Reservoir> Model::getReservoir(size_t index) const {
  if (index >= reservoirs.size()) {
    throw std::out_of_range("Reservoir index out of bounds.");
  }
  return reservoirs[index];
}

std::shared_ptr<Readout> Model::getReadout() const {
  if (!readout) {
    throw std::runtime_error("Readout not set.");
  }
  return readout;
}

Eigen::MatrixXd Model::predictGenerative(const Eigen::MatrixXd &prime_inputs, int n_steps) {
  if (reservoirs.empty() || !readout) {
    throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
  }
  if (n_steps < 0) {
    throw std::invalid_argument("n_steps must be non-negative.");
  }
  if (prime_inputs.rows() > 0 && prime_inputs.cols() == 0) {
    throw std::invalid_argument("prime_inputs must have at least one column when non-empty.");
  }

  // 1. Priming phase
  Eigen::MatrixXd last_state;
  if (prime_inputs.rows() > 0) {
    Eigen::MatrixXd states = collectStates(prime_inputs);
    last_state = states.row(states.rows() - 1);
  } else {
    last_state = collectCurrentStates(0);
  }

  // First prediction is based on the last state of the priming phase
  Eigen::MatrixXd next_input = readout->predict(last_state);

  // 2. Generative phase
  Eigen::MatrixXd generated_outputs(n_steps, next_input.cols());
  if (n_steps > 0) {
    generated_outputs.row(0) = next_input;
  }

  for (int i = 1; i < n_steps; ++i) {
    Eigen::MatrixXd current_state = collectStates(next_input);
    next_input = readout->predict(current_state);
    generated_outputs.row(i) = next_input;
  }

  return generated_outputs;
}

void Model::resetReservoirs() {
#ifdef RCLIB_USE_OPENMP
#  pragma omp parallel for
#endif
  for (int i = 0; i < static_cast<int>(reservoirs.size()); ++i) {
    reservoirs[i]->resetState();
  }
}

Eigen::MatrixXd Model::collectStates(const Eigen::MatrixXd &inputs) {
  if (connection_type == "serial") {
    Eigen::MatrixXd current_input = inputs;
    for (const auto &res : reservoirs) {
      Eigen::MatrixXd res_states(inputs.rows(), res->getOutputDim(static_cast<int>(current_input.cols())));
      for (int i = 0; i < current_input.rows(); ++i) {
        res_states.row(i) = res->advance(current_input.row(i));
      }
      current_input = res_states;
    }
    return current_input;
  }

  std::vector<Eigen::MatrixXd> reservoir_outputs(reservoirs.size());
#ifdef RCLIB_USE_OPENMP
#  pragma omp parallel for
#endif
  for (int r = 0; r < static_cast<int>(reservoirs.size()); ++r) {
    auto &res = reservoirs[static_cast<size_t>(r)];
    Eigen::MatrixXd res_states(inputs.rows(), res->getOutputDim(static_cast<int>(inputs.cols())));
    for (int i = 0; i < inputs.rows(); ++i) {
      res_states.row(i) = res->advance(inputs.row(i));
    }
    reservoir_outputs[static_cast<size_t>(r)] = res_states;
  }

  int total_cols = 0;
  for (const auto &mat : reservoir_outputs) {
    total_cols += static_cast<int>(mat.cols());
  }

  Eigen::MatrixXd all_states(inputs.rows(), total_cols);
  int current_col = 0;
  for (const auto &mat : reservoir_outputs) {
    all_states.middleCols(current_col, mat.cols()) = mat;
    current_col += static_cast<int>(mat.cols());
  }
  return all_states;
}

Eigen::MatrixXd Model::collectCurrentStates(int input_dim) const {
  if (connection_type == "serial") {
    const Eigen::MatrixXd &state = reservoirs.back()->getState();
    if (state.cols() == 0) {
      throw std::runtime_error("Cannot generate without priming an uninitialized reservoir.");
    }
    return state;
  }

  int total_cols = 0;
  for (const auto &res : reservoirs) {
    int cols = static_cast<int>(res->getState().cols());
    if (cols == 0) {
      if (input_dim <= 0) {
        throw std::runtime_error("Cannot generate without priming an uninitialized reservoir.");
      }
      cols = res->getOutputDim(input_dim);
    }
    total_cols += cols;
  }

  Eigen::MatrixXd all_states(1, total_cols);
  int current_col = 0;
  for (const auto &res : reservoirs) {
    const Eigen::MatrixXd &state = res->getState();
    all_states.middleCols(current_col, state.cols()) = state;
    current_col += static_cast<int>(state.cols());
  }
  return all_states;
}
