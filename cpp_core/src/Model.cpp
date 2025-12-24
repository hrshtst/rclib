#include "rclib/Model.h"

#include <omp.h>
#include <stdexcept>

void Model::addReservoir(std::shared_ptr<Reservoir> res, std::string connection_type) {
  reservoirs.push_back(res);
  this->connection_type = connection_type;
}

void Model::setReadout(std::shared_ptr<Readout> readout) { this->readout = readout; }

void Model::fit(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, int washout_len) {
  if (reservoirs.empty() || !readout) {
    throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
  }

  if (washout_len < 0 || washout_len >= inputs.rows()) {
    throw std::out_of_range("washout_len must be non-negative and less than the number of input rows.");
  }

  resetReservoirs();

  // Collect states from all reservoirs for the entire input sequence
  Eigen::MatrixXd all_states_full;

  if (connection_type == "serial") {
    Eigen::MatrixXd current_input = inputs;
    for (const auto &res : reservoirs) {
      Eigen::MatrixXd res_states(inputs.rows(), res->getState().cols());
      for (int i = 0; i < inputs.rows(); ++i) {
        res_states.row(i) = res->advance(current_input.row(i));
      }
      current_input = res_states;
    }
    all_states_full = current_input;
  } else if (connection_type == "parallel") {
    std::vector<Eigen::MatrixXd> reservoir_outputs(reservoirs.size());
#ifdef RCLIB_ENABLE_USER_PARALLELIZATION
#  pragma omp parallel for
#endif
    for (size_t r = 0; r < reservoirs.size(); ++r) {
      auto &res = reservoirs[r];
      Eigen::MatrixXd res_states_for_all_inputs(inputs.rows(), res->getState().cols());
      for (int i = 0; i < inputs.rows(); ++i) {
        res_states_for_all_inputs.row(i) = res->advance(inputs.row(i));
      }
      reservoir_outputs[r] = res_states_for_all_inputs;
    }

    if (!reservoir_outputs.empty()) {
      int total_cols = 0;
      for (const auto &mat : reservoir_outputs) {
        total_cols += mat.cols();
      }
      all_states_full.resize(inputs.rows(), total_cols);
      int current_col = 0;
      for (const auto &mat : reservoir_outputs) {
        all_states_full.middleCols(current_col, mat.cols()) = mat;
        current_col += mat.cols();
      }
    }
  }

  // Apply washout period
  Eigen::MatrixXd fit_states = all_states_full.bottomRows(all_states_full.rows() - washout_len);
  Eigen::MatrixXd fit_targets = targets.bottomRows(targets.rows() - washout_len);

  readout->fit(fit_states, fit_targets);
}

Eigen::MatrixXd Model::predict(const Eigen::MatrixXd &inputs, bool reset_state_before_predict) {
  if (reservoirs.empty() || !readout) {
    throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
  }

  if (reset_state_before_predict) {
    resetReservoirs();
  }

  // Collect states from all reservoirs
  Eigen::MatrixXd all_states;

  if (connection_type == "serial") {
    Eigen::MatrixXd current_input = inputs;
    for (const auto &res : reservoirs) {
      Eigen::MatrixXd res_states(inputs.rows(), res->getState().cols());
      for (int i = 0; i < inputs.rows(); ++i) {
        res_states.row(i) = res->advance(current_input.row(i));
      }
      current_input = res_states;
    }
    all_states = current_input;
  } else if (connection_type == "parallel") {
    std::vector<Eigen::MatrixXd> reservoir_outputs(reservoirs.size());
#ifdef RCLIB_ENABLE_USER_PARALLELIZATION
#  pragma omp parallel for
#endif
    for (size_t r = 0; r < reservoirs.size(); ++r) {
      auto &res = reservoirs[r];
      Eigen::MatrixXd res_states_for_all_inputs(inputs.rows(), res->getState().cols());
      for (int i = 0; i < inputs.rows(); ++i) {
        res_states_for_all_inputs.row(i) = res->advance(inputs.row(i));
      }
      reservoir_outputs[r] = res_states_for_all_inputs;
    }

    if (!reservoir_outputs.empty()) {
      int total_cols = 0;
      for (const auto &mat : reservoir_outputs) {
        total_cols += mat.cols();
      }
      all_states.resize(inputs.rows(), total_cols);
      int current_col = 0;
      for (const auto &mat : reservoir_outputs) {
        all_states.middleCols(current_col, mat.cols()) = mat;
        current_col += mat.cols();
      }
    }
  }

  return readout->predict(all_states);
}

Eigen::MatrixXd Model::predictOnline(const Eigen::MatrixXd &input) {
  if (reservoirs.empty() || !readout) {
    throw std::runtime_error("Model is not fully configured. Add at least one reservoir and a readout.");
  }

  // Collect states from all reservoirs
  Eigen::MatrixXd all_states;

  if (connection_type == "serial") {
    Eigen::MatrixXd current_input = input;
    for (const auto &res : reservoirs) {
      current_input = res->advance(current_input);
    }
    all_states = current_input;
  } else if (connection_type == "parallel") {
    for (const auto &res : reservoirs) {
      Eigen::MatrixXd res_state = res->advance(input);
      if (all_states.size() == 0) {
        all_states = res_state;
      } else {
        all_states.conservativeResize(all_states.rows(), all_states.cols() + res_state.cols());
        all_states.rightCols(res_state.cols()) = res_state;
      }
    }
  }

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

  // 1. Priming phase
  Eigen::MatrixXd last_state;
  if (prime_inputs.rows() > 0) {
    if (connection_type == "serial") {
      Eigen::MatrixXd current_input = prime_inputs;
      for (const auto &res : reservoirs) {
        Eigen::MatrixXd res_states(prime_inputs.rows(), res->getState().cols());
        for (int i = 0; i < prime_inputs.rows(); ++i) {
          res_states.row(i) = res->advance(current_input.row(i));
        }
        current_input = res_states;
      }
      last_state = current_input.row(current_input.rows() - 1);
    } else { // parallel
      std::vector<Eigen::MatrixXd> reservoir_outputs;
      for (const auto &res : reservoirs) {
        Eigen::MatrixXd res_states_for_all_inputs(prime_inputs.rows(), res->getState().cols());
        for (int i = 0; i < prime_inputs.rows(); ++i) {
          res_states_for_all_inputs.row(i) = res->advance(prime_inputs.row(i));
        }
        reservoir_outputs.push_back(res_states_for_all_inputs.row(res_states_for_all_inputs.rows() - 1));
      }

      int total_cols = 0;
      for (const auto &mat : reservoir_outputs) {
        total_cols += mat.cols();
      }
      last_state.resize(1, total_cols);
      int current_col = 0;
      for (const auto &mat : reservoir_outputs) {
        last_state.middleCols(current_col, mat.cols()) = mat;
        current_col += mat.cols();
      }
    }
  } else {
    // If no priming, start from the current state of the reservoirs
    if (connection_type == "serial") {
      last_state = reservoirs.back()->getState();
    } else { // parallel
      int total_cols = 0;
      for (const auto &res : reservoirs) {
        total_cols += res->getState().cols();
      }
      last_state.resize(1, total_cols);
      int current_col = 0;
      for (const auto &res : reservoirs) {
        last_state.middleCols(current_col, res->getState().cols()) = res->getState();
        current_col += res->getState().cols();
      }
    }
  }

  // First prediction is based on the last state of the priming phase
  Eigen::MatrixXd next_input = readout->predict(last_state);

  // 2. Generative phase
  Eigen::MatrixXd generated_outputs(n_steps, next_input.cols());
  if (n_steps > 0) {
    generated_outputs.row(0) = next_input;
  }

  for (int i = 1; i < n_steps; ++i) {
    Eigen::MatrixXd current_state;
    if (connection_type == "serial") {
      Eigen::MatrixXd current_step_input = next_input;
      for (const auto &res : reservoirs) {
        current_step_input = res->advance(current_step_input);
      }
      current_state = current_step_input;
    } else { // parallel
      for (const auto &res : reservoirs) {
        Eigen::MatrixXd res_state = res->advance(next_input);
        if (current_state.size() == 0) {
          current_state = res_state;
        } else {
          current_state.conservativeResize(current_state.rows(), current_state.cols() + res_state.cols());
          current_state.rightCols(res_state.cols()) = res_state;
        }
      }
    }
    next_input = readout->predict(current_state);
    generated_outputs.row(i) = next_input;
  }

  return generated_outputs;
}

void Model::resetReservoirs() {
#ifdef RCLIB_ENABLE_USER_PARALLELIZATION
#  pragma omp parallel for
#endif
  for (int i = 0; i < static_cast<int>(reservoirs.size()); ++i) {
    reservoirs[i]->resetState();
  }
}
