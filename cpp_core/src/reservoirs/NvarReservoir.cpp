#include "rclib/reservoirs/NvarReservoir.h"

NvarReservoir::NvarReservoir(int num_lags) : num_lags(num_lags), initialized(false) {}

void NvarReservoir::initialize(int input_dim) {
  this->input_dim = input_dim;
  state = Eigen::MatrixXd::Zero(1, num_lags * input_dim);
  past_inputs = Eigen::MatrixXd::Zero(num_lags, input_dim);
  initialized = true;
}

Eigen::MatrixXd NvarReservoir::advance(const Eigen::MatrixXd &input) {
  if (!initialized) {
    initialize(input.cols());
  }

  // Shift past inputs
  for (int i = num_lags - 1; i > 0; --i) {
    past_inputs.row(i) = past_inputs.row(i - 1);
  }
  // Add new input
  past_inputs.row(0) = input;

  // Update state
  for (int i = 0; i < num_lags; ++i) {
    state.block(0, i * input_dim, 1, input_dim) = past_inputs.row(i);
  }

  return state;
}

void NvarReservoir::resetState() {
  if (initialized) {
    state.setZero();
    past_inputs.setZero();
  }
}

const Eigen::MatrixXd &NvarReservoir::getState() const { return state; }
