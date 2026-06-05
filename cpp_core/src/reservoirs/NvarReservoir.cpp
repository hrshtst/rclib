#include "rclib/reservoirs/NvarReservoir.h"

#include <limits>
#include <stdexcept>
#include <vector>

NvarReservoir::NvarReservoir(int num_lags, int polynomial_order)
    : num_lags(num_lags), polynomial_order(polynomial_order), input_dim(0), initialized(false) {
  if (num_lags <= 0) {
    throw std::invalid_argument("num_lags must be positive.");
  }
  if (polynomial_order <= 0) {
    throw std::invalid_argument("polynomial_order must be positive.");
  }
}

void NvarReservoir::initialize(int input_dim) {
  if (input_dim <= 0) {
    throw std::invalid_argument("input_dim must be positive.");
  }
  this->input_dim = input_dim;
  state = Eigen::MatrixXd::Zero(1, getOutputDim(input_dim));
  past_inputs = Eigen::MatrixXd::Zero(num_lags, input_dim);
  initialized = true;
}

const Eigen::MatrixXd &NvarReservoir::advance(const Eigen::MatrixXd &input) {
  if (input.rows() != 1) {
    throw std::invalid_argument("NvarReservoir::advance expects a single input row.");
  }
  if (!initialized) {
    initialize(input.cols());
  } else if (input.cols() != input_dim) {
    throw std::invalid_argument("input dimension changed after NVAR initialization.");
  }

  // Shift past inputs
  for (int i = num_lags - 1; i > 0; --i) {
    past_inputs.row(i) = past_inputs.row(i - 1);
  }
  // Add new input
  past_inputs.row(0) = input;

  Eigen::RowVectorXd delayed(static_cast<Eigen::Index>(num_lags) * input_dim);
  for (int i = 0; i < num_lags; ++i) {
    delayed.segment(static_cast<Eigen::Index>(i) * input_dim, input_dim) = past_inputs.row(i);
  }

  std::vector<double> features;
  features.reserve(static_cast<size_t>(state.cols()));
  for (int degree = 1; degree <= polynomial_order; ++degree) {
    appendMonomials(delayed, 0, degree, 1.0, features);
  }

  for (Eigen::Index i = 0; i < state.cols(); ++i) {
    state(0, i) = features[static_cast<size_t>(i)];
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

int NvarReservoir::getOutputDim(int input_dim) const {
  if (input_dim <= 0) {
    throw std::invalid_argument("input_dim must be positive.");
  }
  int n_variables = num_lags * input_dim;
  int total = 0;
  for (int degree = 1; degree <= polynomial_order; ++degree) {
    total += countMonomials(n_variables, degree);
  }
  return total;
}

void NvarReservoir::appendMonomials(const Eigen::RowVectorXd &delayed, int start_index, int remaining_degree,
                                    double current_value, std::vector<double> &features) const {
  if (remaining_degree == 0) {
    features.push_back(current_value);
    return;
  }

  for (int i = start_index; i < delayed.cols(); ++i) {
    appendMonomials(delayed, i, remaining_degree - 1, current_value * delayed(i), features);
  }
}

int NvarReservoir::countMonomials(int n_variables, int degree) {
  // C(n + degree - 1, degree), computed with integer division at each step.
  long long result = 1;
  for (int i = 1; i <= degree; ++i) {
    result = result * (n_variables + i - 1) / i;
    if (result > std::numeric_limits<int>::max()) {
      throw std::overflow_error("NVAR feature count exceeds int range.");
    }
  }
  return static_cast<int>(result);
}
