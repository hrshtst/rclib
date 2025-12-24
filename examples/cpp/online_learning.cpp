#include "rclib/Model.h"
#include "rclib/readouts/RlsReadout.h" // Using RlsReadout for online learning
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

int main() {
  // 1. Create some dummy data for online training
  int num_steps = 1000;
  int input_dim = 1;
  int output_dim = 1;

  Eigen::MatrixXd X_data(num_steps, input_dim);
  Eigen::MatrixXd y_data(num_steps, output_dim);

  for (int i = 0; i < num_steps; ++i) {
    double t = static_cast<double>(i) / 100.0;
    X_data(i, 0) = std::sin(t);
    y_data(i, 0) = std::cos(t); // Target is derivative of input for a simple example
  }

  // 2. Configure Reservoir
  auto res = std::make_shared<RandomSparseReservoir>(100, 0.9, 0.1, 0.3, true); // Smaller reservoir for online example

  // 3. Configure Readout for online learning (RLS)
  auto readout = std::make_shared<RlsReadout>(0.99, 1.0, true); // lambda, delta, include_bias

  // 4. Configure Model
  Model model;
  model.addReservoir(res);
  model.setReadout(readout);

  // 5. Online Training Loop
  std::cout << "Starting online training..." << std::endl;
  for (int i = 0; i < num_steps; ++i) {
    Eigen::MatrixXd current_input = X_data.row(i);
    Eigen::MatrixXd current_target = y_data.row(i);

    // Advance reservoir state
    res->advance(current_input);
    Eigen::MatrixXd current_state = res->getState();

    // Perform partial fit (online update)
    readout->partialFit(current_state, current_target);

    if ((i + 1) % 100 == 0) {
      std::cout << "Step " << (i + 1) << "/" << num_steps << std::endl;
    }
  }
  std::cout << "Online training finished." << std::endl;

  // 6. Predict on some new data (e.g., the same data for evaluation)
  Eigen::MatrixXd y_pred = model.predict(X_data);

  // 7. Print the results
  double mse = (y_pred - y_data).squaredNorm() / y_data.rows();
  std::cout << "Test loss (MSE) after online training: " << std::scientific << std::setprecision(4) << mse << std::endl;

  return 0;
}
