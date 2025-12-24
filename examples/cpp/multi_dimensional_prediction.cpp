#include "rclib/Model.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

// Function to compute the Lorenz system derivatives
Eigen::Vector3d lorenz(const Eigen::Vector3d &xyz, double s = 10.0, double r = 28.0, double b = 2.667) {
  double x = xyz(0);
  double y = xyz(1);
  double z = xyz(2);

  double x_dot = s * (y - x);
  double y_dot = r * x - y - x * z;
  double z_dot = x * y - b * z;

  return Eigen::Vector3d(x_dot, y_dot, z_dot);
}

int main() {
  // --- Configuration Parameters ---
  const int n_total_samples = 2000;
  const int n_train_samples = 1500;
  const double dt = 0.01;

  const int n_neurons = 3000;
  const double spectral_radius = 1.25;
  const double sparsity = 0.05;
  const double leak_rate = 0.3;
  const double input_scaling = 1.0;
  const double ridge_alpha = 1e-5;
  const int washout_len = 200;
  const bool include_bias = true;
  const bool reset_state_before_predict = false;

  // --- 1. Data Generation (Lorenz Attractor) ---
  std::cout << "--- Generating Lorenz Attractor Data ---" << std::endl;
  Eigen::MatrixXd data(n_total_samples + 1, 3);
  data.row(0) = Eigen::Vector3d(0.0, 1.0, 1.05); // Initial condition

  for (int i = 0; i < n_total_samples; ++i) {
    data.row(i + 1) = data.row(i) + lorenz(data.row(i)).transpose() * dt;
  }

  Eigen::MatrixXd input_data = data.topRows(n_total_samples);
  Eigen::MatrixXd target_data = data.bottomRows(n_total_samples);

  Eigen::MatrixXd train_input = input_data.topRows(n_train_samples);
  Eigen::MatrixXd train_target = target_data.topRows(n_train_samples);
  Eigen::MatrixXd test_input = input_data.bottomRows(n_total_samples - n_train_samples);
  Eigen::MatrixXd test_target = target_data.bottomRows(n_total_samples - n_train_samples);

  // --- 2. Instantiate, Train, and Predict ---
  std::cout << "--- Initializing ESN ---" << std::endl;
  auto reservoir = std::make_shared<RandomSparseReservoir>(n_neurons, spectral_radius, sparsity, leak_rate,
                                                           input_scaling, include_bias);

  auto readout = std::make_shared<RidgeReadout>(ridge_alpha, include_bias);

  Model model;
  model.addReservoir(reservoir);
  model.setReadout(readout);

  std::cout << "--- Fitting ESN ---" << std::endl;
  model.fit(train_input, train_target, washout_len);

  std::cout << "--- Predicting with ESN ---" << std::endl;
  Eigen::MatrixXd predictions = model.predict(test_input, reset_state_before_predict);

  // --- 3. Evaluate Results ---
  Eigen::MatrixXd diff = predictions.topRows(test_target.rows()) - test_target;
  double mse = diff.array().square().mean();
  std::cout << "Mean Squared Error: " << mse << std::endl;

  std::cout << "\nSample Predictions vs. True Targets (Dimension 0):" << std::endl;
  for (int i = 0; i < std::min(10, (int)test_target.rows()); ++i) {
    std::cout << "True: " << test_target(i, 0) << ", Predicted: " << predictions(i, 0) << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
