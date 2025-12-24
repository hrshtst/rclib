#include "rclib/Model.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <catch2/catch_all.hpp>
#include <cmath>
#include <memory>
#include <random>

double run_esn_experiment_cpp(bool include_bias, double input_scaling = 1.0) {
  // --- Configuration Parameters ---
  const int n_total_samples = 200; // Smaller for faster tests
  const int n_train_samples = 150;
  const double noise_amplitude = 0.01; // Reduced noise for clearer bias effect

  const int n_neurons = 100; // Smaller for faster tests
  const double spectral_radius = 0.9;
  const double sparsity = 0.1;
  const double leak_rate = 0.3;
  const double ridge_alpha = 1e-6;

  // --- Data Generation ---
  Eigen::VectorXd time_np(n_total_samples);
  for (int i = 0; i < n_total_samples; ++i) {
    time_np(i) = static_cast<double>(i) * 20.0 / (n_total_samples - 1);
  }

  Eigen::VectorXd clean_data = time_np.array().sin();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(0, 1);
  Eigen::VectorXd noise(n_total_samples);
  for (int i = 0; i < n_total_samples; ++i) {
    noise(i) = noise_amplitude * d(gen);
  }

  Eigen::MatrixXd data = (clean_data + noise).reshaped(n_total_samples, 1);

  Eigen::MatrixXd input_data = data.topRows(n_total_samples - 1);
  Eigen::MatrixXd target_data = data.bottomRows(n_total_samples - 1);

  Eigen::MatrixXd train_input = input_data.topRows(n_train_samples);
  Eigen::MatrixXd train_target = target_data.topRows(n_train_samples);
  Eigen::MatrixXd test_input = data.middleRows(n_train_samples, n_total_samples - n_train_samples - 1);
  Eigen::MatrixXd test_target = data.bottomRows(n_total_samples - n_train_samples - 1);

  // --- Instantiate, Train, and Predict ---
  auto reservoir = std::make_shared<RandomSparseReservoir>(n_neurons, spectral_radius, sparsity, leak_rate,
                                                           input_scaling, include_bias);

  auto readout = std::make_shared<RidgeReadout>(ridge_alpha, include_bias);

  Model model;
  model.addReservoir(reservoir);
  model.setReadout(readout);

  model.fit(train_input, train_target);
  Eigen::MatrixXd predictions = model.predict(test_input, true);

  Eigen::MatrixXd diff = predictions.topRows(test_target.rows()) - test_target;
  double mse = diff.array().square().mean();
  return mse;
}

TEST_CASE("Bias effect on ESN performance", "[bias]") {
  double mse_with_bias = run_esn_experiment_cpp(true);
  double mse_without_bias = run_esn_experiment_cpp(false);

  // We expect the MSE values to be different when bias is included vs. not included.
  // The exact difference or which one is better can vary based on random initialization
  // and specific parameters, but they should not be identical.
  REQUIRE(std::abs(mse_with_bias - mse_without_bias) > 1e-6); // Assert that the MSE values are significantly different
}
