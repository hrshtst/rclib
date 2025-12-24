#include "rclib/Model.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// Function to calculate Mean Squared Error
double calculate_mse(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) {
  return (predictions - targets).squaredNorm() / targets.rows();
}

int main() {
  // --- 1. Configuration ---
  std::cout << "--- Configuration ---" << std::endl;
  int n_neurons = 2000;
  double spectral_radius = 0.99;
  double sparsity = 0.02;
  double leak_rate = 0.2;
  bool include_bias = true;

  double ridge_alpha = 1e-4;
  double lms_learning_rate = 0.001;
  double rls_lambda = 0.999;
  double rls_delta = 1.0;

  // --- 2. Generate Data with Changing Dynamics ---
  std::cout << "--- Generating Data ---" << std::endl;
  int n_total = 4000;
  int n_train = 2000;
  int change_point = 3000;

  Eigen::VectorXd time_vec = Eigen::VectorXd::LinSpaced(n_total, 0, 80);
  double freq1 = 1.0, freq2 = 1.5;

  Eigen::MatrixXd signal(n_total, 1);
  for (int i = 0; i < n_total; ++i) {
    if (i < change_point) {
      signal(i, 0) = std::sin(freq1 * time_vec(i));
    } else {
      signal(i, 0) = std::sin(freq2 * time_vec(i));
    }
  }

  Eigen::MatrixXd data = signal;

  Eigen::MatrixXd train_input = data.block(0, 0, n_train - 1, 1);
  Eigen::MatrixXd train_target = data.block(1, 0, n_train - 1, 1);
  Eigen::MatrixXd test_input = data.block(n_train - 1, 0, n_total - (n_train - 1) - 1, 1);
  Eigen::MatrixXd test_target = data.block(n_train, 0, n_total - n_train, 1);

  // --- 3. Offline Training (Ridge) ---
  std::cout << "\n--- 1. Offline Training with Ridge ---" << std::endl;
  auto res_ridge =
      std::make_shared<RandomSparseReservoir>(n_neurons, spectral_radius, sparsity, leak_rate, include_bias);
  auto readout_ridge = std::make_shared<RidgeReadout>(ridge_alpha, include_bias);
  Model esn_ridge;
  esn_ridge.addReservoir(res_ridge);
  esn_ridge.setReadout(readout_ridge);
  esn_ridge.fit(train_input, train_target);

  std::cout << "\n--- 2. Predicting with offline-trained model (no adaptation) ---" << std::endl;
  Eigen::MatrixXd preds_ridge = esn_ridge.predict(test_input);
  double mse_ridge = calculate_mse(preds_ridge, test_target);
  std::cout << "Ridge MSE: " << std::scientific << std::setprecision(4) << mse_ridge << std::endl;

  // --- 4. Online Adaptation (LMS) ---
  std::cout << "\n--- 3. Adapting online with LMS ---" << std::endl;
  auto res_lms = std::make_shared<RandomSparseReservoir>(n_neurons, spectral_radius, sparsity, leak_rate, include_bias);
  auto readout_lms = std::make_shared<LmsReadout>(lms_learning_rate, include_bias);
  Model esn_lms;
  esn_lms.addReservoir(res_lms);
  esn_lms.setReadout(readout_lms);

  Eigen::MatrixXd preds_lms(test_target.rows(), test_target.cols());

  // Initialize W_out and predict first step
  esn_lms.getReservoir(0)->advance(test_input.row(0));
  esn_lms.getReadout()->partialFit(esn_lms.getReservoir(0)->getState(), test_target.row(0));
  preds_lms.row(0) = esn_lms.predict(esn_lms.getReservoir(0)->getState());

  for (int i = 1; i < test_input.rows(); ++i) {
    Eigen::MatrixXd current_input = test_input.row(i);
    Eigen::MatrixXd current_target = test_target.row(i);

    // Predict and adapt online
    preds_lms.row(i) = esn_lms.predictOnline(current_input);
    esn_lms.getReadout()->partialFit(esn_lms.getReservoir(0)->getState(), current_target);
  }
  double mse_lms = calculate_mse(preds_lms, test_target);
  std::cout << "LMS MSE: " << std::scientific << std::setprecision(4) << mse_lms << std::endl;

  // --- 5. Online Adaptation (RLS) ---
  std::cout << "\n--- 4. Adapting online with RLS ---" << std::endl;
  auto res_rls = std::make_shared<RandomSparseReservoir>(n_neurons, spectral_radius, sparsity, leak_rate, include_bias);
  auto readout_rls = std::make_shared<RlsReadout>(rls_lambda, rls_delta, include_bias);
  Model esn_rls;
  esn_rls.addReservoir(res_rls);
  esn_rls.setReadout(readout_rls);

  Eigen::MatrixXd preds_rls(test_target.rows(), test_target.cols());
  // Initialize W_out and P and predict first step
  esn_rls.getReservoir(0)->advance(test_input.row(0));
  esn_rls.getReadout()->partialFit(esn_rls.getReservoir(0)->getState(), test_target.row(0));
  preds_rls.row(0) = esn_rls.predict(esn_rls.getReservoir(0)->getState());

  for (int i = 1; i < test_input.rows(); ++i) {
    Eigen::MatrixXd current_input = test_input.row(i);
    Eigen::MatrixXd current_target = test_target.row(i);

    // Predict and adapt online
    preds_rls.row(i) = esn_rls.predictOnline(current_input);
    esn_rls.getReadout()->partialFit(esn_rls.getReservoir(0)->getState(), current_target);
  }
  double mse_rls = calculate_mse(preds_rls, test_target);
  std::cout << "RLS MSE: " << std::scientific << std::setprecision(4) << mse_rls << std::endl;

  return 0;
}
