#include "rclib/Model.h"
#include "rclib/readouts/LmsReadout.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// Helper to calculate Mean Squared Error
double calculate_mse(const Eigen::MatrixXd &predictions, const Eigen::MatrixXd &targets) {
  if (predictions.rows() != targets.rows()) {
    return -1.0; // Or throw an exception
  }
  return (predictions - targets).squaredNorm() / targets.rows();
}

// Helper to print results in CSV format
void print_result(const std::string &method, double time_s, double mse) {
  std::cout << method << "," << time_s << "," << mse << std::endl;
}

int main() {
  // --- 1. Configuration ---
  int n_neurons = 2000;
  double spectral_radius = 0.99;
  double sparsity = 0.02;
  double leak_rate = 0.2;
  bool include_bias = true;

  double ridge_alpha = 1e-4;
  double lms_learning_rate = 0.001;
  double rls_lambda = 0.999;
  double rls_delta = 1.0;

  // --- 2. Generate Data ---
  int n_total = 4000;
  int n_train = 2000;

  Eigen::VectorXd time_vec = Eigen::VectorXd::LinSpaced(n_total, 0, 80);
  Eigen::MatrixXd signal(n_total, 1);
  for (int i = 0; i < n_total; ++i) {
    signal(i, 0) = std::sin(time_vec(i));
  }

  Eigen::MatrixXd train_input = signal.block(0, 0, n_train - 1, 1);
  Eigen::MatrixXd train_target = signal.block(1, 0, n_train - 1, 1);
  Eigen::MatrixXd test_input = signal.block(n_train - 1, 0, n_total - (n_train - 1) - 1, 1);
  Eigen::MatrixXd test_target = signal.block(n_train, 0, n_total - n_train, 1);

  // Print header
  std::cout << "method,time_s,mse" << std::endl;

  // --- 3. Offline (Ridge) Benchmark ---
  {
    auto res_ridge =
        std::make_shared<RandomSparseReservoir>(n_neurons, spectral_radius, sparsity, leak_rate, include_bias);
    auto readout_ridge = std::make_shared<RidgeReadout>(ridge_alpha, include_bias);
    Model esn_ridge;
    esn_ridge.addReservoir(res_ridge);
    esn_ridge.setReadout(readout_ridge);

    // Time fit
    auto start_fit = std::chrono::high_resolution_clock::now();
    esn_ridge.fit(train_input, train_target);
    auto end_fit = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fit_duration = end_fit - start_fit;
    // MSE for fit is on training data, which is not standard, so we'll predict on train data for a comparable metric
    Eigen::MatrixXd train_preds = esn_ridge.predict(train_input, false);
    print_result("offline_fit", fit_duration.count(), calculate_mse(train_preds, train_target));

    // Time predict
    auto start_predict = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd preds_ridge = esn_ridge.predict(test_input);
    auto end_predict = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> predict_duration = end_predict - start_predict;
    print_result("offline_predict", predict_duration.count(), calculate_mse(preds_ridge, test_target));
  }

  // --- 4. Online (LMS) Benchmark ---
  {
    auto res_lms =
        std::make_shared<RandomSparseReservoir>(n_neurons, spectral_radius, sparsity, leak_rate, include_bias);
    auto readout_lms = std::make_shared<LmsReadout>(lms_learning_rate, include_bias);
    Model esn_lms;
    esn_lms.addReservoir(res_lms);
    esn_lms.setReadout(readout_lms);
    Eigen::MatrixXd preds_lms(test_target.rows(), test_target.cols());

    // Initialize the readout on the first sample BEFORE starting the timer
    Eigen::MatrixXd initial_state_lms = esn_lms.getReservoir(0)->advance(test_input.row(0));
    esn_lms.getReadout()->partialFit(initial_state_lms, test_target.row(0));
    preds_lms.row(0) = esn_lms.getReadout()->predict(initial_state_lms);

    auto start_lms = std::chrono::high_resolution_clock::now();
    // Loop from the second sample
    for (int i = 1; i < test_input.rows(); ++i) {
      preds_lms.row(i) = esn_lms.predictOnline(test_input.row(i));
      esn_lms.getReadout()->partialFit(esn_lms.getReservoir(0)->getState(), test_target.row(i));
    }
    auto end_lms = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> lms_duration = end_lms - start_lms;
    print_result("online_lms", lms_duration.count(), calculate_mse(preds_lms, test_target));
  }

  // --- 5. Online (RLS) Benchmark ---
  {
    auto res_rls =
        std::make_shared<RandomSparseReservoir>(n_neurons, spectral_radius, sparsity, leak_rate, include_bias);
    auto readout_rls = std::make_shared<RlsReadout>(rls_lambda, rls_delta, include_bias);
    Model esn_rls;
    esn_rls.addReservoir(res_rls);
    esn_rls.setReadout(readout_rls);
    Eigen::MatrixXd preds_rls(test_target.rows(), test_target.cols());

    // Initialize the readout on the first sample BEFORE starting the timer
    Eigen::MatrixXd initial_state_rls = esn_rls.getReservoir(0)->advance(test_input.row(0));
    esn_rls.getReadout()->partialFit(initial_state_rls, test_target.row(0));
    preds_rls.row(0) = esn_rls.getReadout()->predict(initial_state_rls);

    auto start_rls = std::chrono::high_resolution_clock::now();
    // Loop from the second sample
    for (int i = 1; i < test_input.rows(); ++i) {
      preds_rls.row(i) = esn_rls.predictOnline(test_input.row(i));
      esn_rls.getReadout()->partialFit(esn_rls.getReservoir(0)->getState(), test_target.row(i));
    }
    auto end_rls = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> rls_duration = end_rls - start_rls;
    print_result("online_rls", rls_duration.count(), calculate_mse(preds_rls, test_target));
  }

  return 0;
}
