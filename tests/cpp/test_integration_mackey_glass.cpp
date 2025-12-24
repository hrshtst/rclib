#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <numeric>
#include <rclib/Model.h>
#include <rclib/readouts/RidgeReadout.h>
#include <rclib/reservoirs/RandomSparseReservoir.h>
#include <vector>

// Deterministic Mackey-Glass generator for C++ tests
std::vector<double> generate_mackey_glass(int n_samples, int tau = 17, int seed = 42) {
  std::vector<double> x(n_samples + tau);

  // Fixed initialization for determinism
  for (int i = 0; i < tau; ++i) {
    x[i] = 0.5 + 0.1 * std::sin(i + seed);
  }

  for (int t = tau; t < n_samples + tau - 1; ++t) {
    double x_tau = x[t - tau];
    x[t + 1] = x[t] + (0.2 * x_tau) / (1.0 + std::pow(x_tau, 10.0)) - 0.1 * x[t];
  }

  std::vector<double> result(x.begin() + tau, x.end());
  return result;
}

TEST_CASE("Integration: Mackey-Glass Accuracy", "[integration][accuracy]") {
  int n_samples = 1500;
  int train_len = 1000;
  int washout_len = 100;
  int test_len = n_samples - train_len - 1;

  auto data = generate_mackey_glass(n_samples);

  Eigen::MatrixXd x_train(train_len, 1);
  Eigen::MatrixXd y_train(train_len, 1);
  for (int i = 0; i < train_len; ++i) {
    x_train(i, 0) = data[i];
    y_train(i, 0) = data[i + 1];
  }

  Eigen::MatrixXd x_test(test_len, 1);
  Eigen::MatrixXd y_test(test_len, 1);
  for (int i = 0; i < test_len; ++i) {
    x_test(i, 0) = data[train_len + i];
    y_test(i, 0) = data[train_len + i + 1];
  }

  auto res = std::make_shared<RandomSparseReservoir>(1000, 1.1, 0.05, 0.1, 0.5, true, 42);

  auto readout = std::make_shared<RidgeReadout>(1e-8, true);

  Model model;
  model.addReservoir(res);
  model.setReadout(readout);

  model.fit(x_train, y_train, washout_len);
  Eigen::MatrixXd y_pred = model.predict(x_test, false);

  double mse = 0;
  for (int i = 0; i < test_len; ++i) {
    double diff = y_pred(i, 0) - y_test(i, 0);
    mse += diff * diff;
  }
  mse /= test_len;

  // Use a loose bound because the C++ initialization might differ slightly from the Python one
  // due to how RandomSparseReservoir is seeded internally (using std::mt19937).
  REQUIRE(mse < 1.0e-3);
}
