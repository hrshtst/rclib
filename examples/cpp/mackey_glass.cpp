#include "rclib/Model.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

// Mackey-Glass time series generation
Eigen::VectorXd mackey_glass(int n_samples = 1500, int tau = 17, int seed = 0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dis(0.5, 1.0);
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n_samples + tau);
  for (int i = 0; i < tau; ++i) {
    x(i) = dis(gen);
  }
  for (int t = tau; t < n_samples + tau - 1; ++t) {
    x(t + 1) = x(t) + (0.2 * x(t - tau)) / (1 + std::pow(x(t - tau), 10)) - 0.1 * x(t);
  }
  return x.tail(n_samples);
}

int main() {
  // 1. Generate Mackey-Glass data
  Eigen::VectorXd data = mackey_glass();
  Eigen::MatrixXd X(data.size() - 1, 1);
  Eigen::MatrixXd y(data.size() - 1, 1);
  X.col(0) = data.head(data.size() - 1);
  y.col(0) = data.tail(data.size() - 1);

  // Split into training and testing sets
  int train_len = 1000;
  int washout_len = 100;
  Eigen::MatrixXd X_train = X.topRows(train_len);
  Eigen::MatrixXd y_train = y.topRows(train_len);
  Eigen::MatrixXd X_test = X.bottomRows(X.rows() - train_len);
  Eigen::MatrixXd y_test = y.bottomRows(y.rows() - train_len);

  // 2. Configure Reservoir
  auto res = std::make_shared<RandomSparseReservoir>(2000, 1.1, 0.05, 0.1, true, 0.5);

  // 3. Configure Readout
  auto readout = std::make_shared<RidgeReadout>(1e-8, true);

  // 4. Configure Model
  Model model;
  model.addReservoir(res);
  model.setReadout(readout);

  // 5. Fit and Predict
  model.fit(X_train, y_train, washout_len);
  Eigen::MatrixXd y_pred = model.predict(X_test, false);

  // 6. Print the results
  double mse = (y_pred - y_test).squaredNorm() / y_test.rows();
  std::cout << "Test loss (MSE): " << std::scientific << std::setprecision(4) << mse << std::endl;

  return 0;
}
