#include "rclib/Model.h"
#include "rclib/readouts/RidgeReadout.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

// Mackey-Glass time series generator (C++ version)
Eigen::VectorXd mackey_glass(int n_points = 2000, int tau = 17, double delta_t = 1.0) {
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n_points);
  // Use a fixed seed for reproducibility, similar to the Python example
  srand(0);
  for (int i = 0; i < tau; ++i) {
    x(i) = (double)rand() / RAND_MAX;
  }

  for (int t = tau; t < n_points; ++t) {
    double x_tau = x(t - tau);
    x(t) = x(t - 1) + (0.2 * x_tau) / (1.0 + std::pow(x_tau, 10)) - 0.1 * x(t - 1);
  }
  return x;
}

void saveData(const std::string &filename, const Eigen::MatrixXd &truth, const Eigen::MatrixXd &prediction,
              int prime_len) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }

  file << "Time,GroundTruth,Prediction\n";

  // Write ground truth for the whole test period
  for (int i = 0; i < truth.rows(); ++i) {
    file << i << "," << truth(i, 0) << ",";
    if (i >= prime_len) {
      // Prediction starts after priming
      file << prediction(i - prime_len, 0);
    }
    file << "\n";
  }

  std::cout << "Data saved to " << filename << std::endl;
}

int main() {
  std::cout << "Running C++ Generative Prediction Example..." << std::endl;

  // 1. Generate data
  Eigen::VectorXd data_vec = mackey_glass(3000);
  Eigen::MatrixXd data(data_vec.size(), 1);
  data.col(0) = data_vec;

  // 2. Split into training and testing sets
  int train_len = 1500;
  int prime_len = 100;

  Eigen::MatrixXd train_data = data.topRows(train_len);
  Eigen::MatrixXd test_data = data.bottomRows(data.rows() - train_len);

  Eigen::MatrixXd X_train = train_data.topRows(train_len - 1);
  Eigen::MatrixXd y_train = train_data.bottomRows(train_len - 1);
  Eigen::MatrixXd X_test = test_data.topRows(test_data.rows() - 1);
  Eigen::MatrixXd y_test = test_data.bottomRows(test_data.rows() - 1);

  // 3. Configure and build the ESN model
  auto res = std::make_shared<RandomSparseReservoir>(500, 1.2, 0.1, 0.3, 1.0, true);
  auto readout = std::make_shared<RidgeReadout>(1e-8, true);

  Model model;
  model.addReservoir(res, "serial");
  model.setReadout(readout);

  // 4. Train the model
  std::cout << "Training the model..." << std::endl;
  int washout = 100;
  model.fit(X_train, y_train, washout);

  // 5. Prime the model and perform generative prediction
  std::cout << "Performing generative prediction..." << std::endl;
  Eigen::MatrixXd prime_data = X_test.topRows(prime_len);
  int n_generate_steps = X_test.rows() - prime_len;

  // Prime the state by running a normal prediction first
  model.predict(prime_data, true);
  // Now generate from the final state of the priming
  Eigen::MatrixXd generated_output = model.predictGenerative(Eigen::MatrixXd(0, 1), n_generate_steps);

  // 6. Save the results to a CSV file
  saveData("generative_prediction_results.csv", y_test, generated_output, prime_len);

  return 0;
}
