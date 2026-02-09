#include "rclib/readouts/RidgeReadout.h"

#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void run_benchmark(int n_neurons, int n_samples, int n_outputs = 1) {
  std::cout << "------------------------------------------------------------" << std::endl;
  std::cout << "Benchmarking with N=" << n_neurons << ", T=" << n_samples << ", Outputs=" << n_outputs << std::endl;

  // Setup Data
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(n_samples, n_neurons);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(n_samples, n_outputs);
  double alpha = 1e-4;

  std::vector<std::pair<std::string, RidgeReadout::Solver>> solvers = {
      {"Cholesky (LDLT)", RidgeReadout::CHOLESKY},
      {"Conjugate Gradient (Explicit)", RidgeReadout::CONJUGATE_GRADIENT},
      {"Conjugate Gradient (Implicit)", RidgeReadout::CONJUGATE_GRADIENT_IMPLICIT}};

  for (const auto &solver_info : solvers) {
    RidgeReadout readout(alpha, false, solver_info.second);

    auto start = std::chrono::high_resolution_clock::now();
    readout.fit(X, Y);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << std::left << std::setw(30) << solver_info.first << ": " << elapsed.count() << " s" << std::endl;

    // Quick Accuracy Check (optional but good for validation)
    Eigen::MatrixXd pred = readout.predict(X);
    // Note: This check is vs training data just to see if it learned SOMETHING.
    // For proper residual check of the normal equations, we'd need XtX which we avoid computing here for implicit.
  }
}

int main() {
  std::vector<int> neurons = {100, 500, 1000, 2000, 4000};
  int fixed_samples = 10000;

  for (int n : neurons) {
    run_benchmark(n, fixed_samples);
  }

  return 0;
}
