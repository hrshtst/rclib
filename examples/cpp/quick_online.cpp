#include "rclib/Model.h"
#include "rclib/readouts/RlsReadout.h"
#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

int main() {
  // 1. Data preparation (sine wave prediction)
  int n_samples = 1000;
  Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n_samples, 0, 100);
  Eigen::MatrixXd data = t.unaryExpr([](double v) { return std::sin(v); });

  Eigen::MatrixXd X = data.topRows(n_samples - 1);
  Eigen::MatrixXd Y = data.bottomRows(n_samples - 1);

  // 2. Initialization of ESN and RLS readout
  Model model;
  model.addReservoir(std::make_shared<RandomSparseReservoir>(500, 0.9));
  model.setReadout(std::make_shared<RlsReadout>(0.99, 1.0, true));

  // 3. Online learning (update weights sample by sample)
  for (int i = 0; i < X.rows(); ++i) {
    model.partialFit(X.row(i), Y.row(i));
  }

  // 4. Prediction and evaluation
  Eigen::MatrixXd Y_pred = model.predict(X);
  double mse = (Y - Y_pred).squaredNorm() / Y.rows();
  std::cout << "MSE: " << std::scientific << std::setprecision(2) << mse << std::endl;

  return 0;
}
