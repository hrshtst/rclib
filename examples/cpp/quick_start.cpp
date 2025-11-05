#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include <Eigen/Dense>

#include "rcl/Model.h"
#include "rcl/reservoirs/RandomSparseReservoir.h"
#include "rcl/readouts/RidgeReadout.h"

int main() {
    // 1. Create some dummy data
    Eigen::MatrixXd X_train = Eigen::MatrixXd::Random(100, 1);
    Eigen::MatrixXd y_train = X_train.unaryExpr([](double x){ return std::sin(x * 10); });

    Eigen::MatrixXd X_test(100, 1);
    X_test.col(0) = Eigen::VectorXd::LinSpaced(100, 0, 1);
    Eigen::MatrixXd y_test = X_test.unaryExpr([](double x){ return std::sin(x * 10); });

    // 2. Configure Reservoir
    auto res = std::make_shared<RandomSparseReservoir>(1000, 0.9, 0.1, 0.3, true);

    // 3. Configure Readout
    auto readout = std::make_shared<RidgeReadout>(0.1, true);

    // 4. Configure Model
    Model model;
    model.addReservoir(res);
    model.setReadout(readout);

    // 5. Fit and Predict
    model.fit(X_train, y_train);
    Eigen::MatrixXd y_pred = model.predict(X_test);

    // 6. Print the results
    double mse = (y_pred - y_test).squaredNorm() / y_test.rows();
    std::cout << "Test loss (MSE): " << mse << std::endl;

    return 0;
}
