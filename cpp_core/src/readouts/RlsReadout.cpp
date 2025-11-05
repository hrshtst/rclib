#include "rcl/readouts/RlsReadout.h"
#include <stdexcept>

RlsReadout::RlsReadout(double lambda, double delta, bool include_bias)
    : lambda(lambda), delta(delta), include_bias(include_bias), initialized(false) {
}

void RlsReadout::fit(const Eigen::MatrixXd& states, const Eigen::MatrixXd& targets) {
    // RLS is an online algorithm, so fit will call partialFit repeatedly.
    // Reset the state before fitting.
    initialized = false; // Force re-initialization in partialFit

    for (int i = 0; i < states.rows(); ++i) {
        partialFit(states.row(i), targets.row(i));
    }
}

void RlsReadout::partialFit(const Eigen::MatrixXd& state, const Eigen::MatrixXd& target) {
    Eigen::MatrixXd x = state;
    if (include_bias) {
        x.conservativeResize(1, x.cols() + 1);
        x(0, x.cols() - 1) = 1.0;
    }

    if (!initialized) {
        // Initialize W_out and P
        int n_features = x.cols();
        int n_targets = target.cols();

        W_out = Eigen::MatrixXd::Zero(n_features, n_targets);
        P = (1.0 / delta) * Eigen::MatrixXd::Identity(n_features, n_features);
        initialized = true;
    }

    // RLS update equations
    Eigen::MatrixXd Px = P * x.transpose();
    Eigen::MatrixXd k = Px / (lambda + (x * Px)(0,0));
    Eigen::MatrixXd y_hat = x * W_out;
    Eigen::MatrixXd error = target - y_hat;

    W_out = W_out + k * error;
    P = (1.0 / lambda) * (P - k * x * P);
}

Eigen::MatrixXd RlsReadout::predict(const Eigen::MatrixXd& states) {
    Eigen::MatrixXd X = states;
    if (include_bias) {
        X.conservativeResize(X.rows(), X.cols() + 1);
        X.col(X.cols() - 1) = Eigen::VectorXd::Ones(X.rows());
    }
    return X * W_out;
}
