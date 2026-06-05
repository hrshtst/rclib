#include "rclib/readouts/RlsReadout.h"

#include <cmath>
#include <stdexcept>

RlsReadout::RlsReadout(double lambda, double delta, bool include_bias, Solver solver)
    : lambda(lambda), delta(delta), include_bias(include_bias), solver(solver), initialized(false) {
  if (lambda <= 0.0 || lambda > 1.0) {
    throw std::invalid_argument("lambda must be in (0, 1].");
  }
  if (delta <= 0.0) {
    throw std::invalid_argument("delta must be positive.");
  }
}

void RlsReadout::fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) {
  // RLS is an online algorithm, so fit will call partialFit.
  // We can call partialFit once with the full batch if it handles it,
  // or loop. Given we added mini-batch support, we call it once.
  initialized = false;
  partialFit(states, targets);
}

void RlsReadout::partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) {
  if (state.rows() == 0 || state.cols() == 0) {
    throw std::invalid_argument("state must be a non-empty 2D matrix.");
  }
  if (target.rows() != state.rows()) {
    throw std::invalid_argument("target must have the same number of rows as state.");
  }
  if (target.cols() == 0) {
    throw std::invalid_argument("target must have at least one column.");
  }

  int n_samples = state.rows();
  int n_in = state.cols();
  int n_features = n_in + (include_bias ? 1 : 0);
  int n_targets = target.cols();

  if (!initialized) {
    // Initialize W_out and P
    W_out = Eigen::MatrixXd::Zero(n_features, n_targets);
    P = (1.0 / delta) * Eigen::MatrixXd::Identity(n_features, n_features);

    // Pre-allocate buffers for RANK1
    x_aug.resize(n_features);
    k.resize(n_features);
    Px.resize(n_features);

    initialized = true;
  } else if (W_out.rows() != n_features || W_out.cols() != n_targets) {
    throw std::invalid_argument("state or target dimensions changed after RlsReadout initialization.");
  }

  if (solver == RANK_K_UPDATE && n_samples > 1 && lambda >= 1.0) {
    // Mini-batch RLS using Woodbury Matrix Identity (Rank-K)
    // Mathematically equivalent to sequential updates only when lambda = 1.0.
    // 1. Prepare X (augmented input matrix)
    Eigen::MatrixXd X;
    if (include_bias) {
      X.resize(n_samples, n_features);
      X.leftCols(n_in) = state;
      X.rightCols(1).setOnes();
    } else {
      X = state;
    }

    // 2. Woodbury update:
    // V = P * X^T (N x K)
    // S = lambda^K * I + X * V (K x K)
    // P = (1 / lambda^K) * (P - V * S^-1 * V^T)

    double lambda_k = std::pow(lambda, n_samples);
    Eigen::MatrixXd V = P.selfadjointView<Eigen::Upper>() * X.transpose();
    Eigen::MatrixXd S = lambda_k * Eigen::MatrixXd::Identity(n_samples, n_samples) + X * V;

    // Solve S * G = V^T  => G = S^-1 * V^T
    // Then P -= V * G
    Eigen::MatrixXd G = S.ldlt().solve(V.transpose());

    // Update P using GEMM (full matrix update as it's Rank-K)
    P.noalias() -= V * G;
    P *= (1.0 / lambda_k);

    // 3. Update Weights
    // W = W + P * X^T * (Y - X * W)  -- Note: using updated P
    Eigen::MatrixXd error = target - (X * W_out);
    W_out.noalias() += (P.selfadjointView<Eigen::Upper>() * X.transpose()) * error;

  } else {
    // Default: Sequential Rank-1 Updates
    for (int i = 0; i < n_samples; ++i) {
      // Prepare input vector x_aug
      x_aug.head(n_in) = state.row(i).transpose();
      if (include_bias) {
        x_aug(n_in) = 1.0;
      }

      // 1. Compute Px = P * x using symmetry (Upper triangle)
      Px.noalias() = P.selfadjointView<Eigen::Upper>() * x_aug;

      // 2. Compute denominator = lambda + x^T * Px
      double denominator = lambda + x_aug.dot(Px);

      // 3. Compute Kalman gain vector k = Px / denominator
      k = Px / denominator;

      // 4. Compute prediction y_hat = x^T * W_out and error
      Eigen::MatrixXd error_vec = target.row(i) - (x_aug.transpose() * W_out);

      // 5. Update weights: W_out = W_out + k * error
      W_out.noalias() += k * error_vec;

      // 6. Update P: P = (1/lambda) * (P - (Px * Px^T) / denominator)
      P.triangularView<Eigen::Upper>() *= (1.0 / lambda);
      P.selfadjointView<Eigen::Upper>().rankUpdate(Px, -1.0 / (lambda * denominator));
    }
  }
}

Eigen::MatrixXd RlsReadout::predict(const Eigen::MatrixXd &states) {
  if (!initialized) {
    throw std::runtime_error("RlsReadout must be fit before predict.");
  }
  if (states.rows() == 0 || states.cols() == 0) {
    throw std::invalid_argument("states must be a non-empty 2D matrix.");
  }
  if (states.cols() != W_out.rows() - (include_bias ? 1 : 0)) {
    throw std::invalid_argument("states column count does not match initialized RlsReadout.");
  }

  Eigen::MatrixXd X = states;
  if (include_bias) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    X.col(X.cols() - 1) = Eigen::VectorXd::Ones(X.rows());
  }
  return X * W_out;
}
