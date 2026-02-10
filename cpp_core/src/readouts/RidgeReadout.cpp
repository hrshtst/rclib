#include "rclib/readouts/RidgeReadout.h"

#include "rclib/readouts/RidgeLinearOperator.h"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <stdexcept>

RidgeReadout::RidgeReadout(double alpha, bool include_bias, Solver solver, double tolerance)
    : alpha(alpha), include_bias(include_bias), solver(solver), tolerance(tolerance) {}

void RidgeReadout::fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) {
  Eigen::Index n_samples = states.rows();
  Eigen::Index n_features = states.cols();
  Eigen::Index n_outputs = targets.cols();
  Eigen::Index dim = n_features + (include_bias ? 1 : 0);

  if (solver == CONJUGATE_GRADIENT_IMPLICIT) {
    // Matrix-Free CG with Zero-Copy
    // 1. Construct XtY directly
    Eigen::MatrixXd XtY(dim, n_outputs);

    // Top part: states^T * targets
    XtY.topRows(n_features).noalias() = states.transpose() * targets;

    // Bottom part (bias): sum(targets)
    if (include_bias) {
      XtY.bottomRows(1) = targets.colwise().sum();
    }

    // 2. Solve using RidgeLinearOperator with virtual bias
    W_out.resize(dim, n_outputs);
    RidgeLinearOperator<Eigen::MatrixXd> ridge_op(states, alpha, include_bias);
    Eigen::ConjugateGradient<RidgeLinearOperator<Eigen::MatrixXd>, Eigen::Lower | Eigen::Upper,
                             Eigen::IdentityPreconditioner>
        cg;
    cg.compute(ridge_op);
    cg.setTolerance(tolerance);

    for (int i = 0; i < n_outputs; ++i) {
      W_out.col(i) = cg.solve(XtY.col(i));
    }

  } else {
    // Explicit Matrix Formation with Zero-Copy
    Eigen::MatrixXd XtX(dim, dim);
    Eigen::MatrixXd XtY(dim, n_outputs);

    // 1. Fill XtX block-wise
    // Top-Left: states^T * states
    XtX.topLeftCorner(n_features, n_features).noalias() = states.transpose() * states;

    if (include_bias) {
      // Calculate column sums once
      Eigen::VectorXd col_sums = states.colwise().sum();

      // Top-Right: col_sums
      XtX.topRightCorner(n_features, 1) = col_sums;

      // Bottom-Left: col_sums^T
      XtX.bottomLeftCorner(1, n_features) = col_sums.transpose();

      // Bottom-Right: n_samples
      XtX(n_features, n_features) = static_cast<double>(n_samples);
    }

    // 2. Add Regularization
    for (int i = 0; i < dim; ++i) {
      XtX(i, i) += alpha;
    }

    // 3. Fill XtY block-wise
    XtY.topRows(n_features).noalias() = states.transpose() * targets;
    if (include_bias) {
      XtY.bottomRows(1) = targets.colwise().sum();
    }

    // 4. Solve
    if (solver == CONJUGATE_GRADIENT) {
      Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower | Eigen::Upper> cg;
      cg.compute(XtX);
      cg.setTolerance(tolerance);
      W_out = cg.solve(XtY);
    } else {
      // Default / Cholesky
      W_out = XtX.ldlt().solve(XtY);
    }
  }
}

void RidgeReadout::partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) {
  // RidgeReadout is a batch-trained method, so partialFit is not applicable.
  throw std::logic_error("partialFit is not implemented for RidgeReadout");
}

Eigen::MatrixXd RidgeReadout::predict(const Eigen::MatrixXd &states) {
  Eigen::Index n_features = states.cols();

  if (include_bias) {
    // W_out = [W_weights; W_bias]
    // Result = states * W_weights + W_bias (broadcast)

    // 1. Compute states * W_weights
    Eigen::MatrixXd result = states * W_out.topRows(n_features);

    // 2. Add bias
    result.rowwise() += W_out.row(n_features);

    return result;
  } else {
    return states * W_out;
  }
}
