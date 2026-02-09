#include "rclib/readouts/RidgeReadout.h"

#include "rclib/readouts/RidgeLinearOperator.h"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <stdexcept>

RidgeReadout::RidgeReadout(double alpha, bool include_bias, Solver solver)
    : alpha(alpha), include_bias(include_bias), solver(solver) {}

void RidgeReadout::fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) {
  Eigen::MatrixXd X = states;
  if (include_bias) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    X.col(X.cols() - 1) = Eigen::VectorXd::Ones(X.rows());
  }

  // Common: Prepare target vector (assuming single output for now or handling multi-output loop inside solver?)
  // Eigen's solvers handle multi-rhs, so XtY is fine.

  if (solver == CONJUGATE_GRADIENT_IMPLICIT) {
    // Matrix-Free CG
    Eigen::MatrixXd XtY = X.transpose() * targets;

    // We need to solve for each column of W_out (each output dimension)
    W_out.resize(X.cols(), targets.cols());

    RidgeLinearOperator<Eigen::MatrixXd> ridge_op(X, alpha);
    Eigen::ConjugateGradient<RidgeLinearOperator<Eigen::MatrixXd>, Eigen::Lower | Eigen::Upper,
                             Eigen::IdentityPreconditioner>
        cg;
    cg.compute(ridge_op);

    for (int i = 0; i < targets.cols(); ++i) {
      W_out.col(i) = cg.solve(XtY.col(i));
    }

  } else {
    // Explicit Matrix Formation
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());
    Eigen::MatrixXd A = XtX + alpha * I;
    Eigen::MatrixXd XtY = X.transpose() * targets;

    if (solver == CONJUGATE_GRADIENT) {
      Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower | Eigen::Upper> cg;
      cg.compute(A);
      W_out = cg.solve(XtY);
    } else {
      // Default / Cholesky
      W_out = A.ldlt().solve(XtY);
    }
  }
}

void RidgeReadout::partialFit(const Eigen::MatrixXd &state, const Eigen::MatrixXd &target) {
  // RidgeReadout is a batch-trained method, so partialFit is not applicable.
  // We could throw an error or do nothing.
  // For now, let's throw an error.
  throw std::logic_error("partialFit is not implemented for RidgeReadout");
}

Eigen::MatrixXd RidgeReadout::predict(const Eigen::MatrixXd &states) {
  Eigen::MatrixXd X = states;
  if (include_bias) {
    X.conservativeResize(X.rows(), X.cols() + 1);
    X.col(X.cols() - 1) = Eigen::VectorXd::Ones(X.rows());
  }
  return X * W_out;
}
