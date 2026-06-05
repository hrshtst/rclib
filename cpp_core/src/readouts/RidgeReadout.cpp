#include "rclib/readouts/RidgeReadout.h"

#include "rclib/readouts/RidgeLinearOperator.h"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <stdexcept>

RidgeReadout::RidgeReadout(double alpha, bool include_bias, Solver solver, double tolerance)
    : alpha(alpha), include_bias(include_bias), solver(solver), effective_solver(solver), tolerance(tolerance) {
  if (alpha < 0.0) {
    throw std::invalid_argument("alpha must be non-negative.");
  }
  if (tolerance <= 0.0) {
    throw std::invalid_argument("tolerance must be positive.");
  }
}

void RidgeReadout::fit(const Eigen::MatrixXd &states, const Eigen::MatrixXd &targets) {
  if (states.rows() == 0 || states.cols() == 0) {
    throw std::invalid_argument("states must be a non-empty 2D matrix.");
  }
  if (targets.rows() != states.rows()) {
    throw std::invalid_argument("targets must have the same number of rows as states.");
  }
  if (targets.cols() == 0) {
    throw std::invalid_argument("targets must have at least one column.");
  }

  Eigen::Index n_samples = states.rows();
  Eigen::Index n_features = states.cols();
  Eigen::Index n_outputs = targets.cols();
  Eigen::Index dim = n_features + (include_bias ? 1 : 0);

  if (solver == AUTO) {
    if (n_features >= 4000) {
      effective_solver = CONJUGATE_GRADIENT_IMPLICIT;
    } else if (n_features > n_samples && n_features > 100) {
      effective_solver = DUAL_CHOLESKY;
    } else {
      effective_solver = CHOLESKY;
    }
  } else {
    effective_solver = solver;
  }

  if (effective_solver == CONJUGATE_GRADIENT_IMPLICIT) {
    // Matrix-Free CG with Zero-Copy
    // 1. Construct XtY directly
    Eigen::MatrixXd XtY(dim, n_outputs);

#ifdef RCLIB_USE_OPENMP
#  ifdef RCLIB_ADAPTIVE_PARALLELIZATION
#    pragma omp parallel sections if (dim > 1000)
#  else
#    pragma omp parallel sections
#  endif
    {
#  pragma omp section
      {
        XtY.topRows(n_features).noalias() = states.transpose() * targets;
      }
#  pragma omp section
      {
        if (include_bias) {
          XtY.bottomRows(1) = targets.colwise().sum();
        }
      }
    }
#else
    XtY.topRows(n_features).noalias() = states.transpose() * targets;
    if (include_bias) {
      XtY.bottomRows(1) = targets.colwise().sum();
    }
#endif

    // 2. Solve using RidgeLinearOperator with virtual bias
    W_out.resize(dim, n_outputs);
    RidgeLinearOperator<Eigen::MatrixXd> ridge_op(states, alpha, include_bias);
    Eigen::ConjugateGradient<RidgeLinearOperator<Eigen::MatrixXd>, Eigen::Lower | Eigen::Upper,
                             Eigen::IdentityPreconditioner>
        cg;
    cg.compute(ridge_op);
    cg.setTolerance(tolerance);

#ifdef RCLIB_USE_OPENMP
#  ifdef RCLIB_ADAPTIVE_PARALLELIZATION
#    pragma omp parallel for if (dim > 1000)
#  else
#    pragma omp parallel for
#  endif
#endif
    for (int i = 0; i < n_outputs; ++i) {
      W_out.col(i) = cg.solve(XtY.col(i));
    }

  } else if (effective_solver == DUAL_CHOLESKY) {
    // Dual Ridge Regression (N > T)
    // 1. Form Kernel matrix K = X_aug * X_aug^T + alpha*I (T x T)
    // Use GEMM instead of rankUpdate for better parallelization
#ifdef RCLIB_USE_OPENMP
    int old_threads = Eigen::nbThreads();
#  ifdef RCLIB_ADAPTIVE_PARALLELIZATION
    if (n_samples <= 1000) {
      Eigen::setNbThreads(1);
    }
#  endif
#endif
    Eigen::MatrixXd K = states * states.transpose();
#ifdef RCLIB_USE_OPENMP
#  ifdef RCLIB_ADAPTIVE_PARALLELIZATION
    Eigen::setNbThreads(old_threads);
#  endif
#endif
    if (include_bias) {
      K.array() += 1.0;
    }
    K.diagonal().array() += alpha;

    // 2. Solve K * beta = targets
    Eigen::MatrixXd beta = K.selfadjointView<Eigen::Lower>().ldlt().solve(targets);

    // 3. Compute W_out = X_aug^T * beta
    W_out.resize(dim, n_outputs);
    W_out.topRows(n_features).noalias() = states.transpose() * beta;
    if (include_bias) {
      W_out.bottomRows(1).noalias() = beta.colwise().sum();
    }

  } else {
    // Explicit Matrix Formation (Primal)
    Eigen::MatrixXd XtX(dim, dim);
    Eigen::MatrixXd XtY(dim, n_outputs);

    // 1. Fill XtX using GEMM
    // Although rankUpdate is theoretically more efficient (only computes half),
    // Eigen's GEMM is much better parallelized in the absence of an external BLAS.
#ifdef RCLIB_USE_OPENMP
    int old_threads = Eigen::nbThreads();
#  ifdef RCLIB_ADAPTIVE_PARALLELIZATION
    if (dim <= 1000) {
      Eigen::setNbThreads(1);
    }
#  endif
#endif
    XtX.topLeftCorner(n_features, n_features).noalias() = states.transpose() * states;
#ifdef RCLIB_USE_OPENMP
#  ifdef RCLIB_ADAPTIVE_PARALLELIZATION
    Eigen::setNbThreads(old_threads);
#  endif
#endif

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
    XtX.diagonal().array() += alpha;

    // 3. Fill XtY block-wise
#ifdef RCLIB_USE_OPENMP
#  ifdef RCLIB_ADAPTIVE_PARALLELIZATION
#    pragma omp parallel sections if (dim > 1000)
#  else
#    pragma omp parallel sections
#  endif
    {
#  pragma omp section
      {
        XtY.topRows(n_features).noalias() = states.transpose() * targets;
      }
#  pragma omp section
      {
        if (include_bias) {
          XtY.bottomRows(1) = targets.colwise().sum();
        }
      }
    }
#else
    XtY.topRows(n_features).noalias() = states.transpose() * targets;
    if (include_bias) {
      XtY.bottomRows(1) = targets.colwise().sum();
    }
#endif

    // 4. Solve
    if (effective_solver == CONJUGATE_GRADIENT) {
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

void RidgeReadout::partialFit(const Eigen::MatrixXd & /*state*/, const Eigen::MatrixXd & /*target*/) {
  // RidgeReadout is a batch-trained method, so partialFit is not applicable.
  throw std::logic_error("partialFit is not implemented for RidgeReadout");
}

Eigen::MatrixXd RidgeReadout::predict(const Eigen::MatrixXd &states) {
  if (W_out.size() == 0) {
    throw std::runtime_error("RidgeReadout must be fit before predict.");
  }
  if (states.rows() == 0 || states.cols() == 0) {
    throw std::invalid_argument("states must be a non-empty 2D matrix.");
  }
  Eigen::Index n_features = states.cols();
  Eigen::Index expected_features = W_out.rows() - (include_bias ? 1 : 0);
  if (n_features != expected_features) {
    throw std::invalid_argument("states column count does not match fitted RidgeReadout.");
  }

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
