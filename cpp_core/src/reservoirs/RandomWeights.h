#pragma once

// Weight generation shared by the reservoirs with random sparse recurrence, so that the
// same hyperparameters and seed give the same weights in each of them. Internal to rclib_core.

#include "rclib/reservoirs/RandomSparseReservoir.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace random_weights {

/// The recurrent weights and the bias drawn for a reservoir.
struct RecurrentWeights {
  Eigen::SparseMatrix<double> W_res; // compressed, scaled to the spectral radius
  Eigen::RowVectorXd bias;           // zero without include_bias
};

/// Draws W_res from mt19937(seed), scales it to spectral_radius (left unscaled when
/// spectral_radius is 0 or the estimate is below 1e-9), then draws the bias from the same
/// generator and multiplies it by bias_scaling. The power iteration start vector comes from
/// mt19937(seed + 2). `context` names the reservoir in error messages.
RecurrentWeights generateRecurrentWeights(int n_neurons, double spectral_radius, double sparsity, bool include_bias,
                                          unsigned int seed, RandomSparseReservoir::SpectralRadiusMethod method,
                                          double bias_scaling, const char *context);

/// Draws W_in, (input_dim, n_neurons), uniformly from [-1, 1] with mt19937(seed + 1) and
/// multiplies it by input_scaling.
Eigen::MatrixXd generateInputWeights(int input_dim, int n_neurons, double input_scaling, unsigned int seed);

} // namespace random_weights
