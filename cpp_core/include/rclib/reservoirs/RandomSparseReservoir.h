#pragma once

#include "rclib/Reservoir.h"

#include <Eigen/Sparse>

class RandomSparseReservoir : public Reservoir {
public:
  RandomSparseReservoir(int n_neurons, double spectral_radius, double sparsity, double leak_rate, double input_scaling,
                        bool include_bias = false, unsigned int seed = 42);

  Eigen::MatrixXd advance(const Eigen::MatrixXd &input) override;
  void resetState() override;
  const Eigen::MatrixXd &getState() const override;

private:
  void initialize_W_in(int input_dim);

  int n_neurons;
  double spectral_radius;
  double sparsity;
  double leak_rate;
  double input_scaling; // New member variable
  bool include_bias;
  unsigned int seed;
  bool W_in_initialized;

  Eigen::MatrixXd state;
  Eigen::SparseMatrix<double> W_res;
  Eigen::MatrixXd W_in;
  Eigen::RowVectorXd bias;
};
