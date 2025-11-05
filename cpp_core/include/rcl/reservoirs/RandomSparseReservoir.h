#pragma once

#include "rcl/Reservoir.h"
#include <Eigen/Sparse>

class RandomSparseReservoir : public Reservoir {
public:
    RandomSparseReservoir(int n_neurons, double spectral_radius, double sparsity, double leak_rate, bool include_bias = false);

    Eigen::MatrixXd& advance(const Eigen::MatrixXd& input) override;
    void resetState() override;
    const Eigen::MatrixXd& getState() const override;

private:
    int n_neurons;
    double spectral_radius;
    double sparsity;
    double leak_rate;
    bool include_bias;

    Eigen::MatrixXd state;
    Eigen::SparseMatrix<double> W_res;
    Eigen::MatrixXd W_in;
    Eigen::RowVectorXd bias;
};
