#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include <numeric>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "rcl/Model.h"
#include "rcl/reservoirs/RandomSparseReservoir.h"
#include "rcl/readouts/RidgeReadout.h"

int main() {
    // --- 1. Configuration Parameters ---
    const int n_total_samples = 1000;
    const int n_train_samples = 800;
    const double noise_amplitude = 0.05;

    const int n_neurons = 2000;
    const double spectral_radius = 0.99;
    const double sparsity = 0.02;
    const double leak_rate = 0.2;
    const double ridge_alpha = 1e-4;
    const bool include_bias = true;

    std::cout << "--- Generating Data ---" << std::endl;
    // Generate time_np
    Eigen::VectorXd time_np(n_total_samples);
    for (int i = 0; i < n_total_samples; ++i) {
        time_np(i) = static_cast<double>(i) * 80.0 / (n_total_samples - 1);
    }

    // Generate clean_data
    Eigen::VectorXd clean_data = time_np.array().sin();

    // Generate noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    Eigen::VectorXd noise(n_total_samples);
    for (int i = 0; i < n_total_samples; ++i) {
        noise(i) = noise_amplitude * d(gen);
    }

    // Combine and reshape data
    Eigen::MatrixXd data = (clean_data + noise).reshaped(n_total_samples, 1);

    Eigen::MatrixXd input_data = data.topRows(n_total_samples - 1);
    Eigen::MatrixXd target_data = data.bottomRows(n_total_samples - 1);

    Eigen::MatrixXd train_input = input_data.topRows(n_train_samples);
    Eigen::MatrixXd train_target = target_data.topRows(n_train_samples);
    Eigen::MatrixXd test_input = data.middleRows(n_train_samples, n_total_samples - n_train_samples - 1);
    Eigen::MatrixXd test_target = data.bottomRows(n_total_samples - n_train_samples - 1);

    // --- 3. Instantiate, Train, and Predict ---
    std::cout << "--- Initializing ESN ---" << std::endl;
    auto reservoir = std::make_shared<RandomSparseReservoir>(
        n_neurons,
        spectral_radius,
        sparsity,
        leak_rate,
        include_bias
    );

    auto readout = std::make_shared<RidgeReadout>(
        ridge_alpha,
        include_bias
    );

    Model model;
    model.addReservoir(reservoir);
    model.setReadout(readout);

    std::cout << "--- Fitting ESN ---" << std::endl;
    model.fit(train_input, train_target);

    std::cout << "--- Predicting with ESN ---" << std::endl;
    Eigen::MatrixXd predictions = model.predict(test_input);

    // --- 4. Evaluate Results ---
    Eigen::MatrixXd diff = predictions.topRows(test_target.rows()) - test_target;
    double mse = diff.array().square().mean();
    std::cout << "Mean Squared Error: " << mse << std::endl;

    std::cout << "\nSample Predictions vs. True Targets:" << std::endl;
    for (int i = 0; i < std::min(10, (int)test_target.rows()); ++i) {
        std::cout << "True: " << test_target(i, 0) << ", Predicted: " << predictions(i, 0) << std::endl;
    }

    return 0;
}
