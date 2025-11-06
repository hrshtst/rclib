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

double run_experiment(
    bool include_bias,
    int n_total_samples,
    int n_train_samples,
    double noise_amplitude,
    int n_neurons,
    double spectral_radius,
    double sparsity,
    double leak_rate,
    double ridge_alpha
) {
    std::cout << "--- Running experiment with include_bias=" << (include_bias ? "true" : "false") << " ---" << std::endl;

    // --- 1. Data Generation ---
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

    // --- 2. Instantiate, Train, and Predict ---
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

    // --- 3. Evaluate Results ---
    Eigen::MatrixXd diff = predictions.topRows(test_target.rows()) - test_target;
    double mse = diff.array().square().mean();
    std::cout << "Mean Squared Error (include_bias=" << (include_bias ? "true" : "false") << "): " << mse << std::endl;

    std::cout << "\nSample Predictions vs. True Targets:" << std::endl;
    for (int i = 0; i < std::min(10, (int)test_target.rows()); ++i) {
        std::cout << "True: " << test_target(i, 0) << ", Predicted: " << predictions(i, 0) << std::endl;
    }
    std::cout << std::endl;

    return mse;
}

int main() {
    // --- Configuration Parameters ---
    const int n_total_samples = 1000;
    const int n_train_samples = 800;
    const double noise_amplitude = 0.05;

    const int n_neurons = 2000;
    const double spectral_radius = 0.99;
    const double sparsity = 0.02;
    const double leak_rate = 0.2;
    const double ridge_alpha = 1e-4;

    // Run with bias
    double mse_with_bias = run_experiment(
        true,
        n_total_samples,
        n_train_samples,
        noise_amplitude,
        n_neurons,
        spectral_radius,
        sparsity,
        leak_rate,
        ridge_alpha
    );

    // Run without bias
    double mse_without_bias = run_experiment(
        false,
        n_total_samples,
        n_train_samples,
        noise_amplitude,
        n_neurons,
        spectral_radius,
        sparsity,
        leak_rate,
        ridge_alpha
    );

    std::cout << "\n--- Comparison of Results ---" << std::endl;
    std::cout << "MSE with bias: " << mse_with_bias << std::endl;
    std::cout << "MSE without bias: " << mse_without_bias << std::endl;

    if (mse_with_bias < mse_without_bias) {
        std::cout << "Conclusion: Model performed better with bias." << std::endl;
    } else if (mse_without_bias < mse_with_bias) {
        std::cout << "Conclusion: Model performed better without bias." << std::endl;
    } else {
        std::cout << "Conclusion: Model performance was similar with and without bias." << std::endl;
    }

    return 0;
}
