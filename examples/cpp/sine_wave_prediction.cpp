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
    double input_scaling, // New parameter
    double ridge_alpha
) {
    std::cout << "--- Running experiment with include_bias=" << (include_bias ? "true" : "false") << ", input_scaling=" << input_scaling << " ---" << std::endl;

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
        input_scaling, // Pass new parameter
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
    model.fit(train_input, train_target /*, washout_len=100 */); // Experiment with washout_len

    std::cout << "--- Predicting with ESN ---" << std::endl;
    Eigen::MatrixXd predictions = model.predict(test_input);

    // --- 3. Evaluate Results ---
    Eigen::MatrixXd diff = predictions.topRows(test_target.rows()) - test_target;
    double mse = diff.array().square().mean();
    std::cout << "Mean Squared Error (include_bias=" << (include_bias ? "true" : "false") << ", input_scaling=" << input_scaling << "): " << mse << std::endl;

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
    const double input_scaling = 1.0; // New parameter
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
        input_scaling,
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
        input_scaling,
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

    std::cout << "\n--- Further Tuning Suggestions ---" << std::endl;
    std::cout << "If predictions are still poor, consider tuning the following parameters:" << std::endl;
    std::cout << "1. spectral_radius: Try values like 0.7, 0.8, 0.9 (must be < 1.0 for stability)." << std::endl;
    std::cout << "2. leak_rate: Experiment with 0.1, 0.5, 0.7." << std::endl;
    std::cout << "3. ridge_alpha: Try 1e-6, 1e-3." << std::endl;
    std::cout << "4. input_scaling: Experiment with 0.1, 0.5, 1.0, 2.0." << std::endl;
    std::cout << "5. n_neurons: Increase for more complex tasks, decrease for simpler ones." << std::endl;

    return 0;
}
