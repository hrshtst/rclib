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
    double input_scaling,
    double ridge_alpha,
    bool reset_state_before_predict // New parameter
) {
    std::cout << "--- Running experiment with include_bias=" << (include_bias ? "true" : "false") << ", input_scaling=" << input_scaling << ", reset_state_before_predict=" << (reset_state_before_predict ? "true" : "false") << " ---" << std::endl;

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
        input_scaling,
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
    Eigen::MatrixXd predictions = model.predict(test_input, reset_state_before_predict);

    // --- 3. Evaluate Results ---
    Eigen::MatrixXd diff = predictions.topRows(test_target.rows()) - test_target;
    double mse = diff.array().square().mean();
    std::cout << "Mean Squared Error (include_bias=" << (include_bias ? "true" : "false") << ", input_scaling=" << input_scaling << ", reset_state_before_predict=" << (reset_state_before_predict ? "true" : "false") << "): " << mse << std::endl;

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
    const double input_scaling = 1.0;
    const double ridge_alpha = 1e-4;

    std::vector<bool> reset_options = {true, false};
    std::map<std::tuple<bool, bool>, double> results;

    for (bool reset_state_before_predict : reset_options) {
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
            ridge_alpha,
            reset_state_before_predict
        );
        results[std::make_tuple(true, reset_state_before_predict)] = mse_with_bias;

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
            ridge_alpha,
            reset_state_before_predict
        );
        results[std::make_tuple(false, reset_state_before_predict)] = mse_without_bias;
    }

    std::cout << "\n--- Comparison of Results (MSE) ---" << std::endl;
    std::cout << "Bias | ResetPredict | MSE" << std::endl;
    std::cout << "--------------------------" << std::endl;
    for (const auto& pair : results) {
        std::string bias_str = std::get<0>(pair.first) ? "True " : "False";
        std::string reset_str = std::get<1>(pair.first) ? "True         " : "False        ";
        std::cout << bias_str << " | " << reset_str << " | " << pair.second << std::endl;
    }

    std::cout << "\n--- Further Tuning Suggestions ---" << std::endl;
    std::cout << "If predictions are still poor, consider tuning the following parameters:" << std::endl;
    std::cout << "1. spectral_radius: Try values like 0.7, 0.8, 0.9 (must be < 1.0 for stability)." << std::endl;
    std::cout << "2. leak_rate: Experiment with 0.1, 0.5, 0.7." << std::endl;
    std::cout << "3. ridge_alpha: Try 1e-6, 1e-3." << std::endl;
    std::cout << "4. input_scaling: Experiment with 0.1, 0.5, 1.0, 2.0." << std::endl;
    std::cout << "5. n_neurons: Increase for more complex tasks, decrease for simpler ones." << std::endl;
    std::cout << "6. reset_state_before_predict: Experiment with true/false." << std::endl;

    return 0;
}
