# Testing & Quality Roadmap

This document outlines the planned expansion of the integration and regression test suite for `rclib`. These tests aim to ensure mathematical correctness, architectural robustness, and numerical stability as the library evolves.

## 1. NVAR Pipeline Integration
While unit tests exist for the `NvarReservoir` class, a full model-level integration test is needed.
*   **Target:** Next-Generation Reservoir Computing (NVAR) architecture.
*   **Test Scenario:** Train an NVAR-based model on a chaotic system (e.g., Lorenz Attractor).
*   **Verification:** Ensure the model correctly constructs the non-linear feature vector and achieves a low MSE without random weight matrices.

## 2. Generative Stability
Generative prediction (closed-loop) is sensitive to accumulated errors.
*   **Target:** `predict_generative` method.
*   **Test Scenario:** Train on a stable periodic signal, then generate 1000+ steps autonomously.
*   **Verification:** Assert that the generated signal remains within a valid range (e.g., $[-2, 2]$) and maintains the target frequency without diverging or decaying to zero.

## 3. Deep ESN Architecture (Serial)
Verifying the stacking logic for deep networks.
*   **Target:** `serial` connection type in the `Model` class.
*   **Test Scenario:** Create a stack of 3+ reservoirs where the state of $Res_n$ serves as the input to $Res_{n+1}$.
*   **Verification:** Confirm that state propagation and concatenation are handled correctly and that the model achieves better performance on complex tasks compared to a shallow reservoir of the same total size.

## 4. Multidimensional I/O Robustness
Ensuring consistent handling of high-dimensional data across the C++/Python boundary.
*   **Target:** Linear algebra and data-passing logic.
*   **Test Scenario:** Implement a "Multiple-Input Multiple-Output" (MIMO) task with 5D input and 5D target vectors.
*   **Verification:** Validate that matrix dimensions, transpositions, and indexing are correctly handled for non-scalar time series.

## 5. Numerical Edge Cases & Stress Testing
Testing the library's resilience to extreme parameters.
*   **Target:** Error handling and stability.
*   **Test Scenario:**
    *   Large spectral radius ($> 1.0$) leading to saturation.
    *   Small RLS forgetting factors causing potential matrix ill-conditioning.
    *   Zero-length or empty input data.
*   **Verification:** Ensure the library throws meaningful exceptions (via `std::runtime_error`) or remains stable through regularization, rather than crashing or returning `NaN`.

## 6. Performance Benchmarking Regression
*   **Target:** Computational efficiency.
*   **Test Scenario:** Measure execution time for fitting a 40,000 neuron reservoir.
*   **Verification:** Establish a baseline and assert that new commits do not deviate by more than 10% from the baseline time on identical CI hardware.
