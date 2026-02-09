# rclib: Reservoir Computing Library

**rclib** is a high-performance, scalable, and general-purpose reservoir computing framework implemented in C++ with Python bindings. It is designed to handle both small-scale networks and medium-to-large scale architectures, supporting deep (stacked) and parallel reservoir configurations.

## Project Goals

*   **Performance:** Core logic in C++17 using Eigen for linear algebra.
*   **Scalability:** Efficient handling of sparse reservoirs and complex architectures.
*   **Flexibility:** Modular design separating Reservoirs and Readouts.
*   **Ease of Use:** Pythonic interface via `pybind11` and `scikit-learn` style API.
*   **Reproducibility:** Deterministic results via explicit seeding of random reservoirs.

## Getting Started

To get started with `rclib`, please refer to the [User Guide](user_guide/index.md) for installation and basic usage instructions.

For a deep dive into the underlying mathematics and architecture, check the [Theory](theory/index.md) section.

Detailed class and function documentation can be found in the [API Reference](api/index.md).
