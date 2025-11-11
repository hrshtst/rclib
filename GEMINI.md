# **Project Specification: "rclib" (Reservoir Computing Library)**

## **1. Project Goal**

The primary goal is to create a high-performance, scalable, and general-purpose reservoir computing framework, "rclib." It must be capable of handling both small-scale (100 neurons) and very large-scale (40,000+ neurons) networks, as well as deep (stacked) and parallel reservoir architectures.

The core library will be implemented in C++ for maximum computational performance, available as both a standalone C++ library and with comprehensive Python bindings for the machine learning community.

## **2. Core Technology Stack**

* **Core Logic:** C++ (C++17 or newer).
* **Build System:** `CMake`. This will manage the C++ core, C++ tests, and the `pybind11` module compilation.
* **Python Interface:** Python 3.10+.
* **Python Project Management:** `uv` (preferred over venv/pip).
* **Python Formatter/Linter:** `Ruff`.
* **Python Dependencies:** `numpy`. The Python API will primarily accept and return NumPy arrays.
* **Testing:**
  * **C++:** `Catch2` (preferred).
  * **Python:** `pytest`.
* **Fixed C++ Dependencies:**
  * `Eigen`: `3.4.1` (for C++ Linear Algebra)
  * `Catch2`: `v3.11.0` (for C++ Testing)
  * `pybind11`: `v3.0.1` (for Python Bindings)

## **3. Build & Dependency Management**

This project uses `git submodule` to manage C++ dependencies. They will be downloaded into `cpp_core/third_party/` and built from source using `CMake`.

**Instructions for Gemini CLI (or User):** When scaffolding, the following commands must be used to add the dependencies.

``` shell
# Ensure you are in the project root 'rclib/'
git submodule add https://gitlab.com/libeigen/eigen.git cpp_core/third_party/eigen
git submodule add https://github.com/catchorg/Catch2.git cpp_core/third_party/catch2
git submodule add https://github.com/pybind/pybind11.git cpp_core/third_party/pybind11
```

The root `CMakeLists.txt` will then add these directories:
* `add_subdirectory(cpp_core/third_party/catch2)`
* `add_subdirectory(cpp_core/third_party/pybind11)`
* `add_subdirectory(cpp_core/third_party/eigen)` (Eigen is header-only but this helps CMake find it)

## **4. Key Architectural Principles**

1. **Modularity:** The **Reservoir** and **Readout** components MUST be implemented as separate, swappable modules.
2. **Performance:** C++ implementations should prioritize computational efficiency and memory management, especially for large, sparse matrices (`Eigen::SparseMatrix`).
3. **Scalability:** The design must naturally support:
   * **Large Reservoirs:** Efficiently handle single reservoirs with 40,000+ neurons.
   * **Deep ESNs:** Allow straightforward serial stacking of reservoir layers.
   * **Parallel ESNs:** Allow multiple reservoirs to be processed in parallel, with their states concatenated before reaching the readout.
4. **Configurability:** All key parameters (spectral radius, sparsity, leak rate, regularization, **bias inclusion**, etc.) MUST be easily configurable from both the C++ and Python APIs.

## **5. Proposed Directory Structure**

```
rclib/
├── .gitmodules            # <- Will be created by git submodule
├── CMakeLists.txt         # Main CMake build file
├── README.md
├── GEMINI.md              # This file
├── pyproject.toml         # For uv and Ruff configuration
├── cpp_core/              # C++ source
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── rclib/
│   │   │   ├── Reservoir.h
│   │   │   ├── Readout.h
│   │   │   ├── Model.h
│   │   │   └── reservoirs/
│   │   │       ├── RandomSparseReservoir.h
│   │   │       └── NvarReservoir.h
│   │   │   └── readouts/
│   │   │       ├── RidgeReadout.h
│   │   │       ├── RlsReadout.h
│   │   │       └── LmsReadout.h
│   ├── src/
│   │   ├── Reservoir.cpp
│   │   ├── ... (other .cpp files)
│   └── third_party/
│       ├── eigen/           # <- Git Submodule
│       ├── pybind11/        # <- Git Submodule
│       └── catch2/          # <- Git Submodule
├── python/                # Python package
│   ├── CMakeLists.txt     # For pybind11 module
│   ├── rclib/
│   │   ├── __init__.py
│   │   ├── model.py       # Python wrapper class for Model
│   │   └── _rclib.cpp       # Pybind11 binding definitions
│   └── setup.py           # To build and install the Python package
├── tests/
│   ├── cpp/               # C++ tests (using Catch2)
│   │   ├── test_reservoir.cpp
│   │   └── ...
│   └── python/            # Python tests
│       ├── test_model.py
│       └── ...
└── examples/
    ├── cpp/               # C++ examples
    │   ├── quick_start.cpp
    │   └── mackey_glass.cpp
    └── python/            # Python examples
        ├── quick_start.py
        └── mackey_glass.py
```

## **5. C++ API Design (High-Level)**

### **CMake Configuration (`cpp_core/CMakeLists.txt`)**

* The C++ core will be built as a static library (e.g., rclib_core).
* It must use `add_subdirectory` for `catch2` and `pybind11` (as specified in Section 3).
* It must find `Eigen` via `find_package(Eigen REQUIRED)` or by linking to the `Eigen3::Eigen` target if `add_subdirectory(eigen)` provides it.

### **Reservoir Interface (`Reservoir.h`)**

* An abstract base class `Reservoir` will define the interface.
* **Key methods:**
  * `virtual Eigen::MatrixXd& advance(const Eigen::MatrixXd& input) = 0;` (Advances state by one step).
  * `virtual void resetState() = 0;` (Resets internal state to zero).
  * `virtual const Eigen::MatrixXd& getState() const = 0;`
* **Implementations:**
  * `RandomSparseReservoir`: Standard ESN. Configurable sparsity, spectral radius, leak rate, weight scaling, and **optional input/reservoir bias**. Uses `Eigen::SparseMatrix` for `W_res`.
  * `NvarReservoir`: Nonlinear Vector Autoregression. Will use a different state update logic based on input lags.

### **Readout Interface (`Readout.h`)**

* An abstract base class `Readout` will define the interface.
* **Key methods:**
  * `virtual void fit(const Eigen::MatrixXd& states, const Eigen::MatrixXd& targets) = 0;` (Batch training).
  * `virtual void partialFit(const Eigen::MatrixXd& state, const Eigen::MatrixXd& target) = 0;` (Online training, for RLS/LMS).
  * `virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& states) = 0;`
* **Implementations:**
  * `RidgeReadout`: Batch-trained Ridge Regression. Solves `(X^T * X + lambda * I)^-1 * X^T * Y`. Must support an **optional bias (intercept) term**.
  * `RlsReadout`: Recursive Least Squares algorithm for online learning.
  * `LmsReadout`: Least Mean Squares algorithm for online learning.

### **Model Class (`Model.h`)**

* Manages the collection of reservoirs and the final readout.
* Will store `std::vector<std::shared_ptr<Reservoir>>` reservoirs.
* Will store `std::shared_ptr<Readout>` readout.
* **Key methods:**
  * `addReservoir(std::shared_ptr<Reservoir> res, std::string connection_type = "serial");` (Connection type can be 'serial' for deep, 'parallel' for parallel).
  * `setReadout(std::shared_ptr<Readout> readout);`
  * `fit(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets);`
  * `predict(const Eigen::MatrixXd& inputs);`
  * `predictOnline(const Eigen::MatrixXd& input);` (For online prediction).

## **6. Python API (Pythonic Interface)**

The `pybind11` module (`_rclib.cpp`) will expose the C++ classes. A Python wrapper (`model.py`) will provide a `scikit-learn`-style API.

**Example Python Usage (`model.py`):**

``` python
# This is the target API we are aiming for.

from rclib import reservoirs, readouts
from rclib.model import ESN

# 1. Configure Reservoir(s)
res1 = reservoirs.RandomSparse(
    n_neurons=1000,
    spectral_radius=0.9,
    sparsity=0.1,
    leak_rate=0.3,
    include_bias=True
)

res2 = reservoirs.RandomSparse(
    n_neurons=2000,
    spectral_radius=1.1,
    sparsity=0.05,
    leak_rate=0.1,
    include_bias=False
)

# 2. Configure Readout
# 'rls' and 'lms' will also be available
readout = readouts.Ridge(alpha=1e-8, include_bias=True)

# 3. Configure Model (Deep ESN)
# 'parallel' connection will also be an option
model = ESN(connection_type='serial')
model.add_reservoir(res1)
model.add_reservoir(res2) # Output of res1 feeds into res2
model.set_readout(readout)

# 4. Fit and Predict
# X_train and Y_train are numpy arrays
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

## **7. Python Example Execution**

To execute Python examples or tests within this project, use the following command structure:

```bash
rm -rf .venv && uv sync --no-cache && uv run python examples/python/your_script_name.py
```

Replace `examples/python/your_script_name.py` with the actual path to the Python script you wish to run. This command ensures that the project's Python environment is correctly set up and dependencies are managed by `uv` before execution.
