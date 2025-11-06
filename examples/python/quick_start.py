import numpy as np

from rcl import ESN, readouts, reservoirs

# 1. Create some dummy data
X_train = np.linspace(0, 1, 100).reshape(-1, 1)
y_train = np.sin(X_train * 10)

X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test = np.sin(X_test * 10)

# 2. Configure Reservoir
res = reservoirs.RandomSparse(n_neurons=1000, spectral_radius=0.9, sparsity=0.1, leak_rate=0.3, include_bias=True)

# 3. Configure Readout
readout = readouts.Ridge(alpha=1e-8, include_bias=True)

# 4. Configure Model
model = ESN()
model.add_reservoir(res)
model.set_readout(readout)

# 5. Fit and Predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Plot the results
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_test, label="True")
    plt.plot(X_test, y_pred, label="Predicted")
    mse = np.mean((y_pred - y_test) ** 2)
    plt.text(0.05, 0.95, f"MSE: {mse:.4e}", transform=plt.gca().transAxes, fontsize=12, verticalalignment="top")
    plt.legend()
    plt.show()
except ImportError:
    print("Matplotlib not found. Skipping plot.")
    print("Test loss (MSE):", np.mean((y_pred - y_test) ** 2))
