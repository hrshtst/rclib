import numpy as np

from rcl import ESN, readouts, reservoirs


def mackey_glass(n_samples=1500, tau=17, seed=0):
    # Mackey-Glass time series generation
    np.random.seed(seed)
    x = np.zeros(n_samples + tau)
    x[0:tau] = 0.5 + 0.5 * np.random.rand(tau)
    for t in range(tau, n_samples + tau - 1):
        x[t+1] = x[t] + (0.2 * x[t-tau]) / (1 + x[t-tau]**10) - 0.1 * x[t]
    return x[tau:]

# 1. Generate Mackey-Glass data
data = mackey_glass()
X = data[:-1].reshape(-1, 1)
y = data[1:].reshape(-1, 1)

# Split into training and testing sets
train_len = 1000
washout_len = 100
X_train, y_train = X[:train_len], y[:train_len]
X_test, y_test = X[train_len:], y[train_len:]

# 2. Configure Reservoir
res = reservoirs.RandomSparse(
    n_neurons=2000,
    spectral_radius=1.1,
    sparsity=0.05,
    leak_rate=0.1,
    include_bias=True
)

# 3. Configure Readout
readout = readouts.Ridge(alpha=1e-8, include_bias=True)

# 4. Configure Model
model = ESN()
model.add_reservoir(res)
model.set_readout(readout)

# 5. Fit and Predict
model.fit(X_train, y_train, washout_len)
y_pred = model.predict(X_test)

# 6. Plot the results
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(y_test)), y_test, label="True")
    plt.plot(range(len(y_pred)), y_pred, label="Predicted")
    mse = np.mean((y_pred - y_test)**2)
    plt.text(0.05, 0.95, f'MSE: {mse:.4e}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.legend()
    plt.show()
except ImportError:
    print("Matplotlib not found. Skipping plot.")
    print("Test loss (MSE):", np.mean((y_pred - y_test)**2))
