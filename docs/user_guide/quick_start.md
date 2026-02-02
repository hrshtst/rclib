# Quick Start

Here is a minimal example to train an ESN on a simple sine wave task.

```python
import numpy as np
from rclib import ESN, readouts, reservoirs

# 1. Prepare data
x = np.linspace(0, 10, 1000).reshape(-1, 1)
y = np.sin(x)

# Split into train/test
train_len = 800
x_train, y_train = x[:train_len], y[:train_len]
x_test, y_test = x[train_len:], y[train_len:]

# 2. Configure Reservoir
res = reservoirs.RandomSparse(
    n_neurons=500,
    spectral_radius=0.9,
    sparsity=0.1,
    leak_rate=0.5,
    seed=42
)

# 3. Configure Readout (Ridge Regression)
readout = readouts.Ridge(alpha=1e-6, include_bias=True)

# 4. Build Model
model = ESN()
model.add_reservoir(res)
model.set_readout(readout)

# 5. Train
model.fit(x_train, y_train, washout_len=50)

# 6. Predict
y_pred = model.predict(x_test)

# Calculate error
mse = np.mean((y_pred - y_test) ** 2)
print(f"Test MSE: {mse:.4e}")
```
