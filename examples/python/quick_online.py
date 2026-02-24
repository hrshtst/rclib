"""Quick online learning example."""

from __future__ import annotations

import numpy as np
from rclib import ESN, readouts, reservoirs


def main() -> None:
    """Run the quick online learning example."""
    # 1. Data preparation (sine wave prediction)
    t = np.linspace(0, 100, 1000)
    data = np.sin(t).reshape(-1, 1)
    x, y = data[:-1], data[1:]

    # 2. Initialization of ESN and RLS readout
    model = ESN()
    model.add_reservoir(reservoirs.RandomSparse(n_neurons=500, spectral_radius=0.9))
    model.set_readout(readouts.Rls(lambda_=0.99, delta=1.0, include_bias=True))

    # 3. Online learning (update weights sample by sample)
    for i in range(len(x)):
        model.partial_fit(x[i : i + 1], y[i : i + 1])

    # 4. Prediction and evaluation
    y_pred = model.predict(x)
    mse = np.mean((y - y_pred) ** 2)
    print(f"MSE: {mse:.2e}")


if __name__ == "__main__":
    main()
