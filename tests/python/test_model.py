import numpy as np
from rcl import reservoirs, readouts, ESN

def test_model_creation():
    model = ESN()
    assert model is not None

def test_model_fit_predict():
    model = ESN()
    res = reservoirs.RandomSparse(n_neurons=100, spectral_radius=0.9, sparsity=0.1, leak_rate=0.2)
    readout = readouts.Ridge(alpha=1e-6)
    model.add_reservoir(res)
    model.set_readout(readout)

    X_train = np.random.rand(200, 1)
    y_train = np.random.rand(200, 1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    assert y_pred.shape == (200, 1)
    assert np.mean((y_pred - y_train)**2) < np.mean(y_train**2)

def test_parallel_model_fit_predict():
    model = ESN(connection_type='parallel')
    res1 = reservoirs.RandomSparse(n_neurons=50, spectral_radius=0.9, sparsity=0.1, leak_rate=0.2)
    res2 = reservoirs.RandomSparse(n_neurons=50, spectral_radius=0.9, sparsity=0.1, leak_rate=0.2)
    readout = readouts.Ridge(alpha=1e-6)
    model.add_reservoir(res1)
    model.add_reservoir(res2)
    model.set_readout(readout)

    X_train = np.random.rand(200, 1)
    y_train = np.random.rand(200, 1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    assert y_pred.shape == (200, 1)
    assert np.mean((y_pred - y_train)**2) < np.mean(y_train**2)
