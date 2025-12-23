from __future__ import annotations


class RandomSparse:
    def __init__(
        self,
        n_neurons,
        spectral_radius,
        sparsity=0.1,
        leak_rate=1.0,
        input_scaling=1.0,
        include_bias=False,
    ):
        self.n_neurons = n_neurons
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.include_bias = include_bias


class Nvar:
    def __init__(self, num_lags):
        self.num_lags = num_lags
