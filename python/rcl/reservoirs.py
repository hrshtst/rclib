class RandomSparse:
    def __init__(self, n_neurons, spectral_radius, sparsity, leak_rate, include_bias):
        self.n_neurons = n_neurons
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.include_bias = include_bias

class Nvar:
    def __init__(self, num_lags):
        self.num_lags = num_lags