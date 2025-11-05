from . import _rcl


class ESN:
    def __init__(self, connection_type='serial'):
        self.model = _rcl.Model()
        self.connection_type = connection_type

    def add_reservoir(self, reservoir):
        self.model.addReservoir(reservoir, self.connection_type)

    def set_readout(self, readout):
        self.model.setReadout(readout)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
