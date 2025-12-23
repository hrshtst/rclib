from __future__ import annotations

from . import (
    _rclib,  # Import the C++ bindings
    readouts,
    reservoirs,
)


class ESN:
    def __init__(self, connection_type="serial"):
        self.connection_type = connection_type
        self._reservoirs_params = []  # Store parameters for Python-side reservoir objects
        self._readout_params = None  # Store parameters for Python-side readout object
        self._cpp_model = _rclib.Model()  # Initialize the C++ Model object

    def add_reservoir(self, reservoir):
        # Store the Python reservoir object's parameters
        self._reservoirs_params.append(reservoir)
        # Create and add the C++ reservoir to the C++ model
        if isinstance(reservoir, reservoirs.RandomSparse):
            cpp_res = _rclib.RandomSparseReservoir(
                reservoir.n_neurons,
                reservoir.spectral_radius,
                reservoir.sparsity,
                reservoir.leak_rate,
                reservoir.input_scaling,
                reservoir.include_bias,
            )
            self._cpp_model.addReservoir(cpp_res, self.connection_type)
        # Add other reservoir types here as they are implemented
        else:
            raise ValueError("Unsupported reservoir type")

    def set_readout(self, readout):
        # Store the Python readout object's parameters
        self._readout_params = readout
        # Create and set the C++ readout to the C++ model
        if isinstance(readout, readouts.Ridge):
            cpp_readout = _rclib.RidgeReadout(readout.alpha, readout.include_bias)
            self._cpp_model.setReadout(cpp_readout)
        elif isinstance(readout, readouts.Rls):
            cpp_readout = _rclib.RlsReadout(readout.lambda_, readout.delta, readout.include_bias)
            self._cpp_model.setReadout(cpp_readout)
        elif isinstance(readout, readouts.Lms):
            cpp_readout = _rclib.LmsReadout(readout.learning_rate, readout.include_bias)
            self._cpp_model.setReadout(cpp_readout)
        else:
            raise ValueError("Unsupported readout type")

    def fit(self, X, y, washout_len=0):
        # Call the C++ model's fit method
        self._cpp_model.fit(X, y, washout_len)

    def predict(self, X, reset_state_before_predict=True):
        # Call the C++ model's predict method
        return self._cpp_model.predict(X, reset_state_before_predict)

    def predict_online(self, X):
        # Call the C++ model's predictOnline method
        return self._cpp_model.predictOnline(X)

    def predict_generative(self, prime_data, n_steps):
        # Call the C++ model's predictGenerative method
        return self._cpp_model.predictGenerative(prime_data, n_steps)

    def get_reservoir(self, index):
        # Return the C++ reservoir object
        return self._cpp_model.getReservoir(index)

    def reset_reservoirs(self):
        # Call the C++ model's resetReservoirs method
        self._cpp_model.resetReservoirs()

    def partial_fit(self, X, y):
        # Assuming only one reservoir for simplicity in online learning for now.
        # If multiple reservoirs are present, the logic would need to be more complex
        # to handle how their states are combined before feeding to the readout.
        if not self._reservoirs_params:
            raise RuntimeError("No reservoir added to the model.")
        if not self._readout_params:
            raise RuntimeError("No readout set for the model.")

        # Get the C++ reservoir object (assuming the first one for now)
        cpp_res = self._cpp_model.getReservoir(0)

        # Advance reservoir state
        cpp_res.advance(X)
        current_state = cpp_res.getState()

        # Get the C++ readout object
        cpp_readout = self._cpp_model.getReadout()

        # Perform partial fit (online update)
        cpp_readout.partialFit(current_state, y)
