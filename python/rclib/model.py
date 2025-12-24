"""Model module for Reservoir Computing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import (
    _rclib,  # Import the C++ bindings
    readouts,
    reservoirs,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class ESN:
    """Echo State Network (ESN) model."""

    def __init__(self, connection_type: str = "serial") -> None:
        """Initialize the ESN model.

        Parameters
        ----------
        connection_type : str, optional
            The type of connection between reservoirs ("serial" or "parallel").
            Default is "serial".
        """
        self.connection_type = connection_type
        self._reservoirs_params: list[Any] = []  # Store parameters for Python-side reservoir objects
        self._readout_params: Any = None  # Store parameters for Python-side readout object
        self._cpp_model = _rclib.Model()  # Initialize the C++ Model object

    def add_reservoir(self, reservoir: Any) -> None:  # noqa: ANN401
        """Add a reservoir to the model.

        Parameters
        ----------
        reservoir : Any
            The reservoir object to add.

        Raises
        ------
        TypeError
            If the reservoir type is unsupported.
        """
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
                reservoir.seed,
            )
            self._cpp_model.addReservoir(cpp_res, self.connection_type)
        # Add other reservoir types here as they are implemented
        else:
            msg = "Unsupported reservoir type"
            raise TypeError(msg)

    def set_readout(self, readout: Any) -> None:  # noqa: ANN401
        """Set the readout for the model.

        Parameters
        ----------
        readout : Any
            The readout object to set.

        Raises
        ------
        TypeError
            If the readout type is unsupported.
        """
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
            msg = "Unsupported readout type"
            raise TypeError(msg)

    def fit(self, x: ArrayLike, y: ArrayLike, washout_len: int = 0) -> None:
        """Fit the model to the data.

        Parameters
        ----------
        x : ArrayLike
            Input data.
        y : ArrayLike
            Target data.
        washout_len : int, optional
            Number of initial samples to discard. Default is 0.
        """
        # Call the C++ model's fit method
        self._cpp_model.fit(x, y, washout_len)

    def predict(self, x: ArrayLike, *, reset_state_before_predict: bool = True) -> np.ndarray:
        """Predict using the trained model.

        Parameters
        ----------
        x : ArrayLike
            Input data.
        reset_state_before_predict : bool, optional
            Whether to reset the reservoir state before prediction. Default is True.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        # Call the C++ model's predict method
        return self._cpp_model.predict(x, reset_state_before_predict)

    def predict_online(self, x: ArrayLike) -> np.ndarray:
        """Predict in online mode (updating state).

        Parameters
        ----------
        x : ArrayLike
            Input data.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        # Call the C++ model's predictOnline method
        return self._cpp_model.predictOnline(x)

    def predict_generative(self, prime_data: ArrayLike, n_steps: int) -> np.ndarray:
        """Generative prediction.

        Parameters
        ----------
        prime_data : ArrayLike
            Initial data to prime the reservoir.
        n_steps : int
            Number of steps to generate.

        Returns
        -------
        np.ndarray
            The generated data.
        """
        # Call the C++ model's predictGenerative method
        return self._cpp_model.predictGenerative(prime_data, n_steps)

    def get_reservoir(self, index: int) -> Any:  # noqa: ANN401
        """Get the reservoir object at the specified index.

        Parameters
        ----------
        index : int
            The index of the reservoir.

        Returns
        -------
        Any
            The C++ reservoir object.
        """
        # Return the C++ reservoir object
        return self._cpp_model.getReservoir(index)

    def reset_reservoirs(self) -> None:
        """Reset the states of all reservoirs."""
        # Call the C++ model's resetReservoirs method
        self._cpp_model.resetReservoirs()

    def partial_fit(self, x: ArrayLike, y: ArrayLike) -> None:
        """Update the model with a single sample (online learning).

        Parameters
        ----------
        x : ArrayLike
            Input data sample.
        y : ArrayLike
            Target data sample.

        Raises
        ------
        RuntimeError
            If no reservoir or readout is set.
        """
        # Assuming only one reservoir for simplicity in online learning for now.
        # If multiple reservoirs are present, the logic would need to be more complex
        # to handle how their states are combined before feeding to the readout.
        if not self._reservoirs_params:
            msg = "No reservoir added to the model."
            raise RuntimeError(msg)
        if not self._readout_params:
            msg = "No readout set for the model."
            raise RuntimeError(msg)

        # Get the C++ reservoir object (assuming the first one for now)
        cpp_res = self._cpp_model.getReservoir(0)

        # Advance reservoir state
        cpp_res.advance(x)
        current_state = cpp_res.getState()

        # Get the C++ readout object
        cpp_readout = self._cpp_model.getReadout()

        # Perform partial fit (online update)
        cpp_readout.partialFit(current_state, y)
