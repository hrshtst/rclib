# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Model module for Reservoir Computing."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from . import (
    _rclib,  # Import the C++ bindings
    readouts,
    reservoirs,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# Solver names of the Python configs mapped to the C++ enums, and back.
_RIDGE_SOLVERS = {
    "auto": _rclib.RidgeReadout.Solver.AUTO,
    "cholesky": _rclib.RidgeReadout.Solver.CHOLESKY,
    "dual_cholesky": _rclib.RidgeReadout.Solver.DUAL_CHOLESKY,
    "conjugate_gradient": _rclib.RidgeReadout.Solver.CONJUGATE_GRADIENT,
    "conjugate_gradient_implicit": _rclib.RidgeReadout.Solver.CONJUGATE_GRADIENT_IMPLICIT,
}
_RLS_SOLVERS = {
    "rank1_update": _rclib.RlsReadout.Solver.RANK1_UPDATE,
    "rank_k_update": _rclib.RlsReadout.Solver.RANK_K_UPDATE,
}
_RIDGE_SOLVER_NAMES = {solver: name for name, solver in _RIDGE_SOLVERS.items()}
_RLS_SOLVER_NAMES = {solver: name for name, solver in _RLS_SOLVERS.items()}


def _reservoir_config_from_cpp(cpp_reservoir: Any) -> reservoirs.RandomSparse | reservoirs.Nvar:  # noqa: ANN401
    """Rebuild the Python configuration of a loaded C++ reservoir."""
    if isinstance(cpp_reservoir, _rclib.RandomSparseReservoir):
        return reservoirs.RandomSparse(
            n_neurons=cpp_reservoir.getNNeurons(),
            spectral_radius=cpp_reservoir.getSpectralRadius(),
            sparsity=cpp_reservoir.getSparsity(),
            leak_rate=cpp_reservoir.getLeakRate(),
            input_scaling=cpp_reservoir.getInputScaling(),
            include_bias=cpp_reservoir.getIncludeBias(),
            seed=cpp_reservoir.getSeed(),
        )
    if isinstance(cpp_reservoir, _rclib.NvarReservoir):
        return reservoirs.Nvar(num_lags=cpp_reservoir.getNumLags(), polynomial_order=cpp_reservoir.getPolynomialOrder())
    msg = f"Unsupported reservoir type: {type(cpp_reservoir).__name__}"
    raise TypeError(msg)


def _readout_config_from_cpp(cpp_readout: Any) -> readouts.Ridge | readouts.Rls | readouts.Lms:  # noqa: ANN401
    """Rebuild the Python configuration of a loaded C++ readout."""
    if isinstance(cpp_readout, _rclib.RidgeReadout):
        return readouts.Ridge(
            alpha=cpp_readout.getAlpha(),
            include_bias=cpp_readout.getIncludeBias(),
            solver=_RIDGE_SOLVER_NAMES[cpp_readout.getSolver()],
            tolerance=cpp_readout.getTolerance(),
        )
    if isinstance(cpp_readout, _rclib.RlsReadout):
        return readouts.Rls(
            lambda_=cpp_readout.getLambda(),
            delta=cpp_readout.getDelta(),
            include_bias=cpp_readout.getIncludeBias(),
            solver=_RLS_SOLVER_NAMES[cpp_readout.getSolver()],
        )
    if isinstance(cpp_readout, _rclib.LmsReadout):
        return readouts.Lms(learning_rate=cpp_readout.getLearningRate(), include_bias=cpp_readout.getIncludeBias())
    msg = f"Unsupported readout type: {type(cpp_readout).__name__}"
    raise TypeError(msg)


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
        if connection_type not in {"serial", "parallel"}:
            msg = "connection_type must be 'serial' or 'parallel'."
            raise ValueError(msg)

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
        elif isinstance(reservoir, reservoirs.Nvar):
            cpp_res = _rclib.NvarReservoir(reservoir.num_lags, reservoir.polynomial_order)
            self._cpp_model.addReservoir(cpp_res, self.connection_type)
        # Add other reservoir types here as they are implemented
        else:
            msg = "Unsupported reservoir type"
            raise TypeError(msg)

        # Update readout in case it's using "auto" solver
        self._update_readout()

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
        self._update_readout()

    def _update_readout(self) -> None:
        """Instantiate or update the C++ readout based on current parameters."""
        if self._readout_params is None:
            return

        # Don't re-instantiate if already exists (to preserve online learning state)
        try:
            if self._cpp_model.getReadout() is not None:
                return
        except RuntimeError:
            # getReadout() throws if not set
            pass

        readout = self._readout_params

        # Create and set the C++ readout to the C++ model
        if isinstance(readout, readouts.Ridge):
            if readout.solver not in _RIDGE_SOLVERS:
                msg = f"Unsupported solver: {readout.solver}"
                raise ValueError(msg)

            cpp_readout = _rclib.RidgeReadout(
                readout.alpha, readout.include_bias, _RIDGE_SOLVERS[readout.solver], readout.tolerance
            )
            self._cpp_model.setReadout(cpp_readout)
        elif isinstance(readout, readouts.Rls):
            if readout.solver not in _RLS_SOLVERS:
                msg = f"Unsupported RLS solver: {readout.solver}"
                raise ValueError(msg)

            cpp_readout = _rclib.RlsReadout(
                readout.lambda_, readout.delta, readout.include_bias, _RLS_SOLVERS[readout.solver]
            )
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
        # Ensure readout is correctly initialized (especially for "auto" solver)
        self._update_readout()
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

    def partial_fit(self, x: ArrayLike | None, y: ArrayLike) -> None:
        """Update the model with a single sample (online learning).

        Parameters
        ----------
        x : ArrayLike, optional
            Input data sample. If None, the reservoir state is not advanced
            (useful if predict_online was already called).
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

        # Ensure readout is correctly initialized
        self._update_readout()

        if x is None:
            cpp_readout = self._cpp_model.getReadout()
            states = [self._cpp_model.getReservoir(i).getState() for i in range(len(self._reservoirs_params))]
            if any(state.shape[1] == 0 for state in states):
                msg = "Reservoir state is uninitialized; call predict_online or partial_fit with input first."
                raise RuntimeError(msg)
            current_state = states[-1] if self.connection_type == "serial" else np.hstack(states)
            cpp_readout.partialFit(current_state, y)
        else:
            self._cpp_model.partialFit(x, y)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the model to a file.

        The file holds the configuration, the trained weights and the current
        reservoir states, so a loaded model continues exactly where this one
        stopped. It can also be loaded from C++ with ``Model::load``. An existing
        file at ``path`` is replaced in one step and is left unchanged if saving
        fails.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file.

        Raises
        ------
        SerializationError
            If the model lacks a reservoir or readout, is internally inconsistent,
            or the file cannot be written.
        """
        self._cpp_model.save(os.fspath(path))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ESN:
        """Load a model saved by :meth:`save` or by C++ ``Model::save``.

        Only load files from sources you trust: the format contains no executable
        code, but it is parsed by native code.

        Parameters
        ----------
        path : str or os.PathLike
            Model file to read.

        Returns
        -------
        ESN
            The restored model, including its reservoir states.

        Raises
        ------
        SerializationError
            If the file cannot be read or is not a valid model file.
        """
        return cls._from_cpp_model(_rclib.Model.load(os.fspath(path)))

    @classmethod
    def _from_cpp_model(cls, cpp_model: Any) -> ESN:  # noqa: ANN401
        """Wrap a loaded C++ model, rebuilding the Python configuration objects."""
        esn = cls(cpp_model.getConnectionType())
        esn._cpp_model = cpp_model
        # partial_fit and _update_readout read these configuration objects.
        try:
            esn._reservoirs_params = [
                _reservoir_config_from_cpp(cpp_model.getReservoir(i)) for i in range(cpp_model.getNumReservoirs())
            ]
            esn._readout_params = _readout_config_from_cpp(cpp_model.getReadout())
        except (TypeError, ValueError) as err:
            msg = f"Invalid model file: {err}"
            raise _rclib.SerializationError(msg) from err
        return esn

    def __getstate__(self) -> dict[str, bytes]:
        """Support pickle and copy.deepcopy by storing the model in the model file format.

        Unpickling can run arbitrary code, so only unpickle data you trust.
        """
        return {"model": self._cpp_model.dumps()}

    def __setstate__(self, state: dict[str, bytes]) -> None:
        """Restore a model pickled by __getstate__."""
        self.__dict__.update(ESN._from_cpp_model(_rclib.Model.loads(state["model"])).__dict__)
