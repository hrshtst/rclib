# Copyright (c) 2025-2026 Hiroshi Atsuta
# SPDX-License-Identifier: Apache-2.0

"""Model module for Reservoir Computing."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from . import (
    _rclib,  # Import the C++ bindings
    learning,
    readouts,
    reservoirs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

# Solver and method names of the Python configs mapped to the C++ enums, and back.
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
_SPECTRAL_RADIUS_METHODS = {
    "power_iteration": _rclib.RandomSparseReservoir.SpectralRadiusMethod.POWER_ITERATION,
    "dense": _rclib.RandomSparseReservoir.SpectralRadiusMethod.DENSE,
}
_RULE = _rclib.RewardModulatedHebbianRule
_DISTRIBUTIONS = {"uniform": _RULE.Distribution.UNIFORM, "gaussian": _RULE.Distribution.GAUSSIAN}
_POST_SIGNALS = {
    "noise": _RULE.PostSignal.NOISE,
    "high_pass": _RULE.PostSignal.HIGH_PASS,
    "potential": _RULE.PostSignal.POTENTIAL,
}
_UPDATES = {"online": _RULE.Update.ONLINE, "trial": _RULE.Update.TRIAL}
_MODULATORS = {"difference": _RULE.Modulator.DIFFERENCE, "heaviside": _RULE.Modulator.HEAVISIDE}
_BASELINES = {"ema": _RULE.Baseline.EMA, "window": _RULE.Baseline.WINDOW, "none": _RULE.Baseline.NONE}
_RIDGE_SOLVER_NAMES = {solver: name for name, solver in _RIDGE_SOLVERS.items()}
_RLS_SOLVER_NAMES = {solver: name for name, solver in _RLS_SOLVERS.items()}
_SPECTRAL_RADIUS_METHOD_NAMES = {method: name for name, method in _SPECTRAL_RADIUS_METHODS.items()}
_DISTRIBUTION_NAMES = {value: name for name, value in _DISTRIBUTIONS.items()}
_POST_SIGNAL_NAMES = {value: name for name, value in _POST_SIGNALS.items()}
_UPDATE_NAMES = {value: name for name, value in _UPDATES.items()}
_MODULATOR_NAMES = {value: name for name, value in _MODULATORS.items()}
_BASELINE_NAMES = {value: name for name, value in _BASELINES.items()}


def _rule_to_cpp(rule: learning.RewardModulatedHebbian) -> Any:  # noqa: ANN401
    """Build the C++ rule described by a Python rule configuration."""
    cpp_rule = _rclib.RewardModulatedHebbianRule()
    cpp_rule.learning_rate = rule.learning_rate
    cpp_rule.noise_distribution = _DISTRIBUTIONS[rule.exploration.distribution]
    cpp_rule.noise_amplitude = rule.exploration.amplitude
    cpp_rule.noise_probability = rule.exploration.probability
    cpp_rule.noise_seed = rule.exploration.seed
    cpp_rule.post = _POST_SIGNALS[rule.post]
    cpp_rule.post_decay = rule.post_decay
    cpp_rule.eligibility_power = rule.eligibility_power
    cpp_rule.update = _UPDATES[rule.update]
    cpp_rule.trace_decay = rule.trace_decay
    cpp_rule.modulator = _MODULATORS[rule.modulator]
    cpp_rule.baseline = _BASELINES[rule.baseline.kind]
    cpp_rule.baseline_decay = rule.baseline.decay
    cpp_rule.baseline_window = rule.baseline.window
    cpp_rule.learning_rate_decay_steps = rule.learning_rate_decay_steps
    cpp_rule.max_update = rule.max_update
    cpp_rule.decorrelation = rule.decorrelation
    return cpp_rule


def _rule_from_cpp(cpp_rule: Any) -> learning.RewardModulatedHebbian:  # noqa: ANN401
    """Rebuild the Python configuration of a C++ rule."""
    return learning.RewardModulatedHebbian(
        learning_rate=cpp_rule.learning_rate,
        exploration=learning.Exploration(
            distribution=_DISTRIBUTION_NAMES[cpp_rule.noise_distribution],
            amplitude=cpp_rule.noise_amplitude,
            probability=cpp_rule.noise_probability,
            seed=cpp_rule.noise_seed,
        ),
        post=_POST_SIGNAL_NAMES[cpp_rule.post],
        post_decay=cpp_rule.post_decay,
        eligibility_power=cpp_rule.eligibility_power,
        update=_UPDATE_NAMES[cpp_rule.update],
        trace_decay=cpp_rule.trace_decay,
        modulator=_MODULATOR_NAMES[cpp_rule.modulator],
        baseline=learning.Baseline(
            kind=_BASELINE_NAMES[cpp_rule.baseline],
            decay=cpp_rule.baseline_decay,
            window=cpp_rule.baseline_window,
        ),
        learning_rate_decay_steps=cpp_rule.learning_rate_decay_steps,
        max_update=cpp_rule.max_update,
        decorrelation=cpp_rule.decorrelation,
    )


def _reservoir_config_from_cpp(
    cpp_reservoir: Any,  # noqa: ANN401
) -> reservoirs.RandomSparse | reservoirs.Nvar | reservoirs.RewardModulated:
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
            spectral_radius_method=_SPECTRAL_RADIUS_METHOD_NAMES[cpp_reservoir.getSpectralRadiusMethod()],
            bias_scaling=cpp_reservoir.getBiasScaling(),
        )
    if isinstance(cpp_reservoir, _rclib.NvarReservoir):
        return reservoirs.Nvar(num_lags=cpp_reservoir.getNumLags(), polynomial_order=cpp_reservoir.getPolynomialOrder())
    if isinstance(cpp_reservoir, _rclib.RewardModulatedReservoir):
        return reservoirs.RewardModulated(
            n_neurons=cpp_reservoir.getNNeurons(),
            spectral_radius=cpp_reservoir.getSpectralRadius(),
            sparsity=cpp_reservoir.getSparsity(),
            leak_rate=cpp_reservoir.getLeakRate(),
            input_scaling=cpp_reservoir.getInputScaling(),
            rule=_rule_from_cpp(cpp_reservoir.getRule()),
            include_bias=cpp_reservoir.getIncludeBias(),
            seed=cpp_reservoir.getSeed(),
            spectral_radius_method=_SPECTRAL_RADIUS_METHOD_NAMES[cpp_reservoir.getSpectralRadiusMethod()],
            bias_scaling=cpp_reservoir.getBiasScaling(),
            plastic_presynaptic=cpp_reservoir.getPlasticPresynaptic(),
            plastic_postsynaptic=cpp_reservoir.getPlasticPostsynaptic(),
        )
    msg = f"Unsupported reservoir type: {type(cpp_reservoir).__name__}"
    raise TypeError(msg)


def _readout_config_from_cpp(
    cpp_readout: Any,  # noqa: ANN401
) -> readouts.Ridge | readouts.Rls | readouts.Lms | readouts.RewardModulated:
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
    if isinstance(cpp_readout, _rclib.RewardModulatedReadout):
        return readouts.RewardModulated(
            n_outputs=cpp_readout.getNOutputs(),
            rule=_rule_from_cpp(cpp_readout.getRule()),
            include_bias=cpp_readout.getIncludeBias(),
        )
    msg = f"Unsupported readout type: {type(cpp_readout).__name__}"
    raise TypeError(msg)


def _cpp_readout_from_config(readout: Any) -> Any:  # noqa: ANN401
    """Build the C++ readout described by a Python readout configuration."""
    if isinstance(readout, readouts.Ridge):
        if readout.solver not in _RIDGE_SOLVERS:
            msg = f"Unsupported solver: {readout.solver}"
            raise ValueError(msg)
        return _rclib.RidgeReadout(
            readout.alpha, readout.include_bias, _RIDGE_SOLVERS[readout.solver], readout.tolerance
        )
    if isinstance(readout, readouts.Rls):
        if readout.solver not in _RLS_SOLVERS:
            msg = f"Unsupported RLS solver: {readout.solver}"
            raise ValueError(msg)
        return _rclib.RlsReadout(readout.lambda_, readout.delta, readout.include_bias, _RLS_SOLVERS[readout.solver])
    if isinstance(readout, readouts.Lms):
        return _rclib.LmsReadout(readout.learning_rate, readout.include_bias)
    if isinstance(readout, readouts.RewardModulated):
        return _rclib.RewardModulatedReadout(readout.n_outputs, _rule_to_cpp(readout.rule), readout.include_bias)
    msg = "Unsupported readout type"
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
        ValueError
            If a RandomSparse reservoir has an unsupported spectral_radius_method.
        """
        # Create and add the C++ reservoir to the C++ model
        if isinstance(reservoir, reservoirs.RandomSparse):
            if reservoir.spectral_radius_method not in _SPECTRAL_RADIUS_METHODS:
                msg = f"Unsupported spectral_radius_method: {reservoir.spectral_radius_method}"
                raise ValueError(msg)
            cpp_res = _rclib.RandomSparseReservoir(
                reservoir.n_neurons,
                reservoir.spectral_radius,
                reservoir.sparsity,
                reservoir.leak_rate,
                reservoir.input_scaling,
                reservoir.include_bias,
                reservoir.seed,
                _SPECTRAL_RADIUS_METHODS[reservoir.spectral_radius_method],
                reservoir.bias_scaling,
            )
            self._cpp_model.addReservoir(cpp_res, self.connection_type)
        elif isinstance(reservoir, reservoirs.Nvar):
            cpp_res = _rclib.NvarReservoir(reservoir.num_lags, reservoir.polynomial_order)
            self._cpp_model.addReservoir(cpp_res, self.connection_type)
        elif isinstance(reservoir, reservoirs.RewardModulated):
            cpp_res = _rclib.RewardModulatedReservoir(
                reservoir.n_neurons,
                reservoir.spectral_radius,
                reservoir.sparsity,
                reservoir.leak_rate,
                reservoir.input_scaling,
                reservoir.include_bias,
                reservoir.seed,
                _SPECTRAL_RADIUS_METHODS[reservoir.spectral_radius_method],
                reservoir.bias_scaling,
                _rule_to_cpp(reservoir.rule),
                reservoir.plastic_presynaptic,
                reservoir.plastic_postsynaptic,
            )
            self._cpp_model.addReservoir(cpp_res, self.connection_type)
        # Add other reservoir types here as they are implemented
        else:
            msg = "Unsupported reservoir type"
            raise TypeError(msg)

        # Store the Python reservoir object's parameters only once the C++ model holds the
        # reservoir, so a rejected reservoir leaves both sides unchanged.
        self._reservoirs_params.append(reservoir)

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
        # Build the C++ readout first and record the configuration only once the C++ model
        # holds it, so a rejected readout leaves both sides unchanged and a second call
        # replaces the readout set before.
        self._cpp_model.setReadout(_cpp_readout_from_config(readout))
        self._readout_params = readout

    def _update_readout(self) -> None:
        """Create the C++ readout from the stored configuration if the C++ model has none."""
        if self._readout_params is None:
            return

        # An existing readout is kept, so its fitted or online learning state survives.
        try:
            if self._cpp_model.getReadout() is not None:
                return
        except RuntimeError:
            # getReadout() throws if not set
            pass

        self._cpp_model.setReadout(_cpp_readout_from_config(self._readout_params))

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

    def fit_sequences(self, inputs: Sequence[ArrayLike], targets: Sequence[ArrayLike], washout_len: int = 0) -> None:
        """Fit the readout once on several independent sequences, such as episodes.

        Each sequence starts from reset reservoirs and discards its own first
        ``washout_len`` samples. The remaining reservoir states and targets of all
        sequences are stacked and the readout is fitted once, so nothing carries over
        from one sequence to the next. With a single sequence this equals :meth:`fit`.

        Parameters
        ----------
        inputs : Sequence[ArrayLike]
            Input data, one 2-D array of shape (n_samples_i, n_inputs) per sequence,
            for example a list of arrays. Sequences may differ in length but not in
            width.
        targets : Sequence[ArrayLike]
            Target data, one 2-D array of shape (n_samples_i, n_outputs) per sequence,
            with as many rows as the matching input sequence.
        washout_len : int, optional
            Number of initial samples to discard from each sequence. Default is 0.

        Raises
        ------
        ValueError
            If a single 2-D array is passed instead of a sequence of arrays, if there
            are no sequences or unequal numbers of input and target sequences, or if a
            sequence is empty, its inputs and targets differ in length, its width
            differs from the first sequence's, or its input width differs from the one
            the reservoirs are locked to by earlier input. Messages name the sequence
            index.
        IndexError
            If ``washout_len`` is negative or not shorter than a sequence.
        RuntimeError
            If the model has no reservoir or no readout.

        Notes
        -----
        These checks run before the model changes. Afterwards the reservoirs hold
        their states at the end of the last sequence.
        """
        # A single 2-D array would otherwise be split into one-row sequences.
        for name, sequences in (("inputs", inputs), ("targets", targets)):
            if isinstance(sequences, np.ndarray) and sequences.ndim == 2:  # noqa: PLR2004
                msg = f"{name} must be a sequence of 2-D arrays, one per sequence, not a single 2-D array."
                raise ValueError(msg)
        self._update_readout()
        self._cpp_model.fitSequences(inputs, targets, washout_len)

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

        Advances the reservoirs through ``prime_data`` (if any), then generates
        ``n_steps`` outputs, feeding each output back as the next input. Every
        generated output, including the last, advances the reservoirs, so a following
        call with empty ``prime_data`` continues the sequence: generating ``a`` steps
        and then ``b`` steps equals generating ``a + b`` steps. With ``n_steps == 0``
        no output is fed back.

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

        Raises
        ------
        ValueError
            If a generated output cannot be fed back because its width differs from
            the input width the reservoirs are locked to (also for ``n_steps == 1``).
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

    def get_readout(self) -> Any:  # noqa: ANN401
        """Get the readout object, such as a fitted ridge readout to read its weights from.

        Returns
        -------
        Any
            The C++ readout object.

        Raises
        ------
        RuntimeError
            If no readout is set.
        """
        return self._cpp_model.getReadout()

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

    def act(self, x: ArrayLike, *, explore: bool = True) -> np.ndarray:
        """Step the model through ``x`` row by row to learn from rewards.

        With ``explore``, the reward-modulated components add exploration noise and record each
        row's eligibility for the next :meth:`reinforce`; without it every step is noise-free and
        no learning state changes, which equals :meth:`predict_online`. A model without a readout
        returns the reservoir states, whose designated neurons can serve as outputs.

        Parameters
        ----------
        x : ArrayLike
            Input rows, one per time step.
        explore : bool, optional
            Whether to explore and record eligibility. Default is True.

        Returns
        -------
        np.ndarray
            The output of each row: the readout's, or the reservoir states without a readout.

        Raises
        ------
        ValueError
            If the inputs are not finite or do not fit the widths the model is locked to.
        RuntimeError
            If a reservoir object is listed twice, or a row's output or eligibility is not
            finite. A failing row changes no learning state and keeps the rows before it.
        """
        self._update_readout()
        return self._cpp_model.act(x, explore)

    def reinforce(self, reward: float, *, key: int = 0, baseline: float | None = None) -> None:
        """Update the reward-modulated components for a reward.

        Each component compares the reward with its baseline for ``key`` (see
        ``learning.Baseline``), or with ``baseline`` when given, and changes its weights in
        proportion to the eligibility recorded by :meth:`act`. The update applies to every
        component or to none.

        Parameters
        ----------
        reward : float
            The reward.
        key : int, optional
            The baseline to use and update, such as a trial type. Default is 0.
        baseline : float, optional
            A baseline to use instead of the internal one, which is then left unchanged.

        Raises
        ------
        ValueError
            If the reward or baseline is not finite.
        RuntimeError
            If the model has no reward-modulated component, a component has not explored since
            its last update, or the update would not be finite.
        """
        self._cpp_model.reinforce(reward, key, baseline)

    def modulate(self, modulator: float) -> None:
        """Update the reward-modulated components with the modulator given directly.

        This applies ``dW = eta * modulator * eligibility`` without a baseline, so any critic,
        such as a temporal-difference error, can drive learning.

        Parameters
        ----------
        modulator : float
            The modulator ``m``.
        """
        self._cpp_model.modulate(modulator)

    def clear_eligibility(self) -> None:
        """Zero the eligibility of the reward-modulated components, for example after an aborted trial."""
        self._cpp_model.clearEligibility()

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

    def __getstate__(self) -> dict[str, Any]:
        """Support pickle and copy.deepcopy.

        Every instance attribute is kept, including ones added by users or
        subclasses; the C++ model is stored as bytes in the model file format.
        Unpickling can run arbitrary code, so only unpickle data you trust.
        """
        state = self.__dict__.copy()
        state["_cpp_model"] = self._cpp_model.dumps()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a model pickled by __getstate__."""
        cpp_model = _rclib.Model.loads(state["_cpp_model"])  # before touching self, in case it fails
        self.__dict__.update(state)
        self._cpp_model = cpp_model
