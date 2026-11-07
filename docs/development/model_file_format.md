# Model File Format

`Model::save` / `ESN.save` write a binary file in the format described here
(version 5). Integers and floats are little-endian and fixed-width; only
little-endian hosts are supported.

## Versions

| Version | Written by | Change |
| :--- | :--- | :--- |
| 1 | rclib 0.2.0 | Initial format. |
| 2 | rclib after 0.2.0 | `RandomSparseReservoir` stores `spectral_radius_method`. |
| 3 | later versions | `RandomSparseReservoir` stores `bias_scaling`. |
| 4 | rclib 0.3.0 | Adds `RewardModulatedReservoir` and `RewardModulatedReadout`, and models without a readout (a `has_readout` flag). |
| 5 | later versions | `RlsReadout` stores `forgetting_method`. |

The writer always writes the current version. The reader accepts every version
up to the current one and reads a field only in the versions that have it, so
older files still load. In version 1 files, `RandomSparseReservoir` reservoirs load
with `spectral_radius_method` POWER_ITERATION, and in version 1 and 2 files with
`bias_scaling` 1, which is how they were built, and files before version 4 always
have a readout. Files before version 5 load their `RlsReadout` with
`forgetting_method` EXPONENTIAL, the only method they had. rclib 0.2.0 rejects
version 2 and later files, builds that write version 2 reject version 3 files,
builds that write version 3 reject version 4 files, and rclib 0.3.0 and other builds
that write version 4 reject version 5 files. The reward-modulated type tags are accepted only in version 4 and later
files.

## Primitives

| Type | Encoding |
| :--- | :--- |
| `bool` | `u8`, 0 or 1; anything else is rejected |
| `u8`, `i32`, `u32`, `u64`, `f64` | fixed-width, IEEE-754 for `f64` |
| `string` | `u32` length (at most 64) followed by the bytes; only used for type tags |
| dense matrix | `i64` rows, `i64` cols, then `rows * cols` `f64` values in column-major order |
| sparse matrix | `i64` rows, `i64` cols, `i64` nnz, then Eigen's compressed column layout: `i32 outer[cols + 1]`, `i32 inner[nnz]`, `f64 values[nnz]` |

Dimensions and counts must lie in `[0, INT32_MAX]`. The reader checks every size
against the bytes remaining in the input before allocating.

## Layout

```
header      "RCLIBMDL" (8 bytes), u32 format_version = 5
model       u8 connection (0 = serial, 1 = parallel), u32 n_reservoirs,
            n_reservoirs x (string type_tag, payload),
            bool has_readout (v4+), [string type_tag, payload]   (the readout)
```

The model must extend to the end of the input; trailing data is rejected.

## Component payloads

A field in brackets is present only when the flag just before it is true, so a
component that is not initialized carries no leftover matrices. A field marked
"v2+" is present from version 2 on, one marked "v3+" from version 3 on, and one
marked "v5+" from version 5 on. A
version 1 to 3 file has no `has_readout` flag and always a readout.

| Type tag | Payload |
| :--- | :--- |
| `RandomSparseReservoir` | `i32` n_neurons, `f64` spectral_radius, `f64` sparsity, `f64` leak_rate, `f64` input_scaling, `bool` include_bias, `u32` seed, `u8` spectral_radius_method (v2+), `f64` bias_scaling (v3+), sparse W_res, dense bias (1 x n), dense state (1 x n), `bool` w_in_initialized, [dense W_in] |
| `NvarReservoir` | `i32` num_lags, `i32` polynomial_order, `bool` initialized, [`i32` input_dim, dense state, dense past_inputs] |
| `RidgeReadout` | `f64` alpha, `bool` include_bias, `u8` solver, `f64` tolerance, `u8` effective_solver, `bool` fitted, [dense W_out] |
| `RlsReadout` | `f64` lambda, `f64` delta, `bool` include_bias, `u8` solver, `u8` forgetting_method (v5+), `bool` initialized, [dense W_out, dense P] |
| `LmsReadout` | `f64` learning_rate, `bool` include_bias, `bool` initialized, [dense W_out] |
| `RewardModulatedReservoir` (v4+) | `i32` n_neurons, `f64` spectral_radius, `f64` sparsity, `f64` leak_rate, `f64` input_scaling, `bool` include_bias, `u32` seed, `u8` spectral_radius_method, `f64` bias_scaling, plastic_presynaptic, plastic_postsynaptic, sparse W_res, dense bias (1 x n), dense state (1 x n), dense potential (1 x n), `bool` w_in_initialized, [dense W_in], learning |
| `RewardModulatedReadout` (v4+) | `i32` n_outputs, `bool` include_bias, learning, [dense W] when the learning state is initialized |

The reward-modulated payloads use two parts:

| Part | Encoding |
| :--- | :--- |
| plastic neuron list | `bool` present, [`u32` count, count x `i32` neuron index, sorted and distinct]; absent means all neurons |
| learning | rule, `u64` noise generator state, `u64` update count, `bool` initialized, [dense eligibility (n_pre x n_post), dense covariance (n_pre x n_pre with decorrelation, else 0 x 0), `f64` eligibility_bound, `f64` covariance_bound, `bool` post_average_initialized, dense post_average (1 x n_post), `bool` has_eligibility], `u32` n_keys, n_keys x (`i32` key, `u32` count, count x `f64` reward) |
| rule | `f64` learning_rate, `u8` noise_distribution, `f64` noise_amplitude, `f64` noise_probability, `u32` noise_seed, `u8` post, `f64` post_decay, `f64` eligibility_power, `u8` update, `f64` trace_decay, `u8` modulator, `u8` baseline, `f64` baseline_decay, `i32` baseline_window, then `bool` set, [`f64` value] for each of learning_rate_decay_steps, max_update and decorrelation |

A key's rewards are the EMA value (one entry) or the window, oldest first. The
learning state holds everything the rule needs to continue exactly, including the
noise generator (SplitMix64), so a loaded model explores and learns as the saved
one would have; rows staged and updates prepared within a call are not saved.

Solver and method codes are fixed and independent of the C++ enum order:

| Enum | Codes |
| :--- | :--- |
| `RandomSparseReservoir::SpectralRadiusMethod` | POWER_ITERATION = 0, DENSE = 1 |
| `RidgeReadout::Solver` | AUTO = 0, CHOLESKY = 1, DUAL_CHOLESKY = 2, CONJUGATE_GRADIENT = 3, CONJUGATE_GRADIENT_IMPLICIT = 4 |
| `RlsReadout::Solver` | RANK1_UPDATE = 0, RANK_K_UPDATE = 1 |
| `RlsReadout::ForgettingMethod` | EXPONENTIAL = 0, DIRECTIONAL = 1 |
| `RewardModulatedHebbianRule::Distribution` | UNIFORM = 0, GAUSSIAN = 1 |
| `RewardModulatedHebbianRule::PostSignal` | NOISE = 0, HIGH_PASS = 1, POTENTIAL = 2 |
| `RewardModulatedHebbianRule::Update` | ONLINE = 0, TRIAL = 1 |
| `RewardModulatedHebbianRule::Modulator` | DIFFERENCE = 0, HEAVISIDE = 1 |
| `RewardModulatedHebbianRule::Baseline` | EMA = 0, WINDOW = 1, NONE = 2 |

## Validation

Saving and loading run the same checks, so every file `save` writes can be
loaded:

- the model has at least one reservoir; from version 4 the readout is optional;
- hyperparameters are finite and in range; stored weights and states may hold
  any value, including NaN and infinity;
- each component's matrices have shapes consistent with its hyperparameters;
- a learning state is one the rule can reach: its eligibility, covariance and
  post-synaptic average are finite, each overflow bound is finite and at least
  the largest magnitude in its matrix, and each key's rewards are finite and fit
  the baseline kind (one EMA value, at most `baseline_window` rewards, none for
  no baseline);
- the components fit together: a serial reservoir's locked input width equals
  its predecessor's output width, parallel reservoirs share one input width, and
  a fitted readout's input width equals the combined reservoir output width. A
  RandomSparse reservoir's output width (`n_neurons`) is known even before its
  first input; NVAR's is known once its input width is. A width that is not known
  yet skips the check that needs it.

Saving additionally rejects component types other than the built-in ones and a
reservoir object that appears twice in a model.

## Changing the format

The frozen fixtures in `tests/data/serialization/v<N>` pin each version: both
test suites load them and check their configuration and results. The C++ suite
also checks that re-saving a fixture of the current version reproduces the file
byte for byte, which pins the writer. Fixtures of older versions cannot be
re-saved in their own version, so they are re-saved in the current one,
reloaded, and checked for the same configuration and results. Never regenerate
fixtures.

To change the format:

1. Bump `serialization_format_version`.
2. Read each new field only when `BinaryReader::formatVersion()` is at least the
   version that added it, and give it a default that matches how older files
   were written, so they keep loading. A reader that has not read a header
   reports the current version, so component payloads written without a header
   need no special case.
3. Add a fixture directory for the new version with its own generator, and move
   the previous version's fixture tests from the byte-for-byte check to the
   re-save and reload check.
4. Add the version to the table above.
