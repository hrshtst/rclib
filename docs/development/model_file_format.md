# Model File Format

`Model::save` / `ESN.save` write a binary file in the format described here
(version 2). Integers and floats are little-endian and fixed-width; only
little-endian hosts are supported.

## Versions

| Version | Written by | Change |
| :--- | :--- | :--- |
| 1 | rclib 0.2.0 | Initial format. |
| 2 | later versions | `RandomSparseReservoir` stores `spectral_radius_method`. |

The writer always writes the current version. The reader accepts every version
up to the current one and reads a field only in the versions that have it, so
version 1 files still load; their `RandomSparseReservoir` reservoirs load with
`spectral_radius_method` POWER_ITERATION, which is how they were built. rclib
0.2.0 rejects version 2 files.

## Primitives

| Type | Encoding |
| :--- | :--- |
| `bool` | `u8`, 0 or 1; anything else is rejected |
| `u8`, `i32`, `u32`, `f64` | fixed-width, IEEE-754 for `f64` |
| `string` | `u32` length (at most 64) followed by the bytes; only used for type tags |
| dense matrix | `i64` rows, `i64` cols, then `rows * cols` `f64` values in column-major order |
| sparse matrix | `i64` rows, `i64` cols, `i64` nnz, then Eigen's compressed column layout: `i32 outer[cols + 1]`, `i32 inner[nnz]`, `f64 values[nnz]` |

Dimensions and counts must lie in `[0, INT32_MAX]`. The reader checks every size
against the bytes remaining in the input before allocating.

## Layout

```
header      "RCLIBMDL" (8 bytes), u32 format_version = 2
model       u8 connection (0 = serial, 1 = parallel), u32 n_reservoirs,
            n_reservoirs x (string type_tag, payload),
            string type_tag, payload                  (the readout)
```

The model must extend to the end of the input; trailing data is rejected.

## Component payloads

A field in brackets is present only when the flag just before it is true, so a
component that is not initialized carries no leftover matrices. A field marked
"v2+" is present from version 2 on.

| Type tag | Payload |
| :--- | :--- |
| `RandomSparseReservoir` | `i32` n_neurons, `f64` spectral_radius, `f64` sparsity, `f64` leak_rate, `f64` input_scaling, `bool` include_bias, `u32` seed, `u8` spectral_radius_method (v2+), sparse W_res, dense bias (1 x n), dense state (1 x n), `bool` w_in_initialized, [dense W_in] |
| `NvarReservoir` | `i32` num_lags, `i32` polynomial_order, `bool` initialized, [`i32` input_dim, dense state, dense past_inputs] |
| `RidgeReadout` | `f64` alpha, `bool` include_bias, `u8` solver, `f64` tolerance, `u8` effective_solver, `bool` fitted, [dense W_out] |
| `RlsReadout` | `f64` lambda, `f64` delta, `bool` include_bias, `u8` solver, `bool` initialized, [dense W_out, dense P] |
| `LmsReadout` | `f64` learning_rate, `bool` include_bias, `bool` initialized, [dense W_out] |

Solver and method codes are fixed and independent of the C++ enum order:

| Enum | Codes |
| :--- | :--- |
| `RandomSparseReservoir::SpectralRadiusMethod` | POWER_ITERATION = 0, DENSE = 1 |
| `RidgeReadout::Solver` | AUTO = 0, CHOLESKY = 1, DUAL_CHOLESKY = 2, CONJUGATE_GRADIENT = 3, CONJUGATE_GRADIENT_IMPLICIT = 4 |
| `RlsReadout::Solver` | RANK1_UPDATE = 0, RANK_K_UPDATE = 1 |

## Validation

Saving and loading run the same checks, so every file `save` writes can be
loaded:

- the model has at least one reservoir and a readout;
- hyperparameters are finite and in range; stored weights and states may hold
  any value, including NaN and infinity;
- each component's matrices have shapes consistent with its hyperparameters;
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
