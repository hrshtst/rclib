# Model File Format

`Model::save` / `ESN.save` write a binary file in the format described here
(version 1). Integers and floats are little-endian and fixed-width; only
little-endian hosts are supported.

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
header      "RCLIBMDL" (8 bytes), u32 format_version = 1
model       u8 connection (0 = serial, 1 = parallel), u32 n_reservoirs,
            n_reservoirs x (string type_tag, payload),
            string type_tag, payload                  (the readout)
```

The model must extend to the end of the input; trailing data is rejected.

## Component payloads

A field in brackets is present only when the flag just before it is true, so a
component that is not initialized carries no leftover matrices.

| Type tag | Payload |
| :--- | :--- |
| `RandomSparseReservoir` | `i32` n_neurons, `f64` spectral_radius, `f64` sparsity, `f64` leak_rate, `f64` input_scaling, `bool` include_bias, `u32` seed, sparse W_res, dense bias (1 x n), dense state (1 x n), `bool` w_in_initialized, [dense W_in] |
| `NvarReservoir` | `i32` num_lags, `i32` polynomial_order, `bool` initialized, [`i32` input_dim, dense state, dense past_inputs] |
| `RidgeReadout` | `f64` alpha, `bool` include_bias, `u8` solver, `f64` tolerance, `u8` effective_solver, `bool` fitted, [dense W_out] |
| `RlsReadout` | `f64` lambda, `f64` delta, `bool` include_bias, `u8` solver, `bool` initialized, [dense W_out, dense P] |
| `LmsReadout` | `f64` learning_rate, `bool` include_bias, `bool` initialized, [dense W_out] |

Solver codes are fixed and independent of the C++ enum order:

| Enum | Codes |
| :--- | :--- |
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
  a fitted readout's input width equals the combined reservoir output width.

Saving additionally rejects component types other than the built-in ones and a
reservoir object that appears twice in a model.

## Changing the format

The frozen fixtures in `tests/data/serialization/v1` pin version 1: both test
suites load them and check their configuration and results, and the C++ suite
checks that re-saving a loaded fixture reproduces the file byte for byte. Never
regenerate them. To change the format, bump `serialization_format_version`, gate
new fields on `BinaryReader::formatVersion()` so version 1 files keep loading,
and add a new fixture directory with its own generator.
