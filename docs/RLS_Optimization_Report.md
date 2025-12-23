# RLS Optimization Report

**Date:** December 4, 2025
**Author:** Gemini Agent

## Executive Summary

This report details the optimizations applied to the Recursive Least Squares (RLS) online learning algorithm within `rclib`. The primary goal was to improve the computational performance of the `RlsReadout::partialFit` method, which is critical for real-time applications.

**Result:** A **~5.5x speedup** was achieved in the `online_rls` benchmark (decreasing execution time from **14.06s** to **2.54s**) without compromising numerical accuracy.

## Mathematical Formulation

The standard RLS update equations used were:

1.  **Gain Calculation:**
    $$ \mathbf{k} = \frac{\mathbf{P} \mathbf{x}}{ \lambda + \mathbf{x}^T \mathbf{P} \mathbf{x} } $$
2.  **Weight Update:**
    $$ \mathbf{W} \leftarrow \mathbf{W} + \mathbf{k} \mathbf{e}^T $$
    where $\mathbf{e} = \mathbf{d} - \mathbf{W}^T \mathbf{x}$ is the prediction error.
3.  **Covariance Matrix Update:**
    $$ \mathbf{P} \leftarrow \lambda^{-1} (\mathbf{P} - \mathbf{k} \mathbf{x}^T \mathbf{P}) $$

### Optimization: Symmetric Rank-1 Update (Previous Summary)

Since $\mathbf{P}$ is a symmetric matrix (Inverse Covariance Matrix), we can optimize step 3. Note that $\mathbf{k} = \frac{\mathbf{P}\mathbf{x}}{D}$ where $D = \lambda + \mathbf{x}^T \mathbf{P} \mathbf{x}$.

Substituting $\mathbf{k}$:
$$ \mathbf{k} \mathbf{x}^T \mathbf{P} = \frac{\mathbf{P}\mathbf{x} (\mathbf{P}\mathbf{x})^T}{D} $$

Thus the update becomes a symmetric rank-1 update:
$$ \mathbf{P} \leftarrow \lambda^{-1} \left( \mathbf{P} - \frac{(\mathbf{P}\mathbf{x})(\mathbf{P}\mathbf{x})^T}{D} \right) $$

In Eigen, this allows us to use the highly optimized `rankUpdate` method on a `selfadjointView`, which only computes and updates the upper triangular part of the matrix, reducing FLOPs by approximately 50%.

### Detailed Explanation of Symmetric Rank-1 Update

The `P` matrix in RLS represents the inverse covariance matrix, which is inherently symmetric. This property can be leveraged for significant performance gains.

#### 1. The Concept of a Rank-1 Update

A **Rank-1 update** modifies a matrix $\mathbf{A}$ by adding the outer product of two vectors, $\mathbf{u}$ and $\mathbf{v}$:
$$ \mathbf{A}_{new} = \mathbf{A} + \alpha \mathbf{u} \mathbf{v}^T $$
If $\mathbf{u} = \mathbf{v}$, the update is **Symmetric**:
$$ \mathbf{A}_{new} = \mathbf{A} + \alpha \mathbf{v} \mathbf{v}^T $$
This operation maintains the symmetry of the matrix $\mathbf{A}$.

#### 2. Mathematical Derivation in RLS

Let's re-examine the RLS covariance update (step 3):
$$ \mathbf{P}_{new} = \frac{1}{\lambda} (\mathbf{P} - \mathbf{k} \mathbf{x}^T \mathbf{P}) $$
The gain vector $\mathbf{k}$ is defined as:
$$ \mathbf{k} = \frac{\mathbf{P} \mathbf{x}}{\lambda + \mathbf{x}^T \mathbf{P} \mathbf{x}} $$
Let $\mathbf{v} = \mathbf{P} \mathbf{x}$ and $D = \lambda + \mathbf{x}^T \mathbf{P} \mathbf{x}$. Then $\mathbf{k} = \frac{\mathbf{v}}{D}$.
Since $\mathbf{P}$ is symmetric ($\mathbf{P} = \mathbf{P}^T$), we know that $\mathbf{x}^T \mathbf{P} = (\mathbf{P} \mathbf{x})^T = \mathbf{v}^T$.

Substituting these into the subtraction term:
$$ \mathbf{k} (\mathbf{x}^T \mathbf{P}) = \left( \frac{\mathbf{v}}{D} \right) \mathbf{v}^T = \frac{1}{D} \mathbf{v} \mathbf{v}^T $$
This clearly shows that the term being subtracted from $\mathbf{P}$ is a symmetric rank-1 matrix formed by $\frac{1}{D} (\mathbf{P}\mathbf{x})(\mathbf{P}\mathbf{x})^T$.

Thus, the optimized covariance matrix update becomes:
$$ \mathbf{P} \leftarrow \lambda^{-1} \left( \mathbf{P} - \frac{(\mathbf{P}\mathbf{x})(\mathbf{P}\mathbf{x})^T}{\lambda + \mathbf{x}^T \mathbf{P} \mathbf{x}} \right) $$

#### 3. Computational Advantages

Exploiting this symmetric rank-1 structure offers significant performance benefits:

*   **Reduced FLOPs:** For an $N \times N$ matrix, a full update typically involves $O(N^2)$ operations. By only computing and updating the unique elements (e.g., the upper or lower triangular part), the number of floating-point operations can be reduced by nearly 50% (from $N^2$ to $\frac{N(N+1)}{2}$).
*   **Improved Memory Access:** Modifying only half of the matrix significantly reduces memory read/write traffic, which can be a major bottleneck for large matrices that don't fit entirely into CPU caches.
*   **Optimized Library Functions:** Linear algebra libraries like Eigen provide highly optimized functions for symmetric rank-1 updates (e.g., `Eigen::SelfAdjointView::rankUpdate`). These functions are often implemented using specialized algorithms and CPU intrinsics (like AVX/SSE) that are much faster than general matrix operations.

In the `rclib` implementation, this was achieved by using `P.selfadjointView<Eigen::Upper>().rankUpdate(Px, -1.0 / (lambda * denominator));` combined with an initial scaling of `P.triangularView<Eigen::Upper>() *= (1.0 / lambda);`.

## Codebase Diffs

### 1. Header File (`cpp_core/include/rclib/readouts/RlsReadout.h`)

**Change:** Added pre-allocated temporary variables to avoid heap allocations during every `partialFit` call.

```diff
--- Old
+++ New
@@ -13,5 +13,11 @@
     Eigen::MatrixXd W_out; // Weight matrix
     Eigen::MatrixXd P;     // Inverse covariance matrix
     bool initialized;
+
+    // Pre-allocated temporaries to avoid reallocation in partialFit
+    Eigen::VectorXd x_aug;
+    Eigen::VectorXd k;
+    Eigen::VectorXd Px;
+    Eigen::RowVectorXd xP; // (Unused in final optimized version but kept for structure)
 };
```

### 2. Source File (`cpp_core/src/readouts/RlsReadout.cpp`)

**Change:** Refactored `partialFit` to use pre-allocated buffers, remove resizing, and utilize Eigen's `selfadjointView` for symmetric updates.

```diff
--- Old
+++ New
@@ -17,12 +17,13 @@

 void RlsReadout::partialFit(const Eigen::MatrixXd& state, const Eigen::MatrixXd& target) {
     Eigen::MatrixXd x = state;
-    if (include_bias) {
-        x.conservativeResize(1, x.cols() + 1);
-        x(0, x.cols() - 1) = 1.0;
-    }
+    // ... Initialization of x_aug and buffers (omitted for brevity) ...

     // RLS update equations
-    Eigen::MatrixXd Px = P * x.transpose();
-    double denominator = lambda + (x * Px)(0,0);
-    Eigen::MatrixXd k = Px / denominator;
-    Eigen::MatrixXd y_hat = x * W_out;
-    Eigen::MatrixXd error = target - y_hat;
-
-    W_out = W_out + k * error;
-
-    // Optimized P update: (1.0 / lambda) * (P - k * (x * P))
-    Eigen::MatrixXd xP = x * P;
-    P = (1.0 / lambda) * (P - k * xP);
+
+    // 1. Compute Px = P * x using symmetry (Upper triangle)
+    Px.noalias() = P.selfadjointView<Eigen::Upper>() * x_aug;
+
+    // 2. Compute denominator = lambda + x^T * Px
+    double denominator = lambda + x_aug.dot(Px);
+
+    // 3. Compute Kalman gain vector k = Px / denominator
+    k = Px / denominator;
+
+    // 4. Compute prediction y_hat = x^T * W_out and error
+    Eigen::MatrixXd error = target - (x_aug.transpose() * W_out);
+
+    // 5. Update weights: W_out = W_out + k * error
+    W_out.noalias() += k * error;
+
+    // 6. Update P: P = (1/lambda) * (P - (Px * Px^T) / denominator)
+    // Scale P
+    P.triangularView<Eigen::Upper>() *= (1.0 / lambda);
+    // Apply rank-1 update
+    P.selfadjointView<Eigen::Upper>().rankUpdate(Px, -1.0 / (lambda * denominator));
 }
```

## Performance Benchmark

Benchmarks were run on the `mackey_glass` time-series prediction task using the `performance_benchmark.cpp` executable.

| Metric | Before Optimization | After Optimization | Improvement |
| :--- | :--- | :--- | :--- |
| **Time (s)** | 14.06s | 2.54s | **5.5x Faster** |
| **MSE** | 0.000213 | 0.000215 | Negligible Diff |

The optimizations successfully removed the bottleneck in the online learning loop, making RLS a viable option for high-frequency updates.
