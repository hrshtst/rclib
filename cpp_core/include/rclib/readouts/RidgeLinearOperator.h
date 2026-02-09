#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

// Forward declaration
template <typename MatrixType> class RidgeLinearOperator;

namespace Eigen {
namespace internal {

// Traits must be defined before the class definition using them
template <typename MatrixType>
struct traits<RidgeLinearOperator<MatrixType>> : public Eigen::internal::traits<MatrixType> {
  typedef Eigen::Sparse StorageKind;
  typedef Eigen::Index StorageIndex;

  enum { CoeffReadCost = Eigen::Dynamic, Flags = Eigen::ColMajor };
};

} // namespace internal
} // namespace Eigen

template <typename MatrixType> class RidgeLinearOperator : public Eigen::EigenBase<RidgeLinearOperator<MatrixType>> {
public:
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Index Index;
  typedef typename MatrixType::StorageIndex StorageIndex;
  typedef Scalar CoeffReturnType;

  // Required by evaluator
  typedef RidgeLinearOperator<MatrixType> NestedExpression;

  enum {
    RowsAtCompileTime = Eigen::Dynamic,
    ColsAtCompileTime = Eigen::Dynamic,
    MaxRowsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false,
    IsVectorAtCompileTime = 0,
    InnerStrideAtCompileTime = 0,
    OuterStrideAtCompileTime = 0,
    Flags = Eigen::ColMajor
  };

  Index rows() const { return X.cols(); }
  Index cols() const { return X.cols(); }

  template <typename Rhs>
  Eigen::Product<RidgeLinearOperator, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const {
    return Eigen::Product<RidgeLinearOperator, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  RidgeLinearOperator(const MatrixType &X_ref, Scalar alpha_val) : X(X_ref), alpha(alpha_val) {}

  const MatrixType &X;
  Scalar alpha;
};

namespace Eigen {
namespace internal {

// Evaluator for RidgeLinearOperator
template <typename MatrixType>
struct evaluator<RidgeLinearOperator<MatrixType>> : evaluator_base<RidgeLinearOperator<MatrixType>> {
  typedef RidgeLinearOperator<MatrixType> XprType;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;

  enum { CoeffReadCost = traits<XprType>::CoeffReadCost, Flags = traits<XprType>::Flags };

  evaluator(const XprType &xpr) : m_d(xpr) {}

  CoeffReturnType coeff(Index row, Index col) const { return 0; }

  const XprType &m_d;
};

// Evaluator for the custom product
template <typename MatrixType, typename Rhs>
struct generic_product_impl<RidgeLinearOperator<MatrixType>, Rhs, SparseShape, DenseShape, GemvProduct>
    : generic_product_impl_base<RidgeLinearOperator<MatrixType>, Rhs,
                                generic_product_impl<RidgeLinearOperator<MatrixType>, Rhs>> {
  typedef typename Product<RidgeLinearOperator<MatrixType>, Rhs>::Scalar Scalar;

  template <typename Dest>
  static void scaleAndAddTo(Dest &dst, const RidgeLinearOperator<MatrixType> &lhs, const Rhs &rhs, Scalar alpha) {
    // dst += alpha * (lhs * rhs)
    // lhs * rhs = X^T * (X * rhs) + reg * rhs

    // 1. Temp = X * rhs
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> temp = lhs.X * rhs;

    // 2. Add X^T * Temp
    dst.noalias() += alpha * (lhs.X.transpose() * temp);

    // 3. Add regularization part: alpha_reg * rhs
    dst.noalias() += (alpha * lhs.alpha) * rhs;
  }
};

} // namespace internal
} // namespace Eigen
