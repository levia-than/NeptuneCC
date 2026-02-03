// NeptuneIR verifier utility declarations.
#ifndef NEPTUNEIR_VERIFIER_UTILS_H
#define NEPTUNEIR_VERIFIER_UTILS_H

#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <cstdlib>

namespace mlir::Neptune::NeptuneIR::detail {

inline int64_t getRankFromBounds(BoundsAttr bounds) {
  return static_cast<int64_t>(bounds.getLb().size());
}

template <typename EmitErrorFn>
inline LogicalResult verifySameLen(llvm::ArrayRef<int64_t> a,
                                   llvm::ArrayRef<int64_t> b,
                                   EmitErrorFn emitError,
                                   llvm::StringRef aName,
                                   llvm::StringRef bName) {
  if (a.size() != b.size()) {
    return emitError() << aName << " rank (" << a.size() << ") != " << bName
                       << " rank (" << b.size() << ")";
  }
  return success();
}

template <typename EmitErrorFn>
inline LogicalResult verifySubBounds(BoundsAttr inner, BoundsAttr outer,
                                     EmitErrorFn emitError) {
  auto innerLb = inner.getLb().asArrayRef();
  auto innerUb = inner.getUb().asArrayRef();
  auto outerLb = outer.getLb().asArrayRef();
  auto outerUb = outer.getUb().asArrayRef();
  if (failed(verifySameLen(innerLb, outerLb, emitError, "inner lb",
                           "outer lb")) ||
      failed(verifySameLen(innerUb, outerUb, emitError, "inner ub",
                           "outer ub"))) {
    return failure();
  }
  for (size_t i = 0; i < innerLb.size(); ++i) {
    if (innerLb[i] < outerLb[i] || innerUb[i] > outerUb[i]) {
      return emitError()
             << "sub-bounds check failed at dim " << i << ": inner=[" << innerLb[i]
             << "," << innerUb[i] << ") outer=[" << outerLb[i] << ","
             << outerUb[i] << ")";
    }
  }
  return success();
}

template <typename EmitErrorFn>
inline LogicalResult verifyOffsetsWithinRadius(llvm::ArrayRef<int64_t> offsets,
                                               llvm::ArrayRef<int64_t> radius,
                                               EmitErrorFn emitError) {
  if (offsets.size() != radius.size()) {
    return emitError() << "offsets rank (" << offsets.size()
                       << ") != radius rank (" << radius.size() << ")";
  }
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (std::abs(offsets[i]) > radius[i]) {
      return emitError()
             << "offset exceeds radius at dim " << i << ": offset=" << offsets[i]
             << " radius=" << radius[i];
    }
  }
  return success();
}

} // namespace mlir::Neptune::NeptuneIR::detail

#endif // NEPTUNEIR_VERIFIER_UTILS_H
