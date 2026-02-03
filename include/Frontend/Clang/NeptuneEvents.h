// Frontend event types for pragmas.
#pragma once
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <memory>

namespace neptune {

enum class EventKind {
  KernelBegin,
  KernelEnd,
  HaloBegin,
  HaloEnd,
  OverlapBegin,
  OverlapEnd
};

struct ClauseKV {
  llvm::SmallString<32> key;
  llvm::SmallString<256> val;
};

struct Event {
  EventKind kind;
  clang::SourceLocation loc;     // pragma token 位置
  unsigned fileOffset = 0;       // SM.getFileOffset(SM.getExpansionLoc(loc))
  llvm::SmallString<256> filePath; // source file path for the pragma
  llvm::SmallString<64> tag;     // tag(...)
  llvm::SmallString<64> name;    // name(...) (optional)
  llvm::SmallVector<ClauseKV, 8> clauses; // 除 tag/name 外所有 key(value)
};

struct KernelInterval {
  Event begin;
  Event end;
  // 绑定结果（Commit-1 输出）
  clang::SourceLocation blockBegin;
  clang::SourceLocation blockEnd;
  unsigned blockBeginOffset = 0;
  unsigned blockEndOffset = 0;
};

struct HaloInterval {
  Event begin;
  Event end;
};

struct OverlapInterval {
  Event begin;
  Event end;
  clang::SourceLocation blockBegin;
  clang::SourceLocation blockEnd;
  unsigned blockBeginOffset = 0;
  unsigned blockEndOffset = 0;
  llvm::SmallString<64> haloTag;
  llvm::SmallString<64> kernelTag;
};

struct EventDB {
  llvm::SmallVector<Event, 32> events;
  llvm::SmallVector<KernelInterval, 16> kernels;
  llvm::SmallVector<HaloInterval, 16> halos;
  llvm::SmallVector<OverlapInterval, 8> overlaps;
  std::unique_ptr<mlir::MLIRContext> mlirContext;
  mlir::OwningOpRef<mlir::ModuleOp> kernelModule;
};

} // namespace neptune
// Defines event records for pragma blocks and rewrite mapping.
