#pragma once
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallString.h"
#include "clang/Basic/SourceLocation.h"

namespace neptune {

enum class EventKind { KernelBegin, KernelEnd };

struct ClauseKV {
  llvm::SmallString<32> key;
  llvm::SmallString<256> val;
};

struct Event {
  EventKind kind;
  clang::SourceLocation loc;     // pragma token 位置
  unsigned fileOffset = 0;       // SM.getFileOffset(SM.getExpansionLoc(loc))
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

struct EventDB {
  llvm::SmallVector<Event, 32> events;
  llvm::SmallVector<KernelInterval, 16> kernels;
};

} // namespace neptune
