// Bind pragma events to source intervals.
#include "Frontend/Clang/NeptuneBinder.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <limits>

namespace neptune {

static llvm::StringRef getClauseValue(const Event &event,
                                      llvm::StringRef key) {
  for (const auto &clause : event.clauses) {
    if (clause.key == key)
      return clause.val;
  }
  return {};
}

void pairKernels(EventDB &db, clang::DiagnosticsEngine &DE) {
  db.kernels.clear();
  llvm::sort(db.events, [](const Event &lhs, const Event &rhs) {
    return lhs.fileOffset < rhs.fileOffset;
  });

  unsigned diagMissingTag = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] kernel pragma missing required tag(...)");
  unsigned diagEndWithoutBegin = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] kernel end without matching begin for tag '%0'");
  unsigned diagBeginWithoutEnd = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] kernel begin without matching end for tag '%0'");

  llvm::StringMap<llvm::SmallVector<Event, 4>> stacks;

  for (const auto &event : db.events) {
    if (event.kind != EventKind::KernelBegin &&
        event.kind != EventKind::KernelEnd) {
      continue;
    }
    if (event.tag.empty()) {
      DE.Report(event.loc, diagMissingTag);
      continue;
    }

    // Stack per tag to support nesting; each end pops the latest begin.
    llvm::StringRef tag = event.tag;
    if (event.kind == EventKind::KernelBegin) {
      stacks[tag].push_back(event);
      continue;
    }

    auto it = stacks.find(tag);
    if (it == stacks.end() || it->second.empty()) {
      DE.Report(event.loc, diagEndWithoutBegin) << tag;
      continue;
    }

    Event begin = it->second.back();
    it->second.pop_back();
    KernelInterval interval;
    interval.begin = begin;
    interval.end = event;
    db.kernels.push_back(interval);
  }

  for (auto &entry : stacks) {
    for (const auto &begin : entry.second) {
      DE.Report(begin.loc, diagBeginWithoutEnd) << begin.tag;
    }
  }
}

void pairHalos(EventDB &db, clang::DiagnosticsEngine &DE) {
  db.halos.clear();
  llvm::sort(db.events, [](const Event &lhs, const Event &rhs) {
    return lhs.fileOffset < rhs.fileOffset;
  });

  unsigned diagMissingTag = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] halo pragma missing required tag(...)");
  unsigned diagEndWithoutBegin = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] halo end without matching begin for tag '%0'");
  unsigned diagBeginWithoutEnd = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] halo begin without matching end for tag '%0'");

  llvm::StringMap<llvm::SmallVector<Event, 4>> stacks;

  for (const auto &event : db.events) {
    if (event.kind != EventKind::HaloBegin &&
        event.kind != EventKind::HaloEnd) {
      continue;
    }
    if (event.tag.empty()) {
      DE.Report(event.loc, diagMissingTag);
      continue;
    }
    // Pair halo begin/end strictly by tag, same as kernels.
    llvm::StringRef tag = event.tag;
    if (event.kind == EventKind::HaloBegin) {
      stacks[tag].push_back(event);
      continue;
    }

    auto it = stacks.find(tag);
    if (it == stacks.end() || it->second.empty()) {
      DE.Report(event.loc, diagEndWithoutBegin) << tag;
      continue;
    }
    Event begin = it->second.back();
    it->second.pop_back();
    HaloInterval interval{begin, event};
    db.halos.push_back(interval);
  }

  for (auto &entry : stacks) {
    for (const auto &begin : entry.second) {
      DE.Report(begin.loc, diagBeginWithoutEnd) << begin.tag;
    }
  }
}

void pairOverlaps(EventDB &db, clang::DiagnosticsEngine &DE) {
  db.overlaps.clear();
  llvm::sort(db.events, [](const Event &lhs, const Event &rhs) {
    return lhs.fileOffset < rhs.fileOffset;
  });

  unsigned diagMissingTag = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] overlap pragma missing required tag(...)");
  unsigned diagEndWithoutBegin = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] overlap end without matching begin for tag '%0'");
  unsigned diagBeginWithoutEnd = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] overlap begin without matching end for tag '%0'");
  unsigned diagMissingKernel = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] overlap pragma missing kernel(...) for tag '%0'");
  unsigned diagMissingHalo = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] overlap pragma missing halo(...) for tag '%0'");

  llvm::StringMap<llvm::SmallVector<Event, 4>> stacks;

  for (const auto &event : db.events) {
    if (event.kind != EventKind::OverlapBegin &&
        event.kind != EventKind::OverlapEnd) {
      continue;
    }
    if (event.tag.empty()) {
      DE.Report(event.loc, diagMissingTag);
      continue;
    }
    // Overlap requires begin/end plus kernel/halo tags on begin.
    llvm::StringRef tag = event.tag;
    if (event.kind == EventKind::OverlapBegin) {
      stacks[tag].push_back(event);
      continue;
    }

    auto it = stacks.find(tag);
    if (it == stacks.end() || it->second.empty()) {
      DE.Report(event.loc, diagEndWithoutBegin) << tag;
      continue;
    }
    Event begin = it->second.back();
    it->second.pop_back();
    OverlapInterval interval;
    interval.begin = begin;
    interval.end = event;

    // Extract required overlap metadata from begin clauses.
    llvm::StringRef kernelTag = getClauseValue(begin, "kernel");
    if (kernelTag.empty()) {
      DE.Report(begin.loc, diagMissingKernel) << begin.tag;
    } else {
      interval.kernelTag = kernelTag;
    }

    llvm::StringRef haloTag = getClauseValue(begin, "halo");
    if (haloTag.empty()) {
      DE.Report(begin.loc, diagMissingHalo) << begin.tag;
    } else {
      interval.haloTag = haloTag;
    }

    db.overlaps.push_back(interval);
  }

  for (auto &entry : stacks) {
    for (const auto &begin : entry.second) {
      DE.Report(begin.loc, diagBeginWithoutEnd) << begin.tag;
    }
  }
}

namespace {

struct BlockInfo {
  clang::SourceLocation beginLoc;
  clang::SourceLocation endLoc;
  unsigned beginOffset = 0;
  unsigned endOffset = 0;
};

class BlockCollector : public clang::RecursiveASTVisitor<BlockCollector> {
public:
  BlockCollector(clang::SourceManager &SM, const clang::LangOptions &LangOpts,
                 llvm::SmallVectorImpl<BlockInfo> &blocks)
      : SM(SM), LangOpts(LangOpts), blocks(blocks) {}

  bool VisitCompoundStmt(clang::CompoundStmt *CS) {
    clang::SourceLocation lBrace = CS->getLBracLoc();
    clang::SourceLocation rBrace = CS->getRBracLoc();
    if (lBrace.isInvalid() || rBrace.isInvalid()) {
      return true;
    }

    clang::SourceLocation rBraceEnd =
        clang::Lexer::getLocForEndOfToken(rBrace, 0, SM, LangOpts);
    if (rBraceEnd.isInvalid()) {
      return true;
    }

    BlockInfo info;
    info.beginLoc = lBrace;
    info.endLoc = rBraceEnd;
    info.beginOffset = SM.getFileOffset(SM.getExpansionLoc(lBrace));
    info.endOffset = SM.getFileOffset(SM.getExpansionLoc(rBraceEnd));
    blocks.push_back(info);
    return true;
  }

private:
  clang::SourceManager &SM;
  const clang::LangOptions &LangOpts;
  llvm::SmallVectorImpl<BlockInfo> &blocks;
};

} // namespace

void bindKernelsToBlocks(EventDB &db, clang::ASTContext &Ctx) {
  clang::SourceManager &SM = Ctx.getSourceManager();
  const clang::LangOptions &LangOpts = Ctx.getLangOpts();

  llvm::SmallVector<BlockInfo, 64> blocks;
  BlockCollector collector(SM, LangOpts, blocks);
  collector.TraverseDecl(Ctx.getTranslationUnitDecl());

  clang::DiagnosticsEngine &DE = Ctx.getDiagnostics();
  unsigned diagNoBlock = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] cannot find block { } within kernel pragma range for tag '%0'");

  for (auto &kernel : db.kernels) {
    BlockInfo *best = nullptr;
    unsigned bestBegin = std::numeric_limits<unsigned>::max();
    uint64_t bestRange = std::numeric_limits<uint64_t>::max();
    // Choose the smallest block fully contained in the pragma range.
    for (auto &block : blocks) {
      if (block.beginOffset < kernel.begin.fileOffset) {
        continue;
      }
      if (block.endOffset > kernel.end.fileOffset) {
        continue;
      }
      uint64_t range = block.endOffset - block.beginOffset;
      if (!best || block.beginOffset < bestBegin ||
          (block.beginOffset == bestBegin && range < bestRange)) {
        bestBegin = block.beginOffset;
        bestRange = range;
        best = &block;
      }
    }

    if (!best) {
      DE.Report(kernel.begin.loc, diagNoBlock) << kernel.begin.tag;
      continue;
    }

    kernel.blockBegin = best->beginLoc;
    kernel.blockEnd = best->endLoc;
    kernel.blockBeginOffset = best->beginOffset;
    kernel.blockEndOffset = best->endOffset;
  }
}

void bindOverlapsToBlocks(EventDB &db, clang::ASTContext &Ctx) {
  clang::SourceManager &SM = Ctx.getSourceManager();
  const clang::LangOptions &LangOpts = Ctx.getLangOpts();

  llvm::SmallVector<BlockInfo, 64> blocks;
  BlockCollector collector(SM, LangOpts, blocks);
  collector.TraverseDecl(Ctx.getTranslationUnitDecl());

  clang::DiagnosticsEngine &DE = Ctx.getDiagnostics();
  unsigned diagNoBlock = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] cannot find block { } within overlap pragma range for tag '%0'");

  for (auto &overlap : db.overlaps) {
    BlockInfo *best = nullptr;
    unsigned bestBegin = std::numeric_limits<unsigned>::max();
    uint64_t bestRange = std::numeric_limits<uint64_t>::max();
    // Same heuristic as kernels: pick the smallest enclosing block.
    for (auto &block : blocks) {
      if (block.beginOffset < overlap.begin.fileOffset) {
        continue;
      }
      if (block.endOffset > overlap.end.fileOffset) {
        continue;
      }
      uint64_t range = block.endOffset - block.beginOffset;
      if (!best || block.beginOffset < bestBegin ||
          (block.beginOffset == bestBegin && range < bestRange)) {
        bestBegin = block.beginOffset;
        bestRange = range;
        best = &block;
      }
    }

    if (!best) {
      DE.Report(overlap.begin.loc, diagNoBlock) << overlap.begin.tag;
      continue;
    }

    overlap.blockBegin = best->beginLoc;
    overlap.blockEnd = best->endLoc;
    overlap.blockBeginOffset = best->beginOffset;
    overlap.blockEndOffset = best->endOffset;
  }
}

} // namespace neptune
