#include "Frontend/Clang/NeptuneMLIRGen.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include <string>

namespace neptune {
namespace {

struct KernelBlockMatch {
  const clang::FunctionDecl *func = nullptr;
  const clang::CompoundStmt *block = nullptr;
};

struct PortSpec {
  llvm::SmallString<32> name;
  llvm::SmallString<32> qualifier;
  bool isInput = false;
  unsigned roleIndex = 0;
};

static bool getBlockOffsets(const clang::CompoundStmt *CS,
                            clang::SourceManager &SM,
                            const clang::LangOptions &LangOpts,
                            unsigned &beginOffset, unsigned &endOffset) {
  clang::SourceLocation lBrace = CS->getLBracLoc();
  clang::SourceLocation rBrace = CS->getRBracLoc();
  if (lBrace.isInvalid() || rBrace.isInvalid()) {
    return false;
  }

  clang::SourceLocation rBraceEnd =
      clang::Lexer::getLocForEndOfToken(rBrace, 0, SM, LangOpts);
  if (rBraceEnd.isInvalid()) {
    return false;
  }

  beginOffset = SM.getFileOffset(SM.getExpansionLoc(lBrace));
  endOffset = SM.getFileOffset(SM.getExpansionLoc(rBraceEnd));
  return true;
}

class KernelBlockFinder
    : public clang::RecursiveASTVisitor<KernelBlockFinder> {
public:
  KernelBlockFinder(clang::SourceManager &SM,
                    const clang::LangOptions &LangOpts, unsigned beginOffset,
                    unsigned endOffset)
      : SM(SM), LangOpts(LangOpts), targetBegin(beginOffset),
        targetEnd(endOffset) {}

  bool TraverseFunctionDecl(clang::FunctionDecl *FD) {
    if (!FD || !FD->hasBody()) {
      return true;
    }
    const clang::FunctionDecl *prev = currentFunc;
    currentFunc = FD;
    bool result =
        clang::RecursiveASTVisitor<KernelBlockFinder>::TraverseFunctionDecl(
            FD);
    currentFunc = prev;
    return result;
  }

  bool VisitCompoundStmt(clang::CompoundStmt *CS) {
    if (match.block) {
      return false;
    }
    unsigned beginOffset = 0;
    unsigned endOffset = 0;
    if (!getBlockOffsets(CS, SM, LangOpts, beginOffset, endOffset)) {
      return true;
    }
    if (beginOffset == targetBegin && endOffset == targetEnd) {
      match.block = CS;
      match.func = currentFunc;
      return false;
    }
    return true;
  }

  KernelBlockMatch match;

private:
  clang::SourceManager &SM;
  const clang::LangOptions &LangOpts;
  unsigned targetBegin = 0;
  unsigned targetEnd = 0;
  const clang::FunctionDecl *currentFunc = nullptr;
};

static void ensureKernelModule(EventDB &db) {
  if (!db.mlirContext) {
    db.mlirContext = std::make_unique<mlir::MLIRContext>();
    db.mlirContext->getOrLoadDialect<mlir::func::FuncDialect>();
    db.mlirContext->getOrLoadDialect<mlir::scf::SCFDialect>();
    db.mlirContext->getOrLoadDialect<mlir::arith::ArithDialect>();
    db.mlirContext->getOrLoadDialect<mlir::memref::MemRefDialect>();
  }
  if (!db.kernelModule) {
    auto loc = mlir::UnknownLoc::get(db.mlirContext.get());
    db.kernelModule = mlir::ModuleOp::create(loc);
  }
}

static void applyKernelAttrs(mlir::Operation *op,
                             const KernelInterval &kernel,
                             mlir::OpBuilder &builder) {
  llvm::StringRef tag = kernel.begin.tag;
  llvm::StringRef name =
      kernel.begin.name.empty() ? kernel.begin.tag : kernel.begin.name;
  op->setAttr("neptunecc.tag", builder.getStringAttr(tag));
  op->setAttr("neptunecc.name", builder.getStringAttr(name));
  op->setAttr("neptunecc.pragma_begin_offset",
              builder.getI64IntegerAttr(
                  static_cast<int64_t>(kernel.begin.fileOffset)));
  op->setAttr("neptunecc.pragma_end_offset",
              builder.getI64IntegerAttr(
                  static_cast<int64_t>(kernel.end.fileOffset)));
  op->setAttr("neptunecc.block_begin_offset",
              builder.getI64IntegerAttr(
                  static_cast<int64_t>(kernel.blockBeginOffset)));
  op->setAttr("neptunecc.block_end_offset",
              builder.getI64IntegerAttr(
                  static_cast<int64_t>(kernel.blockEndOffset)));

  for (const auto &clause : kernel.begin.clauses) {
    if (clause.key.empty()) {
      continue;
    }
    llvm::SmallString<64> attrName("neptunecc.");
    attrName.append(clause.key);
    op->setAttr(attrName, builder.getStringAttr(clause.val));
  }
}

static bool extractArrayInfo(clang::ASTContext &Ctx, clang::QualType type,
                             llvm::SmallVectorImpl<int64_t> &shape,
                             clang::QualType &elemType) {
  shape.clear();
  clang::QualType cur = type;
  while (const auto *arr = Ctx.getAsConstantArrayType(cur)) {
    shape.push_back(arr->getSize().getZExtValue());
    cur = arr->getElementType();
  }
  if (shape.empty()) {
    return false;
  }
  elemType = cur;
  return true;
}

static mlir::Type convertIntegerType(clang::ASTContext &Ctx,
                                     mlir::OpBuilder &builder,
                                     clang::QualType type) {
  if (!type->isIntegerType()) {
    return nullptr;
  }
  unsigned width = Ctx.getTypeSize(type);
  return mlir::IntegerType::get(builder.getContext(), width);
}

static bool parsePortClause(llvm::StringRef clauseVal, bool isInput,
                            unsigned &roleIndex,
                            llvm::SmallVectorImpl<PortSpec> &ports,
                            llvm::StringSet<> &seenPorts,
                            clang::DiagnosticsEngine &DE,
                            unsigned diagInvalidPort,
                            unsigned diagDuplicatePort,
                            clang::SourceLocation loc) {
  llvm::SmallVector<llvm::StringRef, 8> entries;
  clauseVal.split(entries, ',', -1, false);
  bool ok = true;
  for (auto entry : entries) {
    llvm::StringRef trimmed = entry.trim();
    if (trimmed.empty()) {
      continue;
    }
    auto split = trimmed.split(':');
    llvm::StringRef name = split.first.trim();
    llvm::StringRef qualifier = split.second.trim();
    if (name.empty()) {
      DE.Report(loc, diagInvalidPort) << trimmed;
      ok = false;
      continue;
    }
    if (!seenPorts.insert(name).second) {
      DE.Report(loc, diagDuplicatePort) << name;
      ok = false;
      continue;
    }
    PortSpec spec;
    spec.name = name;
    spec.qualifier = qualifier;
    spec.isInput = isInput;
    spec.roleIndex = roleIndex++;
    ports.push_back(spec);
  }
  return ok;
}

class KernelLowerer {
public:
  KernelLowerer(clang::ASTContext &Ctx, mlir::func::FuncOp func,
                const llvm::StringSet<> &portNames)
      : Ctx(Ctx), DE(Ctx.getDiagnostics()), func(func),
        builder(func.getContext()), portNames(portNames) {
    diagUnsupported = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[neptune] unsupported construct in kernel block: %0");
    diagNonCanonicalFor = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[neptune] non-canonical for loop in kernel block");
    diagMissingVar = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[neptune] unsupported variable reference in kernel block: %0");
    diagPortShadowed = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[neptune] kernel port '%0' redeclared inside kernel block");
  }

  void bindParam(const clang::VarDecl *VD, mlir::Value value) {
    bindVar(VD, value);
  }

  mlir::LogicalResult lower(const clang::CompoundStmt *block,
                            mlir::Block *entry) {
    if (!entry) {
      return mlir::failure();
    }
    builder.setInsertionPointToEnd(entry);
    pushScope();
    bool ok = succeeded(lowerStmt(const_cast<clang::CompoundStmt *>(block)));
    popScope();

    if (!ok) {
      return mlir::failure();
    }

    mlir::Location loc = builder.getUnknownLoc();
    builder.setInsertionPointToEnd(entry);
    builder.create<mlir::func::ReturnOp>(loc);
    return mlir::success();
  }

private:
  struct Scope {
    llvm::SmallVector<const clang::VarDecl *, 8> vars;
  };

  void pushScope() { scopes.emplace_back(); }

  void popScope() {
    if (scopes.empty()) {
      return;
    }
    for (const auto *decl : scopes.back().vars) {
      values.erase(decl);
    }
    scopes.pop_back();
  }

  void bindVar(const clang::VarDecl *VD, mlir::Value value) {
    values[VD] = value;
    if (!scopes.empty()) {
      scopes.back().vars.push_back(VD);
    }
  }

  mlir::LogicalResult lowerStmt(clang::Stmt *stmt) {
    if (!stmt) {
      return mlir::success();
    }
    if (auto *CS = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
      return lowerCompound(CS);
    }
    if (auto *DS = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
      return lowerDecl(DS);
    }
    if (auto *FS = llvm::dyn_cast<clang::ForStmt>(stmt)) {
      return lowerFor(FS);
    }
    if (auto *BO = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
      if (BO->isAssignmentOp() || BO->isCompoundAssignmentOp()) {
        return lowerAssign(BO);
      }
    }
    if (llvm::isa<clang::NullStmt>(stmt)) {
      return mlir::success();
    }
    return emitUnsupported(stmt->getBeginLoc(),
                           stmt->getStmtClassName());
  }

  mlir::LogicalResult lowerCompound(clang::CompoundStmt *CS) {
    pushScope();
    for (auto *child : CS->body()) {
      if (failed(lowerStmt(child))) {
        popScope();
        return mlir::failure();
      }
    }
    popScope();
    return mlir::success();
  }

  mlir::LogicalResult lowerDecl(clang::DeclStmt *DS) {
    for (auto *decl : DS->decls()) {
      auto *VD = llvm::dyn_cast<clang::VarDecl>(decl);
      if (!VD) {
        return emitUnsupported(decl->getBeginLoc(),
                               decl->getDeclKindName());
      }
      if (portNames.contains(VD->getName())) {
        DE.Report(VD->getBeginLoc(), diagPortShadowed) << VD->getName();
        return mlir::failure();
      }
      if (!VD->getType()->isIntegerType() &&
          !VD->getType()->isArrayType()) {
        return emitUnsupported(VD->getBeginLoc(), "type");
      }
      mlir::Location loc = builder.getUnknownLoc();
      if (VD->getType()->isArrayType()) {
        llvm::SmallVector<int64_t, 4> shape;
        clang::QualType elemType;
        if (!extractArrayInfo(Ctx, VD->getType(), shape, elemType)) {
          return emitUnsupported(VD->getBeginLoc(), "array");
        }
        auto mlirElemTy = convertIntegerType(Ctx, builder, elemType);
        if (!mlirElemTy) {
          return emitUnsupported(VD->getBeginLoc(), "array element type");
        }
        auto memrefTy = mlir::MemRefType::get(shape, mlirElemTy);
        auto alloc = builder.create<mlir::memref::AllocaOp>(loc, memrefTy);
        bindVar(VD, alloc);
        if (VD->hasInit()) {
          return emitUnsupported(VD->getBeginLoc(), "array init");
        }
      } else {
        auto mlirTy = convertIntegerType(Ctx, builder, VD->getType());
        if (!mlirTy) {
          return emitUnsupported(VD->getBeginLoc(), "scalar type");
        }
        auto memrefTy = mlir::MemRefType::get({}, mlirTy);
        auto alloc = builder.create<mlir::memref::AllocaOp>(loc, memrefTy);
        bindVar(VD, alloc);
        if (VD->hasInit()) {
          mlir::Value init = lowerValueExpr(VD->getInit());
          if (!init) {
            return mlir::failure();
          }
          builder.create<mlir::memref::StoreOp>(loc, init, alloc);
        }
      }
    }
    return mlir::success();
  }

  mlir::LogicalResult lowerFor(clang::ForStmt *FS) {
    auto *init = FS->getInit();
    auto *cond = FS->getCond();
    auto *inc = FS->getInc();
    if (!init || !cond || !inc) {
      return emitNonCanonicalFor(FS);
    }

    auto *declStmt = llvm::dyn_cast<clang::DeclStmt>(init);
    if (!declStmt || declStmt->isSingleDecl() == false) {
      return emitNonCanonicalFor(FS);
    }
    auto *VD = llvm::dyn_cast<clang::VarDecl>(*declStmt->decl_begin());
    if (!VD || !VD->hasInit() || !VD->getType()->isIntegerType()) {
      return emitNonCanonicalFor(FS);
    }

    auto *condBO = llvm::dyn_cast<clang::BinaryOperator>(
        cond->IgnoreParenImpCasts());
    if (!condBO || condBO->getOpcode() != clang::BO_LT) {
      return emitNonCanonicalFor(FS);
    }
    const clang::VarDecl *condVar = getVarDecl(condBO->getLHS());
    if (!condVar || condVar != VD) {
      return emitNonCanonicalFor(FS);
    }

    auto *incUO =
        llvm::dyn_cast<clang::UnaryOperator>(inc->IgnoreParenImpCasts());
    if (!incUO || (incUO->getOpcode() != clang::UO_PreInc &&
                   incUO->getOpcode() != clang::UO_PostInc)) {
      return emitNonCanonicalFor(FS);
    }
    const clang::VarDecl *incVar = getVarDecl(incUO->getSubExpr());
    if (!incVar || incVar != VD) {
      return emitNonCanonicalFor(FS);
    }

    mlir::Value lb = lowerIndexExpr(VD->getInit());
    mlir::Value ub = lowerIndexExpr(condBO->getRHS());
    if (!lb || !ub) {
      return mlir::failure();
    }

    mlir::Location loc = builder.getUnknownLoc();
    auto step =
        builder.create<mlir::arith::ConstantIndexOp>(loc, 1).getResult();
    auto forOp = builder.create<mlir::scf::ForOp>(loc, lb, ub, step);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    pushScope();
    bindVar(VD, forOp.getInductionVar());
    if (failed(lowerStmt(FS->getBody()))) {
      popScope();
      return mlir::failure();
    }
    popScope();
    if (!forOp.getBody()->mightHaveTerminator()) {
      builder.setInsertionPointToEnd(forOp.getBody());
      builder.create<mlir::scf::YieldOp>(loc);
    }
    return mlir::success();
  }

  mlir::LogicalResult lowerAssign(clang::BinaryOperator *BO) {
    mlir::Location loc = builder.getUnknownLoc();
    auto opcode = BO->getOpcode();
    if (opcode != clang::BO_Assign && opcode != clang::BO_AddAssign &&
        opcode != clang::BO_SubAssign) {
      return emitUnsupported(BO->getOperatorLoc(), "assignment");
    }
    mlir::Value rhs = lowerValueExpr(BO->getRHS());
    if (!rhs) {
      return mlir::failure();
    }

    clang::Expr *lhs = BO->getLHS()->IgnoreParenImpCasts();
    if (auto *sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(lhs)) {
      mlir::Value base;
      llvm::SmallVector<mlir::Value, 4> indices;
      if (!lowerArraySubscript(sub, base, indices)) {
        return mlir::failure();
      }
      mlir::Value value = rhs;
      if (opcode == clang::BO_AddAssign || opcode == clang::BO_SubAssign) {
        auto loaded = builder.create<mlir::memref::LoadOp>(loc, base, indices);
        if (opcode == clang::BO_AddAssign) {
          value =
              builder.create<mlir::arith::AddIOp>(loc, loaded, rhs).getResult();
        } else {
          value =
              builder.create<mlir::arith::SubIOp>(loc, loaded, rhs).getResult();
        }
      }
      builder.create<mlir::memref::StoreOp>(loc, value, base, indices);
      return mlir::success();
    }

    const clang::VarDecl *VD = getVarDecl(lhs);
    if (!VD) {
      return emitUnsupported(lhs->getBeginLoc(), "assignment lhs");
    }
    auto it = values.find(VD);
    if (it == values.end()) {
      return emitMissingVar(lhs->getBeginLoc(), VD);
    }
    mlir::Value target = it->second;
    auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(target.getType());
    if (!memrefTy || memrefTy.getRank() != 0) {
      return emitUnsupported(lhs->getBeginLoc(), "assignment target");
    }

    mlir::Value value = rhs;
    if (opcode == clang::BO_AddAssign || opcode == clang::BO_SubAssign) {
      auto loaded = builder.create<mlir::memref::LoadOp>(loc, target);
      if (opcode == clang::BO_AddAssign) {
        value =
            builder.create<mlir::arith::AddIOp>(loc, loaded, rhs).getResult();
      } else {
        value =
            builder.create<mlir::arith::SubIOp>(loc, loaded, rhs).getResult();
      }
    }
    builder.create<mlir::memref::StoreOp>(loc, value, target);
    return mlir::success();
  }

  mlir::Value lowerValueExpr(clang::Expr *expr) {
    if (!expr) {
      return nullptr;
    }
    expr = expr->IgnoreParenImpCasts();
    mlir::Location loc = builder.getUnknownLoc();

    if (auto *lit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
      auto mlirTy = convertIntegerType(Ctx, builder, expr->getType());
      if (!mlirTy) {
        (void)emitUnsupported(expr->getBeginLoc(), "literal");
        return nullptr;
      }
      return builder.create<mlir::arith::ConstantOp>(
          loc, mlirTy, builder.getIntegerAttr(mlirTy, lit->getValue()));
    }
    if (auto *DRE = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
      auto *VD = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl());
      if (!VD) {
        (void)emitMissingVar(expr->getBeginLoc(), nullptr);
        return nullptr;
      }
      auto it = values.find(VD);
      if (it == values.end()) {
        (void)emitMissingVar(expr->getBeginLoc(), VD);
        return nullptr;
      }
      mlir::Value val = it->second;
      if (auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(val.getType())) {
        if (memrefTy.getRank() != 0) {
          (void)emitUnsupported(expr->getBeginLoc(), "array value");
          return nullptr;
        }
        return builder.create<mlir::memref::LoadOp>(loc, val);
      }
      if (llvm::isa<mlir::IndexType>(val.getType())) {
        auto mlirTy = convertIntegerType(Ctx, builder, expr->getType());
        if (!mlirTy) {
          (void)emitUnsupported(expr->getBeginLoc(), "index cast");
          return nullptr;
        }
        return builder.create<mlir::arith::IndexCastOp>(loc, mlirTy, val);
      }
      return val;
    }
    if (auto *sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
      mlir::Value base;
      llvm::SmallVector<mlir::Value, 4> indices;
      if (!lowerArraySubscript(sub, base, indices)) {
        return nullptr;
      }
      return builder.create<mlir::memref::LoadOp>(loc, base, indices);
    }
    if (auto *BO = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
      if (BO->getOpcode() == clang::BO_Add ||
          BO->getOpcode() == clang::BO_Sub ||
          BO->getOpcode() == clang::BO_Mul) {
        mlir::Value lhs = lowerValueExpr(BO->getLHS());
        mlir::Value rhs = lowerValueExpr(BO->getRHS());
        if (!lhs || !rhs) {
          return nullptr;
        }
        if (BO->getOpcode() == clang::BO_Add) {
          return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        }
        if (BO->getOpcode() == clang::BO_Sub) {
          return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
        }
        return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
      }
    }
    if (auto *UO = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
      if (UO->getOpcode() == clang::UO_Minus) {
        mlir::Value val = lowerValueExpr(UO->getSubExpr());
        if (!val) {
          return nullptr;
        }
        auto zero = builder.create<mlir::arith::ConstantOp>(
            loc, val.getType(), builder.getZeroAttr(val.getType()));
        return builder.create<mlir::arith::SubIOp>(loc, zero, val);
      }
    }

    (void)emitUnsupported(expr->getBeginLoc(), expr->getStmtClassName());
    return nullptr;
  }

  mlir::Value lowerIndexExpr(clang::Expr *expr) {
    if (!expr) {
      return nullptr;
    }
    expr = expr->IgnoreParenImpCasts();
    mlir::Location loc = builder.getUnknownLoc();

    if (auto *lit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
      return builder.create<mlir::arith::ConstantIndexOp>(
          loc, lit->getValue().getSExtValue());
    }
    if (auto *DRE = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
      auto *VD = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl());
      if (!VD) {
        (void)emitMissingVar(expr->getBeginLoc(), nullptr);
        return nullptr;
      }
      auto it = values.find(VD);
      if (it == values.end()) {
        (void)emitMissingVar(expr->getBeginLoc(), VD);
        return nullptr;
      }
      mlir::Value val = it->second;
      if (llvm::isa<mlir::IndexType>(val.getType())) {
        return val;
      }
      if (auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(val.getType())) {
        if (memrefTy.getRank() != 0) {
          (void)emitUnsupported(expr->getBeginLoc(), "array index");
          return nullptr;
        }
        mlir::Value loaded = builder.create<mlir::memref::LoadOp>(loc, val);
        return builder.create<mlir::arith::IndexCastOp>(
            loc, builder.getIndexType(), loaded);
      }
      if (llvm::isa<mlir::IntegerType>(val.getType())) {
        return builder.create<mlir::arith::IndexCastOp>(
            loc, builder.getIndexType(), val);
      }
      (void)emitUnsupported(expr->getBeginLoc(), "index");
      return nullptr;
    }
    if (auto *BO = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
      if (BO->getOpcode() == clang::BO_Add ||
          BO->getOpcode() == clang::BO_Sub) {
        mlir::Value lhs = lowerIndexExpr(BO->getLHS());
        mlir::Value rhs = lowerIndexExpr(BO->getRHS());
        if (!lhs || !rhs) {
          return nullptr;
        }
        if (BO->getOpcode() == clang::BO_Add) {
          return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        }
        return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
      }
    }

    (void)emitUnsupported(expr->getBeginLoc(), expr->getStmtClassName());
    return nullptr;
  }

  bool lowerArraySubscript(clang::ArraySubscriptExpr *expr, mlir::Value &base,
                           llvm::SmallVectorImpl<mlir::Value> &indices) {
    llvm::SmallVector<clang::Expr *, 4> idxExprs;
    clang::Expr *cur = expr;
    while (auto *sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(cur)) {
      idxExprs.push_back(sub->getIdx());
      cur = sub->getBase()->IgnoreParenImpCasts();
    }
    auto *DRE = llvm::dyn_cast<clang::DeclRefExpr>(cur);
    if (!DRE) {
      (void)emitUnsupported(expr->getBeginLoc(), "array base");
      return false;
    }
    auto *VD = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl());
    if (!VD) {
      (void)emitMissingVar(expr->getBeginLoc(), nullptr);
      return false;
    }
    auto it = values.find(VD);
    if (it == values.end()) {
      (void)emitMissingVar(expr->getBeginLoc(), VD);
      return false;
    }
    base = it->second;
    if (!llvm::isa<mlir::MemRefType>(base.getType())) {
      (void)emitUnsupported(expr->getBeginLoc(), "array memref");
      return false;
    }

    indices.clear();
    for (auto itExpr = idxExprs.rbegin(); itExpr != idxExprs.rend();
         ++itExpr) {
      mlir::Value idx = lowerIndexExpr(*itExpr);
      if (!idx) {
        return false;
      }
      indices.push_back(idx);
    }
    auto memrefTy = llvm::cast<mlir::MemRefType>(base.getType());
    if (memrefTy.getRank() != static_cast<int64_t>(indices.size())) {
      (void)emitUnsupported(expr->getBeginLoc(), "array rank");
      return false;
    }
    return true;
  }

  const clang::VarDecl *getVarDecl(clang::Expr *expr) {
    if (!expr) {
      return nullptr;
    }
    expr = expr->IgnoreParenImpCasts();
    auto *DRE = llvm::dyn_cast<clang::DeclRefExpr>(expr);
    if (!DRE) {
      return nullptr;
    }
    return llvm::dyn_cast<clang::VarDecl>(DRE->getDecl());
  }

  mlir::LogicalResult emitUnsupported(clang::SourceLocation loc,
                                      llvm::StringRef kind) {
    DE.Report(loc, diagUnsupported) << kind;
    return mlir::failure();
  }

  mlir::LogicalResult emitNonCanonicalFor(clang::ForStmt *FS) {
    DE.Report(FS->getForLoc(), diagNonCanonicalFor);
    return mlir::failure();
  }

  mlir::LogicalResult emitMissingVar(clang::SourceLocation loc,
                                     const clang::VarDecl *VD) {
    llvm::StringRef name = VD ? VD->getName() : llvm::StringRef("<unknown>");
    DE.Report(loc, diagMissingVar) << name;
    return mlir::failure();
  }

  clang::ASTContext &Ctx;
  clang::DiagnosticsEngine &DE;
  mlir::func::FuncOp func;
  mlir::OpBuilder builder;
  const llvm::StringSet<> &portNames;
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> values;
  llvm::SmallVector<Scope, 8> scopes;
  unsigned diagUnsupported = 0;
  unsigned diagNonCanonicalFor = 0;
  unsigned diagMissingVar = 0;
  unsigned diagPortShadowed = 0;
};

} // namespace

void lowerKernelsToMLIR(EventDB &localDb, clang::ASTContext &Ctx,
                        EventDB &outDb) {
  if (localDb.kernels.empty()) {
    return;
  }
  ensureKernelModule(outDb);
  clang::SourceManager &SM = Ctx.getSourceManager();
  const clang::LangOptions &LangOpts = Ctx.getLangOpts();
  clang::DiagnosticsEngine &DE = Ctx.getDiagnostics();

  unsigned diagMissingPort = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] kernel port '%0' not found in function parameters");
  unsigned diagInvalidPort = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] invalid kernel port spec '%0'");
  unsigned diagDuplicatePort = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] duplicate kernel port '%0'");
  unsigned diagPortType = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "[neptune] unsupported kernel port type for '%0'");

  for (const auto &kernel : localDb.kernels) {
    if (!kernel.blockBegin.isValid() || !kernel.blockEnd.isValid()) {
      continue;
    }
    KernelBlockFinder finder(SM, LangOpts, kernel.blockBeginOffset,
                             kernel.blockEndOffset);
    finder.TraverseDecl(Ctx.getTranslationUnitDecl());
    if (!finder.match.block) {
      unsigned diagNoBlock = Ctx.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "[neptune] cannot find kernel block for tag '%0'");
      Ctx.getDiagnostics().Report(kernel.begin.loc, diagNoBlock)
          << kernel.begin.tag;
      continue;
    }
    if (!finder.match.func) {
      unsigned diagNoFunc = Ctx.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "[neptune] cannot find kernel function for tag '%0'");
      Ctx.getDiagnostics().Report(kernel.begin.loc, diagNoFunc)
          << kernel.begin.tag;
      continue;
    }

    llvm::SmallVector<PortSpec, 8> ports;
    llvm::StringSet<> portNames;
    unsigned inIndex = 0;
    unsigned outIndex = 0;
    bool portsValid = true;
    for (const auto &clause : kernel.begin.clauses) {
      if (clause.key == "in") {
        portsValid = parsePortClause(clause.val, true, inIndex, ports, portNames,
                                     DE, diagInvalidPort, diagDuplicatePort,
                                     kernel.begin.loc) &&
                     portsValid;
      } else if (clause.key == "out") {
        portsValid = parsePortClause(clause.val, false, outIndex, ports,
                                     portNames, DE, diagInvalidPort,
                                     diagDuplicatePort, kernel.begin.loc) &&
                     portsValid;
      }
    }

    llvm::StringMap<const PortSpec *> portByName;
    for (const auto &port : ports) {
      portByName[port.name] = &port;
    }

    llvm::SmallVector<mlir::Type, 8> argTypes;
    llvm::SmallVector<const clang::ParmVarDecl *, 8> argParams;
    llvm::StringMap<unsigned> argIndexByName;
    for (const auto *param : finder.match.func->parameters()) {
      auto it = portByName.find(param->getName());
      if (it == portByName.end()) {
        continue;
      }
      llvm::SmallVector<int64_t, 4> shape;
      clang::QualType elemType;
      clang::QualType originalType = param->getOriginalType();
      if (!extractArrayInfo(Ctx, originalType, shape, elemType)) {
        DE.Report(kernel.begin.loc, diagPortType) << param->getName();
        portsValid = false;
        continue;
      }
      mlir::OpBuilder builder(outDb.kernelModule->getContext());
      auto elemMlirTy = convertIntegerType(Ctx, builder, elemType);
      if (!elemMlirTy) {
        DE.Report(kernel.begin.loc, diagPortType) << param->getName();
        portsValid = false;
        continue;
      }
      auto memrefTy = mlir::MemRefType::get(shape, elemMlirTy);
      unsigned argIndex = argTypes.size();
      argTypes.push_back(memrefTy);
      argParams.push_back(param);
      argIndexByName[param->getName()] = argIndex;
    }

    for (const auto &port : ports) {
      if (!argIndexByName.count(port.name)) {
        DE.Report(kernel.begin.loc, diagMissingPort) << port.name;
        portsValid = false;
      }
    }

    if (!portsValid) {
      continue;
    }

    mlir::OpBuilder builder(outDb.kernelModule->getContext());
    mlir::Location loc = builder.getUnknownLoc();
    builder.setInsertionPointToEnd(outDb.kernelModule->getBody());

    llvm::StringRef tag = kernel.begin.tag;
    llvm::StringRef name =
        kernel.begin.name.empty() ? kernel.begin.tag : kernel.begin.name;
    llvm::StringRef funcName = tag.empty() ? name : tag;

    auto funcType = builder.getFunctionType(argTypes, {});
    auto func = builder.create<mlir::func::FuncOp>(loc, funcName, funcType);
    applyKernelAttrs(func.getOperation(), kernel, builder);

    if (!ports.empty()) {
      llvm::SmallVector<mlir::Attribute, 8> portMapAttrs;
      portMapAttrs.reserve(ports.size());
      for (const auto &port : ports) {
        unsigned argIndex = argIndexByName[port.name];
        llvm::SmallString<64> mapping;
        mapping.append(port.name);
        mapping.append("=");
        mapping.append(port.isInput ? "in" : "out");
        mapping.append(std::to_string(port.roleIndex));
        mapping.append(":");
        if (port.qualifier.empty()) {
          mapping.append("unqualified");
        } else {
          mapping.append(port.qualifier);
        }
        mapping.append(":arg");
        mapping.append(std::to_string(argIndex));
        portMapAttrs.push_back(builder.getStringAttr(mapping));
      }
      func->setAttr("neptunecc.port_map", builder.getArrayAttr(portMapAttrs));
    }

    auto *entry = func.addEntryBlock();
    KernelLowerer lowerer(Ctx, func, portNames);
    for (unsigned idx = 0; idx < argParams.size(); ++idx) {
      lowerer.bindParam(argParams[idx], entry->getArgument(idx));
    }
    if (failed(lowerer.lower(finder.match.block, entry))) {
      func.erase();
      continue;
    }
  }
}

} // namespace neptune
