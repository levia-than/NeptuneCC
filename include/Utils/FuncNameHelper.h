// Helpers for stable function name generation.
#ifndef INCLUDE_UTILS_FUNCNAMEHELPER_H
#define INCLUDE_UTILS_FUNCNAMEHELPER_H 

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"

static mlir::LLVM::GlobalOp getOrCreateGlobalString(
    mlir::ModuleOp module,
    mlir::PatternRewriter &rewriter,
    llvm::StringRef symName,
    llvm::StringRef value) {

  if (auto g = module.lookupSymbol<mlir::LLVM::GlobalOp>(symName))
    return g;

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());

  auto ctx = module.getContext();
  auto i8Ty = mlir::IntegerType::get(ctx, 8);
  auto arrTy = mlir::LLVM::LLVMArrayType::get(i8Ty, value.size() + 1);

  // NUL-terminated
  std::string bytes = value.str();
  bytes.push_back('\0');
  auto attr = rewriter.getStringAttr(bytes);

  return rewriter.create<mlir::LLVM::GlobalOp>(
      module.getLoc(),
      arrTy,
      /*isConstant=*/true,
      mlir::LLVM::Linkage::External,
      symName,
      attr);
}

static mlir::Value getGlobalStringPtr(
    mlir::Location loc,
    mlir::ModuleOp module,
    mlir::PatternRewriter &rewriter,
    llvm::StringRef symName,
    llvm::StringRef value) {

  auto g = getOrCreateGlobalString(module, rewriter, symName, value);
  // g: mlir::LLVM::GlobalOp

  auto *ctx = module.getContext();
  auto i8Ty = mlir::IntegerType::get(ctx, 8);
  auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(ctx); // opaque ptr (recommended)

  // address_of gives pointer-to-global (ptr to elementType = g.getType())
  auto addr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, g);

  auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));

  return rewriter.create<mlir::LLVM::GEPOp>(
      loc,
      i8PtrTy,          // resultType: !llvm.ptr  (i8*)
      g.getType(),      // elementType: the global's value type, usually !llvm.array<... x i8>
      addr.getResult(), // basePtr: Value
      mlir::ValueRange{zero, zero} // indices: [0,0]
  );
}

#endif