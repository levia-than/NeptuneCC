#ifndef NEPTUNEIR_PASSES_TD
#define NEPTUNEIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def NeptuneIRVerifyFormsPass : Pass<"neptune-ir-verify-forms", "mlir::ModuleOp"> {
  let summary = "verify NeptuneIR structural invariants (apply/access/temp/field)";
  let dependentDialects = [
    "Neptune::NeptuneIR::NeptuneIRDialect"
  ];
}

def NeptuneIRNormalizeApplyPass
    : Pass<"neptune-ir-normalize-apply", "mlir::ModuleOp"> {
  let summary = "Normalize neptune.ir.apply: infer/verify radius, canonicalize access";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "Neptune::NeptuneIR::NeptuneIRDialect"
  ];
}

def NeptuneIRSplitDomainPass : Pass<"neptune-ir-split-domain", "mlir::ModuleOp"> {
  let summary = "split apply into interior/boundary regions for overlap/codegen";
  let dependentDialects = [
    "Neptune::NeptuneIR::NeptuneIRDialect"
  ];

  let options = [
    Option<"mode", "mode", "std::string", "\"interior-boundary\"",
           "split mode: interior-boundary | interior-only | none">,
    Option<"keepOriginal", "keep-original", "bool", /*default=*/"false",
           "keep the original apply after splitting">
  ];
}

def NeptuneIREmitCHalidePass
    : Pass<"neptune-ir-emitc-halide", "mlir::ModuleOp"> {
  let summary = "Emit Halide Generator C++ via EmitC (one file for all apply kernels)";
  let dependentDialects = [
    "mlir::emitc::EmitCDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "Neptune::NeptuneIR::NeptuneIRDialect"
  ];

  let options = [
    Option<"outFile", "out-file", "std::string",
           /*default=*/"\"neptune_kernels_generators.cpp\"",
           "emitc output filename for generated halide generators">
  ];
}

def SCFToNeptuneIRPass : Pass<"neptunecc-scf-to-neptuneir", "mlir::func::FuncOp"> {
  let summary = "extract 2D 5-point scf.for stencils into neptune.ir.apply";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "Neptune::NeptuneIR::NeptuneIRDialect"
  ];
}

#endif // NEPTUNEIR_PASSES_TD
