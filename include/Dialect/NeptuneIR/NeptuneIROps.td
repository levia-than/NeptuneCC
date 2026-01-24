#ifndef NEPTUNEIR_NEPTUNEIROPS_TD
#define NEPTUNEIR_NEPTUNEIROPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "NeptuneIRDialect.td"
include "NeptuneIRAttrs.td"
include "NeptuneIRTypes.td"

//------------------------------------------------------------------------------
// neptune_ir.wrap
//------------------------------------------------------------------------------
//
// neptune_ir.wrap %buffer : memref<...> -> !neptune_ir.field<...>
//
def NeptuneIR_WrapOp
    : NeptuneIR_Op<"ir.wrap", [Pure]> {
  let summary = "wrap a buffer (e.g., memref) into a Neptune field";
  let description = [{
    Wrap a buffer-like value (typically a memref) into a `FieldType`.
    This is the entry point from buffer-level IR into NeptuneIR.
  }];

  let arguments = (ins AnyMemRef:$buffer);
  let results   = (outs NeptuneIR_FieldType:$var_field);

  let assemblyFormat = [{
    $buffer attr-dict `:` type($buffer) `->` type($var_field)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.unwrap
//------------------------------------------------------------------------------
//
// neptune_ir.unwrap %field : !neptune_ir.field<...> -> memref<...>
//
def NeptuneIR_UnwrapOp
    : NeptuneIR_Op<"ir.unwrap", [NoMemoryEffect]> {
  let summary = "unwrap a Neptune field back to a buffer";
  let description = [{
    Inverse of `neptune_ir.wrap`. Used when lowering NeptuneIR back to
    buffer-level dialects or interfacing with external code.
  }];

  let arguments = (ins NeptuneIR_FieldType:$var_field);
  let results   = (outs AnyMemRef:$buffer);

  let assemblyFormat = [{
    $var_field attr-dict `:` type($var_field) `->` type($buffer)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.load
//------------------------------------------------------------------------------
//
// neptune_ir.load %field : !neptune_ir.field<...> -> !neptune_ir.temp<...>
//
def NeptuneIR_LoadOp
    : NeptuneIR_Op<"ir.load", [Pure]> {
  let summary = "load a field into a value-semantics temp";
  let description = [{
    Load a `FieldType` into a `TempType`. Bounds and location of the temp
    usually match those of the field.
  }];

  let arguments = (ins NeptuneIR_FieldType:$var_field);
  let results   = (outs NeptuneIR_TempType:$result);

  let assemblyFormat = [{
    $var_field attr-dict `:` type($var_field) `->` type($result)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.yield
//------------------------------------------------------------------------------
//
// neptune_ir.yield %val : T
//
// apply region 的 terminator.
//
def NeptuneIR_YieldOp
    : NeptuneIR_Op<"ir.yield", [Terminator]> {
  let summary = "yield scalar result(s) from an apply region";
  let description = [{
    Terminator of a `neptune_ir.apply` region. Usually yields a single
    scalar (the updated value), but multiple results are also allowed.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];

  let extraClassDeclaration = [{
    static void build(::mlir::OpBuilder &builder,
                      ::mlir::OperationState &state) {
      build(builder, state, ::mlir::ValueRange{});
    }
  }];
}

//------------------------------------------------------------------------------
// neptune_ir.return
//------------------------------------------------------------------------------
//
// neptune_ir.return %val : T
//
// Terminator for `neptune_ir.linear_opdef` bodies.
//
def NeptuneIR_ReturnOp
    : NeptuneIR_Op<"ir.return", [Terminator]> {
  let summary = "return results from a linear_opdef body";
  let description = [{
    Terminator for `neptune_ir.linear_opdef` regions. Mirrors
    `func.return` but stays in the NeptuneIR dialect so we can enforce
    linearity constraints before lowering to `func`.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];

  let extraClassDeclaration = [{
    static void build(::mlir::OpBuilder &builder,
                      ::mlir::OperationState &state) {
      build(builder, state, ValueRange{});
    }
  }];
}

//------------------------------------------------------------------------------
// neptune_ir.apply
//------------------------------------------------------------------------------
//
// neptune_ir.apply (%arg0, %arg1, ...)
//     { bounds = #neptune_ir.bounds<...>,
//       shape? = #neptune_ir.stencil_shape<...> }
//   : (!neptune_ir.temp<...>, ...) -> !neptune_ir.temp<...>
//   {
//     ^bb0(%i0 : index, %i1 : index, ...):
//       %u = neptune_ir.access %arg0[0] : ... -> f64
//       ...
//       neptune_ir.yield %u_new : f64
//   }
//
// 注意：SingleBlockImplicitTerminator = NeptuneIR_YieldOp
//
def NeptuneIR_ApplyOp
    : NeptuneIR_Op<"ir.apply",
        [Pure, SingleBlock,
        SingleBlockImplicitTerminator<"mlir::Neptune::NeptuneIR::YieldOp">]> {
  let summary = "structured stencil kernel over a domain";
  let description = [{
    `neptune_ir.apply` expresses a stencil kernel over a logical domain
    (given by `bounds`). Inputs are Temp fields; the result is a Temp
    field. Inside the region you only operate on scalars and use
    `neptune_ir.access` to read neighbors.

    Optionally, `shape` can encode the stencil pattern as a
    `StencilShapeAttr`, which is convenient for optimization passes.
  }];

  // operands
  let arguments = (ins
    Variadic<NeptuneIR_TempType>:$inputs,
    NeptuneIR_BoundsAttr:$bounds,
    OptionalAttr<NeptuneIR_StencilShapeAttr>:$shape,
    OptionalAttr<DenseI64ArrayAttr>:$radius
  );

  let results = (outs NeptuneIR_TempType:$result);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict-with-keyword
      `:` functional-type($inputs, $result)
      $body
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.access
//------------------------------------------------------------------------------
//
// neptune_ir.access %tmp[<offsets>] : !neptune_ir.temp<...> -> elementType
//
def NeptuneIR_AccessOp
    : NeptuneIR_Op<"ir.access", [Pure]> {
  let summary = "access a neighbor element from a temp at given offsets";
  let description = [{
    Access an element of a `TempType` at a compile-time constant offset
    relative to the current logical point. Offsets are given as a
    DenseI64ArrayAttr like [0], [-1], [1] in 1D, or [0,1,0] in 3D.
  }];

  let arguments = (ins
    NeptuneIR_TempType:$input,
    DenseI64ArrayAttr:$offsets
  );

  // 实际上 result 类型应该等于 input.elementType，C++ verifier 里可以检查。
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $input $offsets attr-dict
      `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}

//------------------------------------------------------------------------------
// neptune_ir.store
//------------------------------------------------------------------------------
//
// neptune_ir.store %tmp to %field
//   { bounds? = #neptune_ir.bounds<...> }
//   : !neptune_ir.temp<...> to !neptune_ir.field<...>
//
def NeptuneIR_StoreOp
    : NeptuneIR_Op<"ir.store",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "store a temp field into a destination field";
  let description = [{
    Store a `TempType` into a `FieldType`. Optional `bounds` restricts
    the write to a sub-domain (e.g., interior or boundary region).
  }];

  let arguments = (ins
    NeptuneIR_TempType:$value,
    NeptuneIR_FieldType:$var_field,
    OptionalAttr<NeptuneIR_BoundsAttr>:$bounds
  );

  let assemblyFormat = [{
    $value `to` $var_field attr-dict
      `:` type($value) `to` type($var_field)
  }];

  let hasVerifier = 1;
}


//------------------------------------------------------------------------------
// neptune_ir.reduce
//------------------------------------------------------------------------------
//
// neptune_ir.reduce %tmp in {bounds?} {kind = "sum"}
//   : !neptune_ir.temp<...> -> f64
//
// 场级别 reduction，用于 dot/norm 等线性代数骨架。
// kind 约定： "sum" | "max" | "min" | "l1" | "l2" 等。
//
def NeptuneIR_ReduceOp
    : NeptuneIR_Op<"ir.reduce", [Pure]> {
  let summary = "reduce a temp field over an optional sub-domain";
  let description = [{
    Reduce a `TempType` over the given domain to a scalar. This is the
    building block for dot products, norms, and other PDE/solver-level
    reductions.

    The `kind` attribute selects the reduction operator, e.g. "sum",
    "max", "min", "l1", "l2". Exact semantics are enforced by the
    C++ verifier / implementation.
  }];

  let arguments = (ins
    NeptuneIR_TempType:$input,
    OptionalAttr<NeptuneIR_BoundsAttr>:$bounds,
    StrAttr:$kind
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $input (`in` $bounds^)? attr-dict
      `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
}


#endif
