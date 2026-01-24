// NeptuneIR type definitions.
#ifndef NEPTUNEIR_TYPES_TD
#define NEPTUNEIR_TYPES_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "NeptuneIRDialect.td"

// FieldType: 背后是存储（通常是 memref），记录 elementType + bounds + location。
def NeptuneIR_FieldType
    : NeptuneIR_Type<"Field", "field"> {
  let summary = "stencil field type backed by storage";
  let description = [{
    `FieldType` models a logical field over some domain (bounds), backed
    by storage (memref or descriptor in lower dialects). `location`
    captures whether values are cell-centered, vertex-based or
    face-centered (for FVM).
  }];

let parameters = (ins "Type":$elementType,
                      NeptuneIR_BoundsAttr:$bounds,
                      NeptuneIR_LocationAttr:$location,
                      NeptuneIR_LayoutAttr:$layout);
                      
let assemblyFormat = [{
  `<` `element` `=` $elementType `,`
     `bounds` `=` $bounds `,`
     `location` `=` $location `,`
     `layout` `=` $layout
  `>`
}];

  let genVerifyDecl = 1;
}

// TempType: 值语义临时场，是 apply 的输入/输出类型。
// 
def NeptuneIR_TempType
    : NeptuneIR_Type<"Temp", "temp"> {
  let summary = "value-semantics temporary field";
  let description = [{
    `TempType` represents a logical field computed by a stencil kernel.
    It has value semantics: it may be bufferized later to a memref, but
    within NeptuneIR it is just an SSA value.
  }];

  let parameters =
    (ins "Type":$elementType,
         NeptuneIR_BoundsAttr:$bounds,
         NeptuneIR_LocationAttr:$location);

  let assemblyFormat = [{
    `<`
      `element`  `=` $elementType `,`
      `bounds`   `=` $bounds      `,`
      `location` `=` $location
    `>`
  }];

  let genVerifyDecl = 1;
}

#endif // NEPTUNEIR_TYPES_TD
