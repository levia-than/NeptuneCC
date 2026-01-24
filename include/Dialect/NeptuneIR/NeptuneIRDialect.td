#ifndef NEPTUNEIR_NEPTUNEIRDIALECT_TD
#define NEPTUNEIR_NEPTUNEIRDIALECT_TD

include "mlir/IR/OpBase.td"

def NeptuneIR_Dialect : Dialect {
  let name = "neptune";
  let cppNamespace = "::mlir::Neptune::NeptuneIR";
  let summary = "NeptuneIR expression dialect for stencil-like PDE kernels";
  let useDefaultTypePrinterParser = true;
  let useDefaultAttributePrinterParser = true;
}

class NeptuneIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<NeptuneIR_Dialect, mnemonic, traits>;

class NeptuneIR_Type<string name, string typeMnemonic, list<Trait> traits = []> 
  : TypeDef<NeptuneIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic; 
}

class NeptuneIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<NeptuneIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif
