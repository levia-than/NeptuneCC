// NeptuneIR op declarations.
/*
 * @Author: leviathan 670916484@qq.com
 * @Date: 2025-09-15 17:10:04
 * @LastEditors: leviathan 670916484@qq.com
 * @LastEditTime: 2025-11-27 18:45:00
 * @FilePath: /NeptuneCC/include/Dialect/NeptuneIR/NeptuneIROps.h
 * @Description: 
 * 
 * Copyright (c) 2025 by leviathan, All Rights Reserved. 
 */
#ifndef NEPTUNEIR_NEPTUNEIROPS_H
#define NEPTUNEIR_NEPTUNEIROPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/SymbolTable.h"   // 这一行很关键
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"

#include "Dialect/NeptuneIR/NeptuneIROpsEnumDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/NeptuneIR/NeptuneIROpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/NeptuneIR/NeptuneIROps.h.inc"

#endif
// Declares NeptuneIR ops and builders.
