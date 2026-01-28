// RUN: neptune-opt %s -neptuneir-emitc-halide -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=MLIR
// RUN: mlir-translate %t.mlir -mlir-to-cpp -file-id=halide_kernels | FileCheck %s --check-prefix=CPP

#loc = #neptune.location<"cell">
#b = #neptune.bounds<lb = [0, 0], ub = [4, 4]>
!temp = !neptune.temp<element = f32, bounds = #b, location = #loc>
!field = !neptune.field<element = f32, bounds = #b, location = #loc,
                        layout = #neptune.layout<order = "zyx">>

module {
  func.func @k0(%in: memref<4x4xf32>, %out: memref<4x4xf32>) {
    %fin = neptune.ir.wrap %in : memref<4x4xf32> -> !field
    %fout = neptune.ir.wrap %out : memref<4x4xf32> -> !field
    %tin = neptune.ir.load %fin : !field -> !temp
    %tout = neptune.ir.apply(%tin) attributes {bounds = #b}
      : (!temp) -> !temp {
        ^bb0(%t0: !temp):
          %v0 = neptune.ir.access %t0[0, 0] : !temp -> f32
          %cst = arith.constant 1.0 : f32
          %v1 = arith.addf %v0, %cst : f32
          neptune.ir.yield %v1 : f32
      }
    neptune.ir.store %tout to %fout : !temp to !field
    return
  }
}

// MLIR-LABEL: func @neptuneir_build_halide_kernels
// MLIR: call_opaque "std::invoke"
// MLIR-NOT: emitc.verbatim

// CPP: #include "Halide.h"
// CPP: #include "NeptuneHalideHelpers.h"
// CPP: #include <vector>
// CPP: #include <functional>
// CPP: #include <cstdint>
// CPP: void neptuneir_build_halide_kernels()
// CPP: std::invoke
// CPP: neptune_halide::compile
