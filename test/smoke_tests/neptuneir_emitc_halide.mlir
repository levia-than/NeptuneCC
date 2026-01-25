// RUN: neptune-opt %s -neptuneir-emitc-halide -o %t.mlir
// RUN: mlir-translate %t.mlir -mlir-to-cpp -file-id=halide_kernels | FileCheck %s

#loc = #neptune.location<"cell">
#b = #neptune.bounds<lb = [0], ub = [4]>
!temp = !neptune.temp<element = f64, bounds = #b, location = #loc>
!field = !neptune.field<element = f64, bounds = #b, location = #loc,
                        layout = #neptune.layout<order = "zyx">>

module {
  func.func @k0(%in: memref<4xf64>, %out: memref<4xf64>) {
    %fin = neptune.ir.wrap %in : memref<4xf64> -> !field
    %fout = neptune.ir.wrap %out : memref<4xf64> -> !field
    %tin = neptune.ir.load %fin : !field -> !temp
    %tout = neptune.ir.apply(%tin) attributes {bounds = #b}
      : (!temp) -> !temp {
        ^bb0(%t0: !temp):
          %v0 = neptune.ir.access %t0[0] : !temp -> f64
          neptune.ir.yield %v0 : f64
      }
    neptune.ir.store %tout to %fout : !temp to !field
    return
  }
}

// CHECK: #include "Halide.h"
// CHECK: #include "<vector>"
// CHECK: void neptune_build_halide_kernels()
// CHECK: compile_to_static_library
