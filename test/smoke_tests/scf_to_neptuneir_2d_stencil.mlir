// RUN: neptune-opt %s -neptunecc-scf-to-neptuneir | FileCheck %s

func.func @k0(%arg0: memref<4x4xi32>, %arg1: memref<4x4xi32>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index

  scf.for %i = %c1 to %c3 step %c1 {
    scf.for %j = %c1 to %c3 step %c1 {
      %im1 = arith.subi %i, %c1 : index
      %ip1 = arith.addi %i, %c1 : index
      %jm1 = arith.subi %j, %c1 : index
      %jp1 = arith.addi %j, %c1 : index

      %c = memref.load %arg0[%i, %j] : memref<4x4xi32>
      %u = memref.load %arg0[%im1, %j] : memref<4x4xi32>
      %d = memref.load %arg0[%ip1, %j] : memref<4x4xi32>
      %l = memref.load %arg0[%i, %jm1] : memref<4x4xi32>
      %r = memref.load %arg0[%i, %jp1] : memref<4x4xi32>

      %s1 = arith.addi %c, %u : i32
      %s2 = arith.addi %s1, %d : i32
      %s3 = arith.addi %s2, %l : i32
      %s4 = arith.addi %s3, %r : i32
      memref.store %s4, %arg1[%i, %j] : memref<4x4xi32>
    }
  }
  return
}

// CHECK: neptune.ir.apply
// CHECK: ^bb0(%[[T0:.*]]: !neptune.temp
// CHECK-DAG: neptune.ir.access %[[T0]]\\[0, ?0\\]
// CHECK-DAG: neptune.ir.access %[[T0]]\\[-1, ?0\\]
// CHECK-DAG: neptune.ir.access %[[T0]]\\[1, ?0\\]
// CHECK-DAG: neptune.ir.access %[[T0]]\\[0, ?-1\\]
// CHECK-DAG: neptune.ir.access %[[T0]]\\[0, ?1\\]
// CHECK: neptune.ir.store
