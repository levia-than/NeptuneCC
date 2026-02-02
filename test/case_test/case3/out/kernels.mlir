module {
  func.func @k0(%arg0: memref<8x8x8xi32>, %arg1: memref<8x8x8xi32>) attributes {neptunecc.block_begin_offset = 180 : i64, neptunecc.block_end_offset = 500 : i64, neptunecc.dm = "da", neptunecc.in = "x:ghosted", neptunecc.name = "k0", neptunecc.out = "y:owned", neptunecc.port_map = ["x=in0:ghosted:arg0", "y=out0:owned:arg1"], neptunecc.pragma_begin_offset = 114 : i64, neptunecc.pragma_end_offset = 517 : i64, neptunecc.tag = "k0"} {
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c1 to %c7 step %c1_0 {
      %c1_1 = arith.constant 1 : index
      %c7_2 = arith.constant 7 : index
      %c1_3 = arith.constant 1 : index
      scf.for %arg3 = %c1_1 to %c7_2 step %c1_3 {
        %c1_4 = arith.constant 1 : index
        %c7_5 = arith.constant 7 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg4 = %c1_4 to %c7_5 step %c1_6 {
          %0 = memref.load %arg0[%arg2, %arg3, %arg4] : memref<8x8x8xi32>
          %c1_7 = arith.constant 1 : index
          %1 = arith.subi %arg4, %c1_7 : index
          %2 = memref.load %arg0[%arg2, %arg3, %1] : memref<8x8x8xi32>
          %3 = arith.addi %0, %2 : i32
          %c1_8 = arith.constant 1 : index
          %4 = arith.addi %arg4, %c1_8 : index
          %5 = memref.load %arg0[%arg2, %arg3, %4] : memref<8x8x8xi32>
          %6 = arith.addi %3, %5 : i32
          %c1_9 = arith.constant 1 : index
          %7 = arith.subi %arg3, %c1_9 : index
          %8 = memref.load %arg0[%arg2, %7, %arg4] : memref<8x8x8xi32>
          %9 = arith.addi %6, %8 : i32
          %c1_10 = arith.constant 1 : index
          %10 = arith.addi %arg3, %c1_10 : index
          %11 = memref.load %arg0[%arg2, %10, %arg4] : memref<8x8x8xi32>
          %12 = arith.addi %9, %11 : i32
          %c1_11 = arith.constant 1 : index
          %13 = arith.subi %arg2, %c1_11 : index
          %14 = memref.load %arg0[%13, %arg3, %arg4] : memref<8x8x8xi32>
          %15 = arith.addi %12, %14 : i32
          %c1_12 = arith.constant 1 : index
          %16 = arith.addi %arg2, %c1_12 : index
          %17 = memref.load %arg0[%16, %arg3, %arg4] : memref<8x8x8xi32>
          %18 = arith.addi %15, %17 : i32
          memref.store %18, %arg1[%arg2, %arg3, %arg4] : memref<8x8x8xi32>
        }
      }
    }
    return
  }
}

