module {
  func.func @k0(%arg0: memref<8x8xi32>, %arg1: memref<8x8xi32>) attributes {neptunecc.block_begin_offset = 174 : i64, neptunecc.block_end_offset = 364 : i64, neptunecc.dm = "da", neptunecc.in = "x:ghosted", neptunecc.name = "k0", neptunecc.out = "y:owned", neptunecc.port_map = ["x=in0:ghosted:arg0", "y=out0:owned:arg1"], neptunecc.pragma_begin_offset = 108 : i64, neptunecc.pragma_end_offset = 381 : i64, neptunecc.tag = "k0"} {
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c2 to %c6 step %c1 {
      %c2_0 = arith.constant 2 : index
      %c6_1 = arith.constant 6 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg3 = %c2_0 to %c6_1 step %c1_2 {
        %0 = memref.load %arg0[%arg2, %arg3] : memref<8x8xi32>
        %c1_3 = arith.constant 1 : index
        %1 = arith.subi %arg2, %c1_3 : index
        %2 = memref.load %arg0[%1, %arg3] : memref<8x8xi32>
        %3 = arith.addi %0, %2 : i32
        %c1_4 = arith.constant 1 : index
        %4 = arith.addi %arg2, %c1_4 : index
        %5 = memref.load %arg0[%4, %arg3] : memref<8x8xi32>
        %6 = arith.addi %3, %5 : i32
        %c1_5 = arith.constant 1 : index
        %7 = arith.subi %arg3, %c1_5 : index
        %8 = memref.load %arg0[%arg2, %7] : memref<8x8xi32>
        %9 = arith.addi %6, %8 : i32
        %c1_6 = arith.constant 1 : index
        %10 = arith.addi %arg3, %c1_6 : index
        %11 = memref.load %arg0[%arg2, %10] : memref<8x8xi32>
        %12 = arith.addi %9, %11 : i32
        memref.store %12, %arg1[%arg2, %arg3] : memref<8x8xi32>
      }
    }
    return
  }
}

