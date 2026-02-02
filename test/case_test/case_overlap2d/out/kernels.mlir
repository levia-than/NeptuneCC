module {
  func.func @k0(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) attributes {neptunecc.block_begin_offset = 444 : i64, neptunecc.block_end_offset = 650 : i64, neptunecc.dm = "da", neptunecc.in = "x:ghosted", neptunecc.name = "k0", neptunecc.out = "y:owned", neptunecc.port_map = ["x=in0:ghosted:arg0", "y=out0:owned:arg1"], neptunecc.pragma_begin_offset = 385 : i64, neptunecc.pragma_end_offset = 667 : i64, neptunecc.tag = "k0"} {
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c1 to %c63 step %c1_0 {
      %c1_1 = arith.constant 1 : index
      %c63_2 = arith.constant 63 : index
      %c1_3 = arith.constant 1 : index
      scf.for %arg3 = %c1_1 to %c63_2 step %c1_3 {
        %0 = memref.load %arg0[%arg2, %arg3] : memref<64x64xi32>
        %c1_4 = arith.constant 1 : index
        %1 = arith.subi %arg2, %c1_4 : index
        %2 = memref.load %arg0[%1, %arg3] : memref<64x64xi32>
        %3 = arith.addi %0, %2 : i32
        %c1_5 = arith.constant 1 : index
        %4 = arith.addi %arg2, %c1_5 : index
        %5 = memref.load %arg0[%4, %arg3] : memref<64x64xi32>
        %6 = arith.addi %3, %5 : i32
        %c1_6 = arith.constant 1 : index
        %7 = arith.subi %arg3, %c1_6 : index
        %8 = memref.load %arg0[%arg2, %7] : memref<64x64xi32>
        %9 = arith.addi %6, %8 : i32
        %c1_7 = arith.constant 1 : index
        %10 = arith.addi %arg3, %c1_7 : index
        %11 = memref.load %arg0[%arg2, %10] : memref<64x64xi32>
        %12 = arith.addi %9, %11 : i32
        memref.store %12, %arg1[%arg2, %arg3] : memref<64x64xi32>
      }
    }
    return
  }
}

