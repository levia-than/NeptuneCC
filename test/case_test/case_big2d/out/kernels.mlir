module {
  func.func @k0(%arg0: memref<2048x2048xi32>, %arg1: memref<2048x2048xi32>) attributes {neptunecc.block_begin_offset = 246 : i64, neptunecc.block_end_offset = 444 : i64, neptunecc.dm = "da", neptunecc.in = "x:ghosted", neptunecc.name = "k0", neptunecc.out = "y:owned", neptunecc.port_map = ["x=in0:ghosted:arg0", "y=out0:owned:arg1"], neptunecc.pragma_begin_offset = 180 : i64, neptunecc.pragma_end_offset = 461 : i64, neptunecc.tag = "k0"} {
    %c1 = arith.constant 1 : index
    %c2048 = arith.constant 2048 : index
    %c1_0 = arith.constant 1 : index
    %0 = arith.subi %c2048, %c1_0 : index
    %c1_1 = arith.constant 1 : index
    scf.for %arg2 = %c1 to %0 step %c1_1 {
      %c1_2 = arith.constant 1 : index
      %c2048_3 = arith.constant 2048 : index
      %c1_4 = arith.constant 1 : index
      %1 = arith.subi %c2048_3, %c1_4 : index
      %c1_5 = arith.constant 1 : index
      scf.for %arg3 = %c1_2 to %1 step %c1_5 {
        %2 = memref.load %arg0[%arg2, %arg3] : memref<2048x2048xi32>
        %c1_6 = arith.constant 1 : index
        %3 = arith.subi %arg2, %c1_6 : index
        %4 = memref.load %arg0[%3, %arg3] : memref<2048x2048xi32>
        %5 = arith.addi %2, %4 : i32
        %c1_7 = arith.constant 1 : index
        %6 = arith.addi %arg2, %c1_7 : index
        %7 = memref.load %arg0[%6, %arg3] : memref<2048x2048xi32>
        %8 = arith.addi %5, %7 : i32
        %c1_8 = arith.constant 1 : index
        %9 = arith.subi %arg3, %c1_8 : index
        %10 = memref.load %arg0[%arg2, %9] : memref<2048x2048xi32>
        %11 = arith.addi %8, %10 : i32
        %c1_9 = arith.constant 1 : index
        %12 = arith.addi %arg3, %c1_9 : index
        %13 = memref.load %arg0[%arg2, %12] : memref<2048x2048xi32>
        %14 = arith.addi %11, %13 : i32
        memref.store %14, %arg1[%arg2, %arg3] : memref<2048x2048xi32>
      }
    }
    return
  }
}

