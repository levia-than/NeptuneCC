module {
  func.func @k0(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>) attributes {neptunecc.block_begin_offset = 233 : i64, neptunecc.block_end_offset = 543 : i64, neptunecc.dm = "da", neptunecc.in = "x:ghosted", neptunecc.name = "k0", neptunecc.out = "y:owned", neptunecc.port_map = ["x=in0:ghosted:arg0", "y=out0:owned:arg1"], neptunecc.pragma_begin_offset = 167 : i64, neptunecc.pragma_end_offset = 560 : i64, neptunecc.tag = "k0"} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c1024 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c1024_1 = arith.constant 1024 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg3 = %c0_0 to %c1024_1 step %c1_2 {
        %c0_3 = arith.constant 0 : index
        %0 = arith.cmpi eq, %arg2, %c0_3 : index
        %c0_4 = arith.constant 0 : index
        %1 = arith.cmpi eq, %arg3, %c0_4 : index
        %2 = arith.ori %0, %1 : i1
        %c1024_5 = arith.constant 1024 : index
        %c1_6 = arith.constant 1 : index
        %3 = arith.subi %c1024_5, %c1_6 : index
        %4 = arith.cmpi eq, %arg2, %3 : index
        %5 = arith.ori %2, %4 : i1
        %c1024_7 = arith.constant 1024 : index
        %c1_8 = arith.constant 1 : index
        %6 = arith.subi %c1024_7, %c1_8 : index
        %7 = arith.cmpi eq, %arg3, %6 : index
        %8 = arith.ori %5, %7 : i1
        scf.if %8 {
          %9 = memref.load %arg0[%arg2, %arg3] : memref<1024x1024xi32>
          memref.store %9, %arg1[%arg2, %arg3] : memref<1024x1024xi32>
        } else {
          %9 = memref.load %arg0[%arg2, %arg3] : memref<1024x1024xi32>
          %c1_9 = arith.constant 1 : index
          %10 = arith.subi %arg2, %c1_9 : index
          %11 = memref.load %arg0[%10, %arg3] : memref<1024x1024xi32>
          %12 = arith.addi %9, %11 : i32
          %c1_10 = arith.constant 1 : index
          %13 = arith.addi %arg2, %c1_10 : index
          %14 = memref.load %arg0[%13, %arg3] : memref<1024x1024xi32>
          %15 = arith.addi %12, %14 : i32
          %c1_11 = arith.constant 1 : index
          %16 = arith.subi %arg3, %c1_11 : index
          %17 = memref.load %arg0[%arg2, %16] : memref<1024x1024xi32>
          %18 = arith.addi %15, %17 : i32
          %c1_12 = arith.constant 1 : index
          %19 = arith.addi %arg3, %c1_12 : index
          %20 = memref.load %arg0[%arg2, %19] : memref<1024x1024xi32>
          %21 = arith.addi %18, %20 : i32
          memref.store %21, %arg1[%arg2, %arg3] : memref<1024x1024xi32>
        }
      }
    }
    return
  }
}

