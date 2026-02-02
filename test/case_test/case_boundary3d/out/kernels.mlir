module {
  func.func @k0(%arg0: memref<192x192x192xi32>, %arg1: memref<192x192x192xi32>) attributes {neptunecc.block_begin_offset = 223 : i64, neptunecc.block_end_offset = 717 : i64, neptunecc.dm = "da", neptunecc.in = "x:ghosted", neptunecc.name = "k0", neptunecc.out = "y:owned", neptunecc.port_map = ["x=in0:ghosted:arg0", "y=out0:owned:arg1"], neptunecc.pragma_begin_offset = 157 : i64, neptunecc.pragma_end_offset = 734 : i64, neptunecc.tag = "k0"} {
    %c0 = arith.constant 0 : index
    %c192 = arith.constant 192 : index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c192 step %c1 {
      %c0_0 = arith.constant 0 : index
      %c192_1 = arith.constant 192 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg3 = %c0_0 to %c192_1 step %c1_2 {
        %c0_3 = arith.constant 0 : index
        %c192_4 = arith.constant 192 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg4 = %c0_3 to %c192_4 step %c1_5 {
          %c0_6 = arith.constant 0 : index
          %0 = arith.cmpi eq, %arg2, %c0_6 : index
          %c0_7 = arith.constant 0 : index
          %1 = arith.cmpi eq, %arg3, %c0_7 : index
          %2 = arith.ori %0, %1 : i1
          %c0_8 = arith.constant 0 : index
          %3 = arith.cmpi eq, %arg4, %c0_8 : index
          %4 = arith.ori %2, %3 : i1
          %c192_9 = arith.constant 192 : index
          %c1_10 = arith.constant 1 : index
          %5 = arith.subi %c192_9, %c1_10 : index
          %6 = arith.cmpi eq, %arg2, %5 : index
          %7 = arith.ori %4, %6 : i1
          %c192_11 = arith.constant 192 : index
          %c1_12 = arith.constant 1 : index
          %8 = arith.subi %c192_11, %c1_12 : index
          %9 = arith.cmpi eq, %arg3, %8 : index
          %10 = arith.ori %7, %9 : i1
          %c192_13 = arith.constant 192 : index
          %c1_14 = arith.constant 1 : index
          %11 = arith.subi %c192_13, %c1_14 : index
          %12 = arith.cmpi eq, %arg4, %11 : index
          %13 = arith.ori %10, %12 : i1
          scf.if %13 {
            %14 = memref.load %arg0[%arg2, %arg3, %arg4] : memref<192x192x192xi32>
            memref.store %14, %arg1[%arg2, %arg3, %arg4] : memref<192x192x192xi32>
          } else {
            %14 = memref.load %arg0[%arg2, %arg3, %arg4] : memref<192x192x192xi32>
            %c1_15 = arith.constant 1 : index
            %15 = arith.subi %arg2, %c1_15 : index
            %16 = memref.load %arg0[%15, %arg3, %arg4] : memref<192x192x192xi32>
            %17 = arith.addi %14, %16 : i32
            %c1_16 = arith.constant 1 : index
            %18 = arith.addi %arg2, %c1_16 : index
            %19 = memref.load %arg0[%18, %arg3, %arg4] : memref<192x192x192xi32>
            %20 = arith.addi %17, %19 : i32
            %c1_17 = arith.constant 1 : index
            %21 = arith.subi %arg3, %c1_17 : index
            %22 = memref.load %arg0[%arg2, %21, %arg4] : memref<192x192x192xi32>
            %23 = arith.addi %20, %22 : i32
            %c1_18 = arith.constant 1 : index
            %24 = arith.addi %arg3, %c1_18 : index
            %25 = memref.load %arg0[%arg2, %24, %arg4] : memref<192x192x192xi32>
            %26 = arith.addi %23, %25 : i32
            %c1_19 = arith.constant 1 : index
            %27 = arith.subi %arg4, %c1_19 : index
            %28 = memref.load %arg0[%arg2, %arg3, %27] : memref<192x192x192xi32>
            %29 = arith.addi %26, %28 : i32
            %c1_20 = arith.constant 1 : index
            %30 = arith.addi %arg4, %c1_20 : index
            %31 = memref.load %arg0[%arg2, %arg3, %30] : memref<192x192x192xi32>
            %32 = arith.addi %29, %31 : i32
            memref.store %32, %arg1[%arg2, %arg3, %arg4] : memref<192x192x192xi32>
          }
        }
      }
    }
    return
  }
}

