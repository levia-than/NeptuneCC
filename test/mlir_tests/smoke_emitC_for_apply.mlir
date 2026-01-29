// RUN: neptune-opt %s -neptuneir-emitc-halide -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefix=MLIR

module {
  func.func @smoke_apply_laplace3d(
      %in  : memref<16x16x16xf32>,
      %out : memref<16x16x16xf32>) {

    %fin = neptune.ir.wrap %in
      : memref<16x16x16xf32>
      -> !neptune.field<
           element=f32,
           bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
           location=#neptune.location<"cell">,
           layout=#neptune.layout<order="zyx", strides=[1,16,256], halo=[1,1,1]>
         >

    %fout = neptune.ir.wrap %out
      : memref<16x16x16xf32>
      -> !neptune.field<
           element=f32,
           bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
           location=#neptune.location<"cell">,
           layout=#neptune.layout<order="zyx", strides=[1,16,256], halo=[1,1,1]>
         >

    %tin = neptune.ir.load %fin
      : !neptune.field<
          element=f32,
          bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
          location=#neptune.location<"cell">,
          layout=#neptune.layout<order="zyx", strides=[1,16,256], halo=[1,1,1]>
        >
      -> !neptune.temp<
          element=f32,
          bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
          location=#neptune.location<"cell">>

    %tnew = neptune.ir.apply(%tin) attributes 
        { bounds=#neptune.bounds<lb=[1,1,1], ub=[15,15,15]>,
        shape=#neptune.stencil_shape<[
          array<i64: -1, 0, 0>,
          array<i64: 1, 0, 0>,
          array<i64: 0, -1, 0>,
          array<i64: 0, 1, 0>,
          array<i64: 0, 0, -1>,
          array<i64: 0, 0, 1>,
          array<i64: 0, 0, 0>
        ]>,
        radius=array<i64: 1, 1, 1> }
      : (!neptune.temp<
          element=f32,
          bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
          location=#neptune.location<"cell">>)
      -> !neptune.temp<
          element=f32,
          bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
          location=#neptune.location<"cell">> {
        ^bb0(%ix: index, %iy: index, %iz: index):
          %c2 = arith.constant 2.0 : f32

          %c  = neptune.ir.access %tin[0,0,0]
            : !neptune.temp<
                element=f32,
                bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
                location=#neptune.location<"cell">> -> f32

          %xm = neptune.ir.access %tin[-1,0,0]
            : !neptune.temp<
                element=f32,
                bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
                location=#neptune.location<"cell">> -> f32

          %xp = neptune.ir.access %tin[1,0,0]
            : !neptune.temp<
                element=f32,
                bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
                location=#neptune.location<"cell">> -> f32

          %ym = neptune.ir.access %tin[0,-1,0]
            : !neptune.temp<
                element=f32,
                bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
                location=#neptune.location<"cell">> -> f32

          %yp = neptune.ir.access %tin[0,1,0]
            : !neptune.temp<
                element=f32,
                bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
                location=#neptune.location<"cell">> -> f32

          %zm = neptune.ir.access %tin[0,0,-1]
            : !neptune.temp<
                element=f32,
                bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
                location=#neptune.location<"cell">> -> f32

          %zp = neptune.ir.access %tin[0,0,1]
            : !neptune.temp<
                element=f32,
                bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
                location=#neptune.location<"cell">> -> f32

          %s1  = arith.addf %xm, %xp : f32
          %s2  = arith.addf %ym, %yp : f32
          %s3  = arith.addf %zm, %zp : f32
          %ss  = arith.addf %s1, %s2 : f32
          %ss2 = arith.addf %ss, %s3 : f32
          %cc  = arith.mulf %c2, %c : f32
          %res = arith.subf %cc, %ss2 : f32

          neptune.ir.yield %res : f32
      }

    neptune.ir.store %tnew to %fout
      { bounds=#neptune.bounds<lb=[1,1,1], ub=[15,15,15]> }
      : !neptune.temp<
          element=f32,
          bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
          location=#neptune.location<"cell">>
        to !neptune.field<
          element=f32,
          bounds=#neptune.bounds<lb=[0,0,0], ub=[16,16,16]>,
          location=#neptune.location<"cell">,
          layout=#neptune.layout<order="zyx", strides=[1,16,256], halo=[1,1,1]>
        >

    return
  }
}

// MLIR-LABEL: func @neptuneir_build_halide_kernels
// MLIR: call_opaque "neptune_halide::compile"
// MLIR-NOT: emitc.verbatim
