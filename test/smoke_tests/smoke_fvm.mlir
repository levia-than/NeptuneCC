// RUN: neptune-opt %s --canonicalize
// (示意：核心是“FVM -> assemble_matrix -> solve_linear”，k 是运行时输入，不当常数)

#loc = #neptune_ir.location<"cell">
#b   = #neptune_ir.bounds<lb = [0, 0], ub = [32, 32]>  // Nx=32, Ny=32

!temp2d  = !neptune_ir.temp<element = f64, bounds = #b, location = #loc>
!field2d = !neptune_ir.field<element = f64, bounds = #b, location = #loc>

module {
  // ------------------------------------------------------------
  // A(u;k) for -div(k grad u)=f, Dirichlet BC via row override:
  //   boundary cell:  A(u)=u
  //   interior:       A(u)= L_k(u)   (TPFA-style 5-point with face k)
  //
  // NOTE:
  //   - arg0 = unknown u
  //   - arg1 = capture k (variable coefficient field, NOT constant)
  // ------------------------------------------------------------
  neptune_ir.linear_opdef @fvm_tpfa_A : (!temp2d, !temp2d) -> !temp2d {
  ^bb0(%u: !temp2d, %k: !temp2d):

    %Au = neptune_ir.apply(%u, %k) attributes {bounds = #b}
      : (!temp2d, !temp2d) -> !temp2d {

      // apply region args: (i, j, u, k)
      ^bb0(%i: index, %j: index, %u_in: !temp2d, %k_in: !temp2d):

        // constants
        %c0  = arith.constant 0  : index
        %c31 = arith.constant 31 : index   // Nx-1, Ny-1

        // boundary predicate: i==0 || i==Nx-1 || j==0 || j==Ny-1
        %iL = arith.cmpi eq, %i, %c0  : index
        %iR = arith.cmpi eq, %i, %c31 : index
        %jB = arith.cmpi eq, %j, %c0  : index
        %jT = arith.cmpi eq, %j, %c31 : index
        %t0 = arith.ori %iL, %iR : i1
        %t1 = arith.ori %jB, %jT : i1
        %isB = arith.ori %t0, %t1 : i1

        // local values (cell-centered)
        %uP = neptune_ir.access %u_in[0, 0] : !temp2d -> f64

        // boundary: A(u)=u  (row override)
        %out = scf.if %isB -> (f64) {
          scf.yield %uP : f64
        } else {
          // neighbors
          %uE = neptune_ir.access %u_in[ 1,  0] : !temp2d -> f64
          %uW = neptune_ir.access %u_in[-1,  0] : !temp2d -> f64
          %uN = neptune_ir.access %u_in[ 0,  1] : !temp2d -> f64
          %uS = neptune_ir.access %u_in[ 0, -1] : !temp2d -> f64

          // variable coefficient k at cells
          %kP = neptune_ir.access %k_in[0, 0] : !temp2d -> f64
          %kE0 = neptune_ir.access %k_in[ 1,  0] : !temp2d -> f64
          %kW0 = neptune_ir.access %k_in[-1,  0] : !temp2d -> f64
          %kN0 = neptune_ir.access %k_in[ 0,  1] : !temp2d -> f64
          %kS0 = neptune_ir.access %k_in[ 0, -1] : !temp2d -> f64

          // face k via harmonic mean: k_face = 2*kP*kNb/(kP+kNb)
          %two = arith.constant 2.0 : f64

          %kPkE = arith.mulf %kP, %kE0 : f64
          %kPkW = arith.mulf %kP, %kW0 : f64
          %kPkN = arith.mulf %kP, %kN0 : f64
          %kPkS = arith.mulf %kP, %kS0 : f64

          %kPpE = arith.addf %kP, %kE0 : f64
          %kPpW = arith.addf %kP, %kW0 : f64
          %kPpN = arith.addf %kP, %kN0 : f64
          %kPpS = arith.addf %kP, %kS0 : f64

          %numE = arith.mulf %two, %kPkE : f64
          %numW = arith.mulf %two, %kPkW : f64
          %numN = arith.mulf %two, %kPkN : f64
          %numS = arith.mulf %two, %kPkS : f64

          %kE = arith.divf %numE, %kPpE : f64
          %kW = arith.divf %numW, %kPpW : f64
          %kN = arith.divf %numN, %kPpN : f64
          %kS = arith.divf %numS, %kPpS : f64

          // geometry (uniform grid here; dx=dy=1/31 as an example)
          %invdx2 = arith.constant 961.0 : f64  // (31^2)  == 1/dx^2
          %invdy2 = arith.constant 961.0 : f64

          // TPFA 5-pt operator for L(u)= -div(k grad u):
          // aE = kE/dx^2, etc
          %aE = arith.mulf %kE, %invdx2 : f64
          %aW = arith.mulf %kW, %invdx2 : f64
          %aN = arith.mulf %kN, %invdy2 : f64
          %aS = arith.mulf %kS, %invdy2 : f64
          %aP0 = arith.addf %aE, %aW : f64
          %aP1 = arith.addf %aN, %aS : f64
          %aP  = arith.addf %aP0, %aP1 : f64

          %tE = arith.mulf %aE, %uE : f64
          %tW = arith.mulf %aW, %uW : f64
          %tN = arith.mulf %aN, %uN : f64
          %tS = arith.mulf %aS, %uS : f64
          %tP = arith.mulf %aP, %uP : f64

          // L(u)_P = aP*uP - aE*uE - aW*uW - aN*uN - aS*uS
          %s0 = arith.addf %tE, %tW : f64
          %s1 = arith.addf %tN, %tS : f64
          %s2 = arith.addf %s0, %s1 : f64
          %Lu = arith.subf %tP, %s2 : f64

          scf.yield %Lu : f64
        }

        neptune_ir.yield %out : f64
      }
    neptune_ir.return %Au : !temp2d
  }

  // ------------------------------------------------------------
  // Build RHS b = f  (Dirichlet: b=g on boundary, interior b=f )
  // （如果你想演示“单隐式时间步”，这里改成 b = u_n + dt*f 即可）
  // ------------------------------------------------------------
  func.func @entry(
      %out: memref<?x?xf64>,
      %k_in: memref<?x?xf64>,
      %f_in: memref<?x?xf64>,
      %g_bc: memref<?x?xf64>) -> memref<?x?xf64> {

    %fout = neptune_ir.wrap %out  : memref<?x?xf64> -> !field2d
    %fk   = neptune_ir.wrap %k_in : memref<?x?xf64> -> !field2d
    %ff   = neptune_ir.wrap %f_in : memref<?x?xf64> -> !field2d
    %fg   = neptune_ir.wrap %g_bc : memref<?x?xf64> -> !field2d

    %k = neptune_ir.load %fk : !field2d -> !temp2d
    %f = neptune_ir.load %ff : !field2d -> !temp2d
    %g = neptune_ir.load %fg : !field2d -> !temp2d

    %b = neptune_ir.apply(%f, %g) attributes {bounds = #b}
      : (!temp2d, !temp2d) -> !temp2d {
      ^bb0(%i: index, %j: index, %f_in: !temp2d, %g_in: !temp2d):
        %c0  = arith.constant 0  : index
        %c31 = arith.constant 31 : index
        %iL = arith.cmpi eq, %i, %c0  : index
        %iR = arith.cmpi eq, %i, %c31 : index
        %jB = arith.cmpi eq, %j, %c0  : index
        %jT = arith.cmpi eq, %j, %c31 : index
        %t0 = arith.ori %iL, %iR : i1
        %t1 = arith.ori %jB, %jT : i1
        %isB = arith.ori %t0, %t1 : i1

        %fv = neptune_ir.access %f_in[0,0] : !temp2d -> f64
        %gv = neptune_ir.access %g_in[0,0] : !temp2d -> f64

        %out = scf.if %isB -> (f64) { scf.yield %gv : f64 }
                            else  { scf.yield %fv : f64 }
        neptune_ir.yield %out : f64
      }

    // Assemble explicit matrix from linear_opdef with capture k
    %A = neptune_ir.assemble_matrix @fvm_tpfa_A(%k)
        : (!temp2d) -> memref<?x?xf64>

    // Solve A u = b (assembled mode)
    %u = neptune_ir.solve_linear %A, %b {
           solver = "cg",
           tol = 1.0e-10,
           max_iters = 200
         } : (memref<?x?xf64>, !temp2d) -> !temp2d

    neptune_ir.store %u to %fout : !temp2d to !field2d
    %res = neptune_ir.unwrap %fout : !field2d -> memref<?x?xf64>
    func.return %res : memref<?x?xf64>
  }
}
