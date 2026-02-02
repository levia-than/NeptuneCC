#pragma once
#include "Halide.h"
#include <string>
#include <vector>

namespace neptune_halide {

inline void push_arg(std::vector<Halide::Argument> &args,
                     const Halide::Argument &arg) {
  args.push_back(arg);
}

inline void assign(Halide::FuncRef &ref, const Halide::Expr &expr) {
  ref = expr;
}

inline void schedule_1d(Halide::Func &func, int Tx, int VL,
                        int enableParallel, int threads, int unroll) {
  Halide::Var x("d0"), xo("xo"), xi("xi");
  if (Tx > 0) {
    func.split(x, xo, xi, Tx);
  } else {
    xo = x;
    xi = x;
  }
  if (VL > 1) func.vectorize(xi, VL);
  if (enableParallel && threads > 1) func.parallel(xo);
  if (unroll > 1) func.unroll(xi, unroll);
  func.compute_root();
}

inline void schedule_2d(Halide::Func &func, int Tx, int Ty, int VL,
                        int Uy, int enableParallel, int threads) {
  Halide::Var x("d0"), y("d1"), xo("xo"), yo("yo"), xi("xi"),
      yi("yi");
  if (Tx > 0 && Ty > 0) {
    func.tile(x, y, xo, yo, xi, yi, Tx, Ty);
  } else {
    xo = x;
    yo = y;
    xi = x;
    yi = y;
  }
  if (VL > 1) func.vectorize(xi, VL);
  if (enableParallel && threads > 1) func.parallel(yo);
  if (Uy > 1) func.unroll(yi, Uy);
  func.compute_root();
}

inline void schedule_3d(Halide::Func &func, int Tx, int Ty, int Tz,
                        int VL, int Uy, int par_dim,
                        int enableParallel, int threads) {
  Halide::Var x("d0"), y("d1"), z("d2"), xo("xo"), yo("yo"),
      zo("zo"), xi("xi"), yi("yi"), zi("zi");
  if (Tx > 0 && Ty > 0 && Tz > 0) {
    func.split(x, xo, xi, Tx);
    func.split(y, yo, yi, Ty);
    func.split(z, zo, zi, Tz);
    func.reorder(xi, yi, zi, xo, yo, zo);
  } else {
    xo = x;
    yo = y;
    zo = z;
    xi = x;
    yi = y;
    zi = z;
  }
  if (VL > 1) func.vectorize(xi, VL);
  if (enableParallel && threads > 1) {
    if (par_dim == 2) func.parallel(zo);
    else if (par_dim == 1) func.parallel(yo);
  }
  if (Uy > 1) func.unroll(yi, Uy);
  func.compute_root();
}

inline void compile(Halide::Func &func, const std::string &prefix,
                    const std::vector<Halide::Argument> &args,
                    const std::string &fn, const Halide::Target &target) {
  func.compile_to_static_library(prefix, args, fn, target);
}

} // namespace neptune_halide
