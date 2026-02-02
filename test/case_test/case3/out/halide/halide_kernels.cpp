#include "Halide.h"
#include "NeptuneHalideHelpers.h"
#include <vector>
#include <functional>
#include <cstdint>
void neptuneir_build_halide_kernels() {
  int32_t v1 = 32;
  Halide::Type v2 = Halide::Int(v1);
  int32_t v3 = 3;
  Halide::ImageParam v4 = Halide::ImageParam(v2, v3, "in0");
  Halide::Func v5 = Halide::Func("k0");
  Halide::Var v6 = Halide::Var("d0");
  Halide::Var v7 = Halide::Var("d1");
  Halide::Var v8 = Halide::Var("d2");
  std::vector<Halide::Argument> v9 = std::vector<Halide::Argument>();
  neptune_halide::push_arg(v9, v4);
  Halide::Target v10 = Halide::get_host_target();
  Halide::Expr v11 = std::invoke(v4, v6, v7, v8);
  Halide::Expr v12 = Halide::operator-(v6, 1);
  Halide::Expr v13 = std::invoke(v4, v12, v7, v8);
  Halide::Expr v14 = v11 + v13;
  Halide::Expr v15 = Halide::operator+(v6, 1);
  Halide::Expr v16 = std::invoke(v4, v15, v7, v8);
  Halide::Expr v17 = v14 + v16;
  Halide::Expr v18 = Halide::operator-(v7, 1);
  Halide::Expr v19 = std::invoke(v4, v6, v18, v8);
  Halide::Expr v20 = v17 + v19;
  Halide::Expr v21 = Halide::operator+(v7, 1);
  Halide::Expr v22 = std::invoke(v4, v6, v21, v8);
  Halide::Expr v23 = v20 + v22;
  Halide::Expr v24 = Halide::operator-(v8, 1);
  Halide::Expr v25 = std::invoke(v4, v6, v7, v24);
  Halide::Expr v26 = v23 + v25;
  Halide::Expr v27 = Halide::operator+(v8, 1);
  Halide::Expr v28 = std::invoke(v4, v6, v7, v27);
  Halide::Expr v29 = v26 + v28;
  Halide::FuncRef v30 = std::invoke(v5, v6, v7, v8);
  neptune_halide::assign(v30, v29);
  neptune_halide::schedule_3d(v5, 6, 6, 6, 1, 2, 0, 0, 1);
  neptune_halide::compile(v5, "k0", v9, "k0", v10);
  return;
}



