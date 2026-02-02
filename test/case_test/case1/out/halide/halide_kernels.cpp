#include "Halide.h"
#include "NeptuneHalideHelpers.h"
#include <vector>
#include <functional>
#include <cstdint>
void neptuneir_build_halide_kernels() {
  int32_t v1 = 32;
  Halide::Type v2 = Halide::Int(v1);
  int32_t v3 = 2;
  Halide::ImageParam v4 = Halide::ImageParam(v2, v3, "in0");
  Halide::Func v5 = Halide::Func("k0");
  Halide::Var v6 = Halide::Var("d0");
  Halide::Var v7 = Halide::Var("d1");
  std::vector<Halide::Argument> v8 = std::vector<Halide::Argument>();
  neptune_halide::push_arg(v8, v4);
  Halide::Target v9 = Halide::get_host_target();
  Halide::Expr v10 = std::invoke(v4, v6, v7);
  Halide::Expr v11 = Halide::operator-(v7, 1);
  Halide::Expr v12 = std::invoke(v4, v6, v11);
  Halide::Expr v13 = v10 + v12;
  Halide::Expr v14 = Halide::operator+(v7, 1);
  Halide::Expr v15 = std::invoke(v4, v6, v14);
  Halide::Expr v16 = v13 + v15;
  Halide::Expr v17 = Halide::operator-(v6, 1);
  Halide::Expr v18 = std::invoke(v4, v17, v7);
  Halide::Expr v19 = v16 + v18;
  Halide::Expr v20 = Halide::operator+(v6, 1);
  Halide::Expr v21 = std::invoke(v4, v20, v7);
  Halide::Expr v22 = v19 + v21;
  Halide::FuncRef v23 = std::invoke(v5, v6, v7);
  neptune_halide::assign(v23, v22);
  neptune_halide::schedule_2d(v5, 6, 6, 1, 2, 0, 1);
  neptune_halide::compile(v5, "k0", v8, "k0", v9);
  return;
}



