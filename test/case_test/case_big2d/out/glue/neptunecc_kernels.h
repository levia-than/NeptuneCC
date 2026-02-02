#pragma once
#include <cstdint>
namespace neptunecc {
  int k0(int32_t* x, int32_t* y);
  int k0_interior(int32_t* x, int32_t* y);
  int k0_face_top(int32_t* x, int32_t* y);
  int k0_face_bottom(int32_t* x, int32_t* y);
  int k0_face_left(int32_t* x, int32_t* y);
  int k0_face_right(int32_t* x, int32_t* y);
} // namespace neptunecc
