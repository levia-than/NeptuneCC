#pragma once
#include <cstdint>
namespace neptunecc {
  int k0(int32_t* x, int32_t* y);
  int k0_interior(int32_t* x, int32_t* y);
  int k0_face_zlo(int32_t* x, int32_t* y);
  int k0_face_zhi(int32_t* x, int32_t* y);
  int k0_face_ylo(int32_t* x, int32_t* y);
  int k0_face_yhi(int32_t* x, int32_t* y);
  int k0_face_xlo(int32_t* x, int32_t* y);
  int k0_face_xhi(int32_t* x, int32_t* y);
} // namespace neptunecc
