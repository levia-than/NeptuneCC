#include "neptunecc_kernels.h"
#include "HalideRuntime.h"
#include "../halide/k0.h"

namespace neptunecc {

static inline halide_type_t i32_type() {
  halide_type_t t;
  t.code = halide_type_int;
  t.bits = 32;
  t.lanes = 1;
  return t;
}

static const halide_type_t k_i32_type = i32_type();

// dim0 corresponds to the last IR index (fastest varying),
// dim1 corresponds to the first IR index. offset_elems uses
// row-major: offset = sum(outMin[d] * stride[d]).

static const halide_dimension_t k0_in0_dims[2] = {
  {-1, 2048, 1, 0},
  {-1, 2048, 2048, 0}
};
static const halide_buffer_t k0_in0_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_in0_dims),
  nullptr
};

static inline void init_k0_in0(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_dims[2] = {
  {0, 2046, 1, 0},
  {0, 2046, 2048, 0}
};
static const halide_buffer_t k0_out0_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_out0_dims),
  nullptr
};

static inline void init_k0_out0(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 2049);
}

static const halide_dimension_t k0_in0_interior_dims[2] = {
  {-2, 2048, 1, 0},
  {-2, 2048, 2048, 0}
};
static const halide_buffer_t k0_in0_interior_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_in0_interior_dims),
  nullptr
};

static inline void init_k0_in0_interior(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_interior_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_interior_dims[2] = {
  {0, 2044, 1, 0},
  {0, 2044, 2048, 0}
};
static const halide_buffer_t k0_out0_interior_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_out0_interior_dims),
  nullptr
};

static inline void init_k0_out0_interior(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_interior_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 4098);
}

static const halide_dimension_t k0_in0_face_top_dims[2] = {
  {-1, 2048, 1, 0},
  {-1, 2048, 2048, 0}
};
static const halide_buffer_t k0_in0_face_top_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_in0_face_top_dims),
  nullptr
};

static inline void init_k0_in0_face_top(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_face_top_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_face_top_dims[2] = {
  {0, 2046, 1, 0},
  {0, 1, 2048, 0}
};
static const halide_buffer_t k0_out0_face_top_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_out0_face_top_dims),
  nullptr
};

static inline void init_k0_out0_face_top(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_face_top_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 2049);
}

static const halide_dimension_t k0_in0_face_bottom_dims[2] = {
  {-1, 2048, 1, 0},
  {-2046, 2048, 2048, 0}
};
static const halide_buffer_t k0_in0_face_bottom_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_in0_face_bottom_dims),
  nullptr
};

static inline void init_k0_in0_face_bottom(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_face_bottom_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_face_bottom_dims[2] = {
  {0, 2046, 1, 0},
  {0, 1, 2048, 0}
};
static const halide_buffer_t k0_out0_face_bottom_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_out0_face_bottom_dims),
  nullptr
};

static inline void init_k0_out0_face_bottom(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_face_bottom_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 4190209);
}

static const halide_dimension_t k0_in0_face_left_dims[2] = {
  {-1, 2048, 1, 0},
  {-2, 2048, 2048, 0}
};
static const halide_buffer_t k0_in0_face_left_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_in0_face_left_dims),
  nullptr
};

static inline void init_k0_in0_face_left(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_face_left_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_face_left_dims[2] = {
  {0, 1, 1, 0},
  {0, 2044, 2048, 0}
};
static const halide_buffer_t k0_out0_face_left_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_out0_face_left_dims),
  nullptr
};

static inline void init_k0_out0_face_left(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_face_left_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 4097);
}

static const halide_dimension_t k0_in0_face_right_dims[2] = {
  {-2046, 2048, 1, 0},
  {-2, 2048, 2048, 0}
};
static const halide_buffer_t k0_in0_face_right_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_in0_face_right_dims),
  nullptr
};

static inline void init_k0_in0_face_right(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_face_right_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_face_right_dims[2] = {
  {0, 1, 1, 0},
  {0, 2044, 2048, 0}
};
static const halide_buffer_t k0_out0_face_right_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  2,
  const_cast<halide_dimension_t*>(k0_out0_face_right_dims),
  nullptr
};

static inline void init_k0_out0_face_right(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_face_right_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 6142);
}

int k0(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_interior(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_interior(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_interior(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_face_top(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_face_top(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_face_top(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_face_bottom(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_face_bottom(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_face_bottom(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_face_left(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_face_left(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_face_left(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_face_right(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_face_right(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_face_right(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

} // namespace neptunecc
