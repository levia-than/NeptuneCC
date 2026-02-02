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

static const halide_dimension_t k0_in0_dims[3] = {
  {-1, 192, 1, 0},
  {-1, 192, 192, 0},
  {-1, 192, 36864, 0}
};
static const halide_buffer_t k0_in0_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_in0_dims),
  nullptr
};

static inline void init_k0_in0(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_dims[3] = {
  {0, 190, 1, 0},
  {0, 190, 192, 0},
  {0, 190, 36864, 0}
};
static const halide_buffer_t k0_out0_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_out0_dims),
  nullptr
};

static inline void init_k0_out0(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 37057);
}

static const halide_dimension_t k0_in0_interior_dims[3] = {
  {-2, 192, 1, 0},
  {-2, 192, 192, 0},
  {-2, 192, 36864, 0}
};
static const halide_buffer_t k0_in0_interior_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_in0_interior_dims),
  nullptr
};

static inline void init_k0_in0_interior(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_interior_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_interior_dims[3] = {
  {0, 188, 1, 0},
  {0, 188, 192, 0},
  {0, 188, 36864, 0}
};
static const halide_buffer_t k0_out0_interior_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_out0_interior_dims),
  nullptr
};

static inline void init_k0_out0_interior(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_interior_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 74114);
}

static const halide_dimension_t k0_in0_face_zlo_dims[3] = {
  {-1, 192, 1, 0},
  {-1, 192, 192, 0},
  {-1, 192, 36864, 0}
};
static const halide_buffer_t k0_in0_face_zlo_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_in0_face_zlo_dims),
  nullptr
};

static inline void init_k0_in0_face_zlo(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_face_zlo_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_face_zlo_dims[3] = {
  {0, 1, 1, 0},
  {0, 190, 192, 0},
  {0, 190, 36864, 0}
};
static const halide_buffer_t k0_out0_face_zlo_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_out0_face_zlo_dims),
  nullptr
};

static inline void init_k0_out0_face_zlo(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_face_zlo_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 37057);
}

static const halide_dimension_t k0_in0_face_zhi_dims[3] = {
  {-190, 192, 1, 0},
  {-1, 192, 192, 0},
  {-1, 192, 36864, 0}
};
static const halide_buffer_t k0_in0_face_zhi_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_in0_face_zhi_dims),
  nullptr
};

static inline void init_k0_in0_face_zhi(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_face_zhi_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_face_zhi_dims[3] = {
  {0, 1, 1, 0},
  {0, 190, 192, 0},
  {0, 190, 36864, 0}
};
static const halide_buffer_t k0_out0_face_zhi_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_out0_face_zhi_dims),
  nullptr
};

static inline void init_k0_out0_face_zhi(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_face_zhi_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 37246);
}

static const halide_dimension_t k0_in0_face_ylo_dims[3] = {
  {-2, 192, 1, 0},
  {-1, 192, 192, 0},
  {-1, 192, 36864, 0}
};
static const halide_buffer_t k0_in0_face_ylo_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_in0_face_ylo_dims),
  nullptr
};

static inline void init_k0_in0_face_ylo(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_face_ylo_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_face_ylo_dims[3] = {
  {0, 188, 1, 0},
  {0, 1, 192, 0},
  {0, 190, 36864, 0}
};
static const halide_buffer_t k0_out0_face_ylo_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_out0_face_ylo_dims),
  nullptr
};

static inline void init_k0_out0_face_ylo(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_face_ylo_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 37058);
}

static const halide_dimension_t k0_in0_face_yhi_dims[3] = {
  {-2, 192, 1, 0},
  {-190, 192, 192, 0},
  {-1, 192, 36864, 0}
};
static const halide_buffer_t k0_in0_face_yhi_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_in0_face_yhi_dims),
  nullptr
};

static inline void init_k0_in0_face_yhi(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_face_yhi_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_face_yhi_dims[3] = {
  {0, 188, 1, 0},
  {0, 1, 192, 0},
  {0, 190, 36864, 0}
};
static const halide_buffer_t k0_out0_face_yhi_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_out0_face_yhi_dims),
  nullptr
};

static inline void init_k0_out0_face_yhi(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_face_yhi_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 73346);
}

static const halide_dimension_t k0_in0_face_xlo_dims[3] = {
  {-2, 192, 1, 0},
  {-2, 192, 192, 0},
  {-1, 192, 36864, 0}
};
static const halide_buffer_t k0_in0_face_xlo_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_in0_face_xlo_dims),
  nullptr
};

static inline void init_k0_in0_face_xlo(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_face_xlo_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_face_xlo_dims[3] = {
  {0, 188, 1, 0},
  {0, 188, 192, 0},
  {0, 1, 36864, 0}
};
static const halide_buffer_t k0_out0_face_xlo_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_out0_face_xlo_dims),
  nullptr
};

static inline void init_k0_out0_face_xlo(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_face_xlo_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 37250);
}

static const halide_dimension_t k0_in0_face_xhi_dims[3] = {
  {-2, 192, 1, 0},
  {-2, 192, 192, 0},
  {-190, 192, 36864, 0}
};
static const halide_buffer_t k0_in0_face_xhi_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_in0_face_xhi_dims),
  nullptr
};

static inline void init_k0_in0_face_xhi(halide_buffer_t &buf, int32_t *base) {
  buf = k0_in0_face_xhi_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 0);
}

static const halide_dimension_t k0_out0_face_xhi_dims[3] = {
  {0, 188, 1, 0},
  {0, 188, 192, 0},
  {0, 1, 36864, 0}
};
static const halide_buffer_t k0_out0_face_xhi_templ = {
  0,
  nullptr,
  nullptr,
  0,
  k_i32_type,
  3,
  const_cast<halide_dimension_t*>(k0_out0_face_xhi_dims),
  nullptr
};

static inline void init_k0_out0_face_xhi(halide_buffer_t &buf, int32_t *base) {
  buf = k0_out0_face_xhi_templ;
  buf.host = reinterpret_cast<uint8_t*>(base + 7004546);
}

int k0(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0(out0_buf, y);
  // copy-boundary (slabs, 3D)
  for (int64_t i = 0; i < 192; ++i) {
    for (int64_t j = 0; j < 192; ++j) {
      for (int64_t k = 0; k < 1; ++k) {
        y[((i * 192) + j) * 192 + k] = x[((i * 192) + j) * 192 + k];
      }
    }
  }
  for (int64_t i = 0; i < 192; ++i) {
    for (int64_t j = 0; j < 192; ++j) {
      for (int64_t k = 191; k < 192; ++k) {
        y[((i * 192) + j) * 192 + k] = x[((i * 192) + j) * 192 + k];
      }
    }
  }
  for (int64_t i = 0; i < 192; ++i) {
    for (int64_t j = 0; j < 1; ++j) {
      for (int64_t k = 1; k < 191; ++k) {
        y[((i * 192) + j) * 192 + k] = x[((i * 192) + j) * 192 + k];
      }
    }
  }
  for (int64_t i = 0; i < 192; ++i) {
    for (int64_t j = 191; j < 192; ++j) {
      for (int64_t k = 1; k < 191; ++k) {
        y[((i * 192) + j) * 192 + k] = x[((i * 192) + j) * 192 + k];
      }
    }
  }
  for (int64_t i = 0; i < 1; ++i) {
    for (int64_t j = 1; j < 191; ++j) {
      for (int64_t k = 1; k < 191; ++k) {
        y[((i * 192) + j) * 192 + k] = x[((i * 192) + j) * 192 + k];
      }
    }
  }
  for (int64_t i = 191; i < 192; ++i) {
    for (int64_t j = 1; j < 191; ++j) {
      for (int64_t k = 1; k < 191; ++k) {
        y[((i * 192) + j) * 192 + k] = x[((i * 192) + j) * 192 + k];
      }
    }
  }
  
  return ::k0(&in0_buf, &out0_buf);
}

int k0_interior(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_interior(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_interior(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_face_zlo(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_face_zlo(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_face_zlo(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_face_zhi(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_face_zhi(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_face_zhi(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_face_ylo(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_face_ylo(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_face_ylo(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_face_yhi(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_face_yhi(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_face_yhi(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_face_xlo(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_face_xlo(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_face_xlo(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

int k0_face_xhi(int32_t* x, int32_t* y) {
  halide_buffer_t in0_buf;
  init_k0_in0_face_xhi(in0_buf, x);
  halide_buffer_t out0_buf;
  init_k0_out0_face_xhi(out0_buf, y);
  return ::k0(&in0_buf, &out0_buf);
}

} // namespace neptunecc
