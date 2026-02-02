#include <cstdint>
#include <cstdio>
#include <vector>

#include "neptunecc_kernels.h"
#define D 192

void case_boundary3d_kernel(int32_t x[D][D][D], int32_t y[D][D][D]) {
neptunecc::k0(&x[0][0][0], &y[0][0][0]);
}

static void init_inputs(int32_t *x) {
  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < D; ++j) {
      for (int k = 0; k < D; ++k) {
        x[(i * D + j) * D + k] = i * 7 + j * 3 + k;
      }
    }
  }
}

int main() {
  std::vector<int32_t> x(D * D * D);
  std::vector<int32_t> y(D * D * D);
  init_inputs(x.data());
  for (int i = 0; i < D * D * D; ++i) {
    y[i] = 0;
  }

  auto *x3 = reinterpret_cast<int32_t (*)[D][D]>(x.data());
  auto *y3 = reinterpret_cast<int32_t (*)[D][D]>(y.data());

  const int iters = 400;
  int64_t sum = 0;
  for (int iter = 0; iter < iters; ++iter) {
    case_boundary3d_kernel(x3, y3);
    sum += y3[0][0][0] + y3[1][1][1] + y3[D - 2][D - 2][D - 2] +
           y3[D - 1][D - 1][D - 1];
  }

  std::printf("sum=%lld\n", static_cast<long long>(sum));
  std::printf("y000=%d y111=%d yend=%d ylast=%d\n", y3[0][0][0], y3[1][1][1],
              y3[D - 2][D - 2][D - 2], y3[D - 1][D - 1][D - 1]);
  return 0;
}
