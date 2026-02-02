#include <cstdint>
#include <cstdio>

#include "neptunecc_kernels.h"
void case_kernel(int32_t x[8][8], int32_t y[8][8]) {
neptunecc::k0(&x[0][0], &y[0][0]);
}

static void init_inputs(int32_t x[8][8]) {
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      x[i][j] = i * 10 + j;
    }
  }
}

int main() {
  int32_t x[8][8];
  int32_t y[8][8];
  init_inputs(x);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      y[i][j] = 0;
    }
  }

  const int iters = 5000000;
  int64_t sum = 0;
  for (int iter = 0; iter < iters; ++iter) {
    case_kernel(x, y);
    sum += y[1][1] + y[6][6] + y[3][4];
  }

  std::printf("sum=%lld\n", static_cast<long long>(sum));
  std::printf("y11=%d y66=%d y34=%d\n", y[1][1], y[6][6], y[3][4]);
  return 0;
}
