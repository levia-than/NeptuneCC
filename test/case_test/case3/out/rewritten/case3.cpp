#include <cstdint>
#include <cstdio>

#include "neptunecc_kernels.h"
void case3_kernel(int32_t x[8][8][8], int32_t y[8][8][8]) {
neptunecc::k0(&x[0][0][0], &y[0][0][0]);
}

static void init_inputs(int32_t x[8][8][8]) {
  for (int k = 0; k < 8; ++k) {
    for (int j = 0; j < 8; ++j) {
      for (int i = 0; i < 8; ++i) {
        x[k][j][i] = k * 100 + j * 10 + i;
      }
    }
  }
}

int main() {
  int32_t x[8][8][8];
  int32_t y[8][8][8];
  init_inputs(x);
  for (int k = 0; k < 8; ++k) {
    for (int j = 0; j < 8; ++j) {
      for (int i = 0; i < 8; ++i) {
        y[k][j][i] = 0;
      }
    }
  }

  case3_kernel(x, y);

  int64_t sum = 0;
  sum += y[1][1][1] + y[6][6][6] + y[3][4][5];
  std::printf("sum=%lld\n", static_cast<long long>(sum));
  std::printf("y111=%d y666=%d y345=%d\n", y[1][1][1], y[6][6][6], y[3][4][5]);
  return 0;
}
