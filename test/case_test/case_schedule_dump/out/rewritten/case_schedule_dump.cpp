#include <cstdint>
#include <cstdio>

#include "neptunecc_kernels.h"
void schedule_kernel(int32_t x[8][8], int32_t y[8][8]) {
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

  schedule_kernel(x, y);
  std::printf("y11=%d y66=%d\n", y[1][1], y[6][6]);
  return 0;
}
