#include <cstdint>
#include <cstdio>

void schedule_kernel(int32_t x[8][8], int32_t y[8][8]) {
#pragma neptune kernel begin tag(k0) name(k0) dm(da) in(x:ghosted) out(y:owned)
  {
    for (int i = 1; i < 7; ++i) {
      for (int j = 1; j < 7; ++j) {
        y[i][j] = x[i][j] + x[i - 1][j] + x[i + 1][j] + x[i][j - 1] +
                  x[i][j + 1];
      }
    }
  }
#pragma neptune kernel end tag(k0)
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
