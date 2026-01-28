#include <cstdint>
#include <cstdio>

void case3_kernel(int32_t x[8][8][8], int32_t y[8][8][8]) {
#pragma neptune kernel begin tag(k0) name(k0) dm(da) in(x:ghosted) out(y:owned)
  {
    for (int k = 1; k < 7; ++k) {
      for (int j = 1; j < 7; ++j) {
        for (int i = 1; i < 7; ++i) {
          y[k][j][i] = x[k][j][i] + x[k][j][i - 1] + x[k][j][i + 1] +
                       x[k][j - 1][i] + x[k][j + 1][i] + x[k - 1][j][i] +
                       x[k + 1][j][i];
        }
      }
    }
  }
#pragma neptune kernel end tag(k0)
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
