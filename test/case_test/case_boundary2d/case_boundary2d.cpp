#include <cstdint>
#include <cstdio>
#include <vector>

#define H 1024
#define W 1024

void case_boundary2d_kernel(int32_t x[H][W], int32_t y[H][W]) {
#pragma neptune kernel begin tag(k0) name(k0) dm(da) in(x:ghosted) out(y:owned)
  {
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        if (i == 0 || j == 0 || i == H - 1 || j == W - 1) {
          y[i][j] = x[i][j];
        } else {
          y[i][j] = x[i][j] + x[i - 1][j] + x[i + 1][j] + x[i][j - 1] +
                    x[i][j + 1];
        }
      }
    }
  }
#pragma neptune kernel end tag(k0)
}

static void init_inputs(int32_t *x) {
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      x[i * W + j] = i * 7 + j * 3;
    }
  }
}

int main() {
  std::vector<int32_t> x(H * W);
  std::vector<int32_t> y(H * W);
  init_inputs(x.data());
  for (int i = 0; i < H * W; ++i) {
    y[i] = 0;
  }

  auto *x2 = reinterpret_cast<int32_t (*)[W]>(x.data());
  auto *y2 = reinterpret_cast<int32_t (*)[W]>(y.data());

  const int iters = 50;
  int64_t sum = 0;
  for (int iter = 0; iter < iters; ++iter) {
    case_boundary2d_kernel(x2, y2);
    sum += y2[0][0] + y2[1][1] + y2[H - 2][W - 2] + y2[H - 1][W - 1];
  }

  std::printf("sum=%lld\n", static_cast<long long>(sum));
  std::printf("y00=%d y11=%d yend=%d ylast=%d\n", y2[0][0], y2[1][1],
              y2[H - 2][W - 2], y2[H - 1][W - 1]);
  return 0;
}
