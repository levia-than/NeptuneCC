#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#define H 2048
#define W 2048

void case_big2d_kernel(int32_t x[H][W], int32_t y[H][W]) {
#pragma neptune kernel begin tag(k0) name(k0) dm(da) in(x:ghosted) out(y:owned)
  {
    for (int i = 1; i < H - 1; ++i) {
      for (int j = 1; j < W - 1; ++j) {
        y[i][j] = x[i][j] + x[i - 1][j] + x[i + 1][j] + x[i][j - 1] +
                  x[i][j + 1];
      }
    }
  }
#pragma neptune kernel end tag(k0)
}

static void init_inputs(int32_t *x) {
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      x[i * W + j] = i * 3 + j;
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

  const int iters = 2000;
  int64_t sum = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iters; ++iter) {
    case_big2d_kernel(x2, y2);
    sum += y2[1][1] + y2[H / 2][W / 2] + y2[H - 2][W - 2];
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
  std::fprintf(stderr, "elapsed_ms=%lld\n", static_cast<long long>(ms));

  std::printf("sum=%lld\n", static_cast<long long>(sum));
  std::printf("y11=%d ymid=%d yend=%d\n", y2[1][1], y2[H / 2][W / 2],
              y2[H - 2][W - 2]);
  return 0;
}
