#include <cstdint>
#include <cstdio>
#include <vector>

#define H 128
#define W 128

void case_boundary_multi2d_kernel(int32_t x[H][W], int32_t y[H][W],
                                  int32_t a[H][W], int32_t b[H][W]) {
#pragma neptune kernel begin tag(k0) name(k0) dm(da) in(x:ghosted, a:ghosted) out(y:owned, b:owned)
  {
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        b[i][j] = a[i][j];
      }
    }
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

static void init_inputs(int32_t *x, int32_t *a) {
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      x[i * W + j] = i * 5 + j * 7;
      a[i * W + j] = i * 11 + j;
    }
  }
}

int main() {
  std::vector<int32_t> x(H * W);
  std::vector<int32_t> y(H * W);
  std::vector<int32_t> a(H * W);
  std::vector<int32_t> b(H * W);
  init_inputs(x.data(), a.data());
  for (int i = 0; i < H * W; ++i) {
    y[i] = 0;
    b[i] = 0;
  }

  auto *x2 = reinterpret_cast<int32_t (*)[W]>(x.data());
  auto *y2 = reinterpret_cast<int32_t (*)[W]>(y.data());
  auto *a2 = reinterpret_cast<int32_t (*)[W]>(a.data());
  auto *b2 = reinterpret_cast<int32_t (*)[W]>(b.data());

  const int iters = 200;
  int64_t sum = 0;
  for (int iter = 0; iter < iters; ++iter) {
    case_boundary_multi2d_kernel(x2, y2, a2, b2);
    sum += y2[0][0] + y2[1][1] + y2[H - 2][W - 2] + y2[H - 1][W - 1];
    sum += b2[0][0] + b2[H - 1][W - 1];
  }

  std::printf("sum=%lld\n", static_cast<long long>(sum));
  std::printf("y00=%d y11=%d yend=%d ylast=%d\n", y2[0][0], y2[1][1],
              y2[H - 2][W - 2], y2[H - 1][W - 1]);
  std::printf("b00=%d blast=%d\n", b2[0][0], b2[H - 1][W - 1]);
  return 0;
}
