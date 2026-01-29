#include <cstdint>
#include <cstdio>
#include <vector>

#define N 1024

void case_boundary1d_kernel(int32_t x[N], int32_t y[N]) {
#pragma neptune kernel begin tag(k0) name(k0) dm(da) in(x:ghosted) out(y:owned)
  {
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == N - 1) {
        y[i] = x[i];
      } else {
        y[i] = x[i - 1] + x[i] + x[i + 1];
      }
    }
  }
#pragma neptune kernel end tag(k0)
}

static void init_inputs(int32_t *x) {
  for (int i = 0; i < N; ++i) {
    x[i] = i * 3 + 1;
  }
}

int main() {
  std::vector<int32_t> x(N);
  std::vector<int32_t> y(N);
  init_inputs(x.data());
  for (int i = 0; i < N; ++i) {
    y[i] = 0;
  }

  const int iters = 1000;
  int64_t sum = 0;
  for (int iter = 0; iter < iters; ++iter) {
    case_boundary1d_kernel(x.data(), y.data());
    sum += y[0] + y[1] + y[N - 2] + y[N - 1];
  }

  std::printf("sum=%lld\n", static_cast<long long>(sum));
  std::printf("y0=%d y1=%d yend=%d ylast=%d\n", y[0], y[1], y[N - 2],
              y[N - 1]);
  return 0;
}
