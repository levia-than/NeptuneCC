#include <cstdint>
#include <cstdio>

static void halo_begin_call() {}
static void halo_end_call() {}

constexpr int N = 128;

void comp_heavy_kernel(int32_t x[N][N], int32_t y[N][N], void *da) {
#pragma neptune overlap begin tag(o0) halo(h0) kernel(k0) policy(auto)
  {
#pragma neptune halo begin tag(h0) dm(da) field(x) kind(global_to_local_begin)
    halo_begin_call();
#pragma neptune kernel begin tag(k0) dm(da) in(x:ghosted) out(y:owned)
    {
      for (int i = 1; i < 127; ++i) {
        for (int j = 1; j < 127; ++j) {
          y[i][j] = x[i][j] + x[i - 1][j] + x[i + 1][j] + x[i][j - 1] +
                    x[i][j + 1];
        }
      }
    }
#pragma neptune kernel end tag(k0)
#pragma neptune halo end tag(h0) kind(global_to_local_end)
    halo_end_call();
  }
#pragma neptune overlap end tag(o0)

  for (int i = 0; i < N; ++i) {
    y[i][0] = x[i][0];
    y[i][N - 1] = x[i][N - 1];
  }
  for (int j = 0; j < N; ++j) {
    y[0][j] = x[0][j];
    y[N - 1][j] = x[N - 1][j];
  }
}

int main() {
  static int32_t x[N][N];
  static int32_t y[N][N];

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      x[i][j] = static_cast<int32_t>((i * 3 + j * 5) & 0xff);
      y[i][j] = 0;
    }
  }

  void *da = reinterpret_cast<void *>(0x1);
  comp_heavy_kernel(x, y, da);

  long long sum = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      sum += y[i][j];
    }
  }

  std::printf("sum=%lld\n", sum);
  return 0;
}
