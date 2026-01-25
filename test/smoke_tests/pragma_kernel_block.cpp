// 更真实的“host + kernel block + epilogue”结构
void pragma_kernel_block_smoke(int a[8], int b[8],
                               int x[4][4], int y[4][4]) {
  // --- host prologue: 正常业务代码（不进 kernel） ---
  int checksum = 0;
  for (int i = 0; i < 8; ++i) checksum += a[i];

#pragma neptune kernel begin tag(k0) name(k0) dm(da) \
  in(a:ghosted, x:ghosted) out(b:owned, y:owned)
  {
    // (A) 1D map：b[i] = a[i] + 1
    for (int i = 0; i < 8; ++i) {
      b[i] = a[i] + 1;
    }

    // (B) 2D 5-point stencil（interior），不写 if，直接收缩边界范围
    for (int i = 1; i < 3; ++i) {
      for (int j = 1; j < 3; ++j) {
        y[i][j] = x[i][j] + x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1];
      }
    }

    // (C) boundary handling：用“独立 loop”表达边界（依旧无 if）
    for (int i = 0; i < 4; ++i) {
      y[i][0] = x[i][0];
      y[i][3] = x[i][3];
    }
    for (int j = 0; j < 4; ++j) {
      y[0][j] = x[0][j];
      y[3][j] = x[3][j];
    }
  }
#pragma neptune kernel end tag(k0)

  // --- host epilogue: 用输出做点事（不进 kernel） ---
  if (checksum == 0) b[0] = 0;
}
