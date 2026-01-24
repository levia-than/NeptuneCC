void pragma_kernel_block_smoke() {
#pragma neptune kernel begin tag(k0) name(k0) dm(da) in(x_local:ghosted) out(y:owned)
  {
    int a = 0;
    a += 1;
  }
#pragma neptune kernel end tag(k0)
}
