#pragma once

#if defined(__clang__) || defined(__GNUC__)
#define NEPTUNECC_WEAK __attribute__((weak))
#else
#define NEPTUNECC_WEAK
#endif

namespace neptunecc {

using NeptuneDMHandle = void *;

struct HaloToken {
  void *impl0;
  void *impl1;
};

// Default no-op implementation (weak); users may provide strong definitions
// in their own translation units to integrate MPI/PETSc halo exchange.
NEPTUNECC_WEAK HaloToken halo_begin(const char *tag, NeptuneDMHandle dm);

// Default no-op implementation (weak); users may provide strong definitions.
NEPTUNECC_WEAK void halo_end(const char *tag, NeptuneDMHandle dm,
                             HaloToken tok);

} // namespace neptunecc
