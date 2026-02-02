#include "Runtime/NeptuneHaloRuntime.h"

#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

namespace neptunecc {

static bool shouldLogHalo() {
  if (auto val = llvm::sys::Process::GetEnv("NEPTUNECC_LOG_HALO")) {
    return !val->empty() && *val != "0";
  }
  return false;
}

HaloToken halo_begin(const char *tag, NeptuneDMHandle dm) {
  if (shouldLogHalo()) {
    llvm::errs() << "neptune-cc: halo_begin tag=" << (tag ? tag : "")
                 << " dm=" << dm << "\n";
  }
  HaloToken tok{nullptr, nullptr};
  return tok;
}

void halo_end(const char *tag, NeptuneDMHandle dm, HaloToken tok) {
  (void)tok;
  if (shouldLogHalo()) {
    llvm::errs() << "neptune-cc: halo_end tag=" << (tag ? tag : "")
                 << " dm=" << dm << "\n";
  }
}

} // namespace neptunecc
