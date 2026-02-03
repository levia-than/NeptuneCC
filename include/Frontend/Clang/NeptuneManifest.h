// Manifest emission interfaces.
#pragma once

#include "Frontend/Clang/NeptuneEvents.h"
#include "llvm/ADT/StringRef.h"

namespace neptune {

bool writeManifest(const EventDB &db, llvm::StringRef outDir,
                   bool emitKernelsMLIR = false);

} // namespace neptune
