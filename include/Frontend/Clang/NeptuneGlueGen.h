// Glue generation interfaces.
#pragma once

#include "Frontend/Clang/NeptuneEvents.h"
#include "llvm/ADT/StringRef.h"

namespace neptune {

bool writeGlue(const EventDB &db, llvm::StringRef outDir);
bool rewriteKernelSources(const EventDB &db, llvm::StringRef outDir);
bool writeHalideHelper(llvm::StringRef outDir);
bool writeHalideGenerators(const EventDB &db, llvm::StringRef outDir,
                           bool emitEmitcMLIR, bool emitHalideCpp);

} // namespace neptune
// Declares glue/rewriter generation for Halide AOT integration.
