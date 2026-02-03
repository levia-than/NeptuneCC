// Frontend MLIR generation interfaces.
#pragma once

#include "Frontend/Clang/NeptuneEvents.h"

namespace clang {
class ASTContext;
}

namespace neptune {

void lowerKernelsToMLIR(EventDB &localDb, clang::ASTContext &Ctx,
                        EventDB &outDb);

} // namespace neptune
