#pragma once

#include "Frontend/Clang/NeptuneEvents.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"

namespace neptune {

void pairKernels(EventDB &db, clang::DiagnosticsEngine &DE);
void bindKernelsToBlocks(EventDB &db, clang::ASTContext &Ctx);

} // namespace neptune
