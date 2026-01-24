#pragma once

#include "Frontend/Clang/NeptuneEvents.h"
#include "clang/Lex/Pragma.h"

namespace clang {
class Preprocessor;
class SourceManager;
} // namespace clang

namespace neptune {

class NeptunePragmaHandler : public clang::PragmaHandler {
public:
  NeptunePragmaHandler(EventDB &db, clang::SourceManager &SM);

  void HandlePragma(clang::Preprocessor &PP,
                    clang::PragmaIntroducer Introducer,
                    clang::Token &FirstToken) override;

private:
  EventDB &db;
  clang::SourceManager &SM;
};

void registerNeptunePragmas(clang::Preprocessor &PP, EventDB &db,
                            clang::SourceManager &SM);

} // namespace neptune
