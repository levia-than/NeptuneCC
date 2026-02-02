#include "Frontend/Clang/NeptunePragma.h"

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/StringRef.h"

using clang::Preprocessor;
using clang::PragmaIntroducer;
using clang::SourceManager;
using clang::Token;

namespace neptune {

NeptunePragmaHandler::NeptunePragmaHandler(EventDB &db, SourceManager &SM)
    : clang::PragmaHandler("kernel"), db(db), SM(SM) {}

static void consumeToEOD(Preprocessor &PP) {
  Token Tok;
  do {
    PP.Lex(Tok);
  } while (!Tok.is(clang::tok::eod));
}

void NeptunePragmaHandler::HandlePragma(Preprocessor &PP,
                                        PragmaIntroducer Introducer,
                                        Token &FirstToken) {
  (void)Introducer;

  if (!FirstToken.is(clang::tok::identifier)) {
    consumeToEOD(PP);
    return;
  }

  clang::IdentifierInfo *firstIdent = FirstToken.getIdentifierInfo();
  if (!firstIdent) {
    consumeToEOD(PP);
    return;
  }
  llvm::StringRef pragmaKind = firstIdent->getName();
  if (pragmaKind != "kernel" && pragmaKind != "halo" &&
      pragmaKind != "overlap") {
    consumeToEOD(PP);
    return;
  }

  Token KindTok;
  PP.Lex(KindTok);
  if (!KindTok.is(clang::tok::identifier) || !KindTok.getIdentifierInfo()) {
    consumeToEOD(PP);
    return;
  }

  llvm::StringRef kind = KindTok.getIdentifierInfo()->getName();
  Event event;
  if (kind == "begin") {
    if (pragmaKind == "kernel") {
      event.kind = EventKind::KernelBegin;
    } else if (pragmaKind == "halo") {
      event.kind = EventKind::HaloBegin;
    } else {
      event.kind = EventKind::OverlapBegin;
    }
  } else if (kind == "end") {
    if (pragmaKind == "kernel") {
      event.kind = EventKind::KernelEnd;
    } else if (pragmaKind == "halo") {
      event.kind = EventKind::HaloEnd;
    } else {
      event.kind = EventKind::OverlapEnd;
    }
  } else {
    consumeToEOD(PP);
    return;
  }

  event.loc = FirstToken.getLocation();
  event.fileOffset = SM.getFileOffset(SM.getExpansionLoc(event.loc));
  event.filePath = SM.getFilename(SM.getExpansionLoc(event.loc));

  bool reachedEOD = false;
  while (!reachedEOD) {
    Token Tok;
    PP.Lex(Tok);
    if (Tok.is(clang::tok::eod)) {
      break;
    }
    if (!Tok.is(clang::tok::identifier)) {
      continue;
    }

    llvm::StringRef key = Tok.getIdentifierInfo()->getName();
    Token LParen;
    PP.Lex(LParen);
    if (LParen.is(clang::tok::eod)) {
      break;
    }
    if (!LParen.is(clang::tok::l_paren)) {
      continue;
    }

    llvm::SmallString<256> val;
    bool sawRParen = false;
    while (true) {
      Token ValTok;
      PP.Lex(ValTok);
      if (ValTok.is(clang::tok::eod)) {
        reachedEOD = true;
        break;
      }
      if (ValTok.is(clang::tok::r_paren)) {
        sawRParen = true;
        break;
      }
      val.append(PP.getSpelling(ValTok));
    }

    if (!sawRParen) {
      continue;
    }

    if (key == "tag") {
      event.tag = val;
    } else if (key == "name") {
      event.name = val;
    } else {
      ClauseKV kv;
      kv.key = key;
      kv.val = val;
      event.clauses.push_back(kv);
    }
  }

  db.events.push_back(event);
}

void registerNeptunePragmas(Preprocessor &PP, EventDB &db,
                            SourceManager &SM) {
  auto *handler = new NeptunePragmaHandler(db, SM);
  PP.AddPragmaHandler("neptune", handler);
}

} // namespace neptune
