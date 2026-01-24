#include "Frontend/Clang/NeptuneBinder.h"
#include "Frontend/Clang/NeptuneManifest.h"
#include "Frontend/Clang/NeptunePragma.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <string>

namespace {

llvm::cl::OptionCategory NeptuneCategory("neptune-cc options");

llvm::cl::opt<std::string> OutDirOpt(
    "out-dir", llvm::cl::desc("Output directory for manifest.json"),
    llvm::cl::init("neptune_out"), llvm::cl::cat(NeptuneCategory));

class NeptuneASTConsumer final : public clang::ASTConsumer {
public:
  NeptuneASTConsumer(neptune::EventDB &localDb, neptune::EventDB &outDb)
      : localDb(localDb), outDb(outDb) {}

  void HandleTranslationUnit(clang::ASTContext &Ctx) override {
    neptune::pairKernels(localDb, Ctx.getDiagnostics());
    neptune::bindKernelsToBlocks(localDb, Ctx);
    outDb.kernels.append(localDb.kernels.begin(), localDb.kernels.end());
  }

private:
  neptune::EventDB &localDb;
  neptune::EventDB &outDb;
};

class NeptuneFrontendAction final : public clang::ASTFrontendAction {
public:
  explicit NeptuneFrontendAction(neptune::EventDB &outDb) : outDb(outDb) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    llvm::StringRef InFile) override {
    (void)InFile;
    neptune::registerNeptunePragmas(CI.getPreprocessor(), localDb,
                                    CI.getSourceManager());
    return std::make_unique<NeptuneASTConsumer>(localDb, outDb);
  }

private:
  neptune::EventDB &outDb;
  neptune::EventDB localDb;
};

class NeptuneFrontendActionFactory
    : public clang::tooling::FrontendActionFactory {
public:
  explicit NeptuneFrontendActionFactory(neptune::EventDB &outDb)
      : outDb(outDb) {}

  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<NeptuneFrontendAction>(outDb);
  }

private:
  neptune::EventDB &outDb;
};

static bool hasTargetArg(const clang::tooling::CommandLineArguments &args) {
  for (const auto &arg : args) {
    llvm::StringRef argRef(arg);
    if (argRef == "-target" || argRef == "--target" || argRef == "-triple") {
      return true;
    }
    if (argRef.starts_with("-target=") || argRef.starts_with("--target=") ||
        argRef.starts_with("-triple=")) {
      return true;
    }
  }
  return false;
}

static std::string defaultTargetTriple() {
  std::string triple = llvm::sys::getDefaultTargetTriple();
  llvm::StringRef tripleRef(triple);
  if (!tripleRef.empty() && !tripleRef.starts_with("unknown")) {
    return triple;
  }
#if defined(__x86_64__)
  return "x86_64-unknown-linux-gnu";
#elif defined(__aarch64__)
  return "aarch64-unknown-linux-gnu";
#elif defined(__arm__)
  return "armv7-unknown-linux-gnueabihf";
#else
  return "x86_64-unknown-linux-gnu";
#endif
}

} // namespace

int main(int argc, const char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  auto expectedParser = clang::tooling::CommonOptionsParser::create(
      argc, argv, NeptuneCategory);
  if (!expectedParser) {
    llvm::errs() << llvm::toString(expectedParser.takeError()) << "\n";
    return 1;
  }

  clang::tooling::CommonOptionsParser &optionsParser =
      expectedParser.get();
  clang::tooling::ClangTool tool(optionsParser.getCompilations(),
                                 optionsParser.getSourcePathList());

  std::string triple = defaultTargetTriple();
  tool.appendArgumentsAdjuster(
      [triple](const clang::tooling::CommandLineArguments &args,
               llvm::StringRef filename) {
        (void)filename;
        if (hasTargetArg(args)) {
          return args;
        }
        clang::tooling::CommandLineArguments adjusted(args);
        auto insertIt = adjusted.begin();
        if (insertIt != adjusted.end()) {
          ++insertIt; // keep argv0 as the first argument
        }
        adjusted.insert(insertIt, {"-target", triple});
        return adjusted;
      });

  neptune::EventDB outDb;
  NeptuneFrontendActionFactory factory(outDb);
  int result = tool.run(&factory);

  llvm::SmallString<256> outDir(OutDirOpt);
  if (!neptune::writeManifest(outDb, outDir)) {
    return result == 0 ? 1 : result;
  }

  return result;
}
