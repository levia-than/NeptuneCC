#include "Frontend/Clang/NeptuneManifest.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>
#include <utility>

namespace neptune {

bool writeManifest(const EventDB &db, llvm::StringRef outDir) {
  llvm::SmallString<256> outputDir(outDir);
  std::error_code ec = llvm::sys::fs::create_directories(outputDir);
  if (ec) {
    llvm::errs() << "neptune-cc: failed to create output dir '"
                 << outputDir << "': " << ec.message() << "\n";
    return false;
  }

  llvm::SmallString<256> outputPath(outputDir);
  llvm::sys::path::append(outputPath, "manifest.json");

  std::error_code fileEC;
  llvm::raw_fd_ostream os(outputPath, fileEC, llvm::sys::fs::OF_Text);
  if (fileEC) {
    llvm::errs() << "neptune-cc: failed to open manifest '"
                 << outputPath << "': " << fileEC.message() << "\n";
    return false;
  }

  llvm::json::Array kernels;
  for (const auto &kernel : db.kernels) {
    if (!kernel.blockBegin.isValid() || !kernel.blockEnd.isValid()) {
      continue;
    }

    llvm::json::Object clauses;
    for (const auto &clause : kernel.begin.clauses) {
      clauses[clause.key] = clause.val;
    }

    llvm::StringRef name =
        kernel.begin.name.empty() ? kernel.begin.tag : kernel.begin.name;

    llvm::json::Object obj;
    obj["tag"] = kernel.begin.tag;
    obj["name"] = name;
    obj["pragma_begin_offset"] =
        static_cast<uint64_t>(kernel.begin.fileOffset);
    obj["pragma_end_offset"] = static_cast<uint64_t>(kernel.end.fileOffset);
    obj["block_begin_offset"] =
        static_cast<uint64_t>(kernel.blockBeginOffset);
    obj["block_end_offset"] = static_cast<uint64_t>(kernel.blockEndOffset);
    obj["clauses"] = std::move(clauses);
    kernels.push_back(std::move(obj));
  }

  llvm::json::Object root;
  root["version"] = 1;
  root["kernels"] = std::move(kernels);
  os << llvm::json::Value(std::move(root)) << "\n";
  return true;
}

} // namespace neptune
