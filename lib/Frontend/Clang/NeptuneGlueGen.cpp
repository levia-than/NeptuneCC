#include "Frontend/Clang/NeptuneGlueGen.h"

#include "Dialect/NeptuneIR/NeptuneIRAttrs.h"
#include "Dialect/NeptuneIR/NeptuneIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "Passes/NeptuneIRPasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <optional>
#include <string>

namespace neptune {
namespace {

struct PortInfo {
  std::string name;
  bool isInput = false;
  bool isGhosted = false;
  unsigned roleIndex = 0;
  unsigned argIndex = 0;
  unsigned rank = 0;
  llvm::SmallVector<int64_t, 4> shape;
};

struct KernelInfo {
  std::string tag;
  llvm::SmallVector<PortInfo, 8> ports;
  unsigned numInputs = 0;
  unsigned numOutputs = 0;
  bool halideCompatible = false;
  bool hasOutputBounds = false;
  bool boundsValid = false;
  llvm::SmallVector<int64_t, 4> outMin;
  llvm::SmallVector<int64_t, 4> outExtent;
  llvm::SmallVector<int64_t, 4> radius;
  std::string radiusSource;
  std::string boundsSource;
};

struct Replacement {
  size_t start = 0;
  size_t end = 0;
  std::string text;
};

struct StencilBounds {
  int64_t lbI = 0;
  int64_t ubI = 0;
  int64_t lbJ = 0;
  int64_t ubJ = 0;
  unsigned outputArgIndex = 0;
};

struct ApplyStencilInfo {
  llvm::SmallVector<int64_t, 4> lb;
  llvm::SmallVector<int64_t, 4> ub;
  llvm::SmallVector<int64_t, 4> radius;
  std::string radiusSource;
};

static bool matchConstantIndex(mlir::Value v, int64_t &out) {
  auto cst = v.getDefiningOp<mlir::arith::ConstantOp>();
  if (!cst || !cst.getType().isIndex()) {
    return false;
  }
  auto attr = llvm::dyn_cast<mlir::IntegerAttr>(cst.getValue());
  if (!attr) {
    return false;
  }
  out = attr.getInt();
  return true;
}

static std::optional<StencilBounds> findStencilBounds(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::scf::ForOp, 4> candidates;
  func.walk([&](mlir::scf::ForOp forOp) {
    if (!forOp->getParentOfType<mlir::scf::ForOp>()) {
      candidates.push_back(forOp);
    }
  });

  for (mlir::scf::ForOp outer : candidates) {
    int64_t lbI = 0;
    int64_t ubI = 0;
    int64_t stepI = 0;
    if (!matchConstantIndex(outer.getLowerBound(), lbI) ||
        !matchConstantIndex(outer.getUpperBound(), ubI) ||
        !matchConstantIndex(outer.getStep(), stepI) || stepI != 1) {
      continue;
    }

    mlir::scf::ForOp inner;
    mlir::Block &outerBody = *outer.getBody();
    for (mlir::Operation &op : outerBody.getOperations()) {
      if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
        if (inner) {
          inner = nullptr;
          break;
        }
        inner = forOp;
        continue;
      }
      if (llvm::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                    mlir::arith::ConstantOp, mlir::scf::YieldOp>(op)) {
        continue;
      }
      inner = nullptr;
      break;
    }
    if (!inner) {
      continue;
    }

    int64_t lbJ = 0;
    int64_t ubJ = 0;
    int64_t stepJ = 0;
    if (!matchConstantIndex(inner.getLowerBound(), lbJ) ||
        !matchConstantIndex(inner.getUpperBound(), ubJ) ||
        !matchConstantIndex(inner.getStep(), stepJ) || stepJ != 1) {
      continue;
    }

    mlir::memref::StoreOp store;
    mlir::Block &innerBody = *inner.getBody();
    for (mlir::Operation &op : innerBody.getOperations()) {
      if (auto s = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
        if (store) {
          store = nullptr;
          break;
        }
        store = s;
        continue;
      }
      if (llvm::isa<mlir::memref::LoadOp, mlir::arith::AddIOp,
                    mlir::arith::SubIOp, mlir::arith::ConstantOp,
                    mlir::scf::YieldOp>(op)) {
        continue;
      }
      store = nullptr;
      break;
    }
    if (!store) {
      continue;
    }
    if (store.getIndices().size() != 2) {
      continue;
    }
    if (store.getIndices()[0] != outer.getInductionVar() ||
        store.getIndices()[1] != inner.getInductionVar()) {
      continue;
    }

    auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(store.getMemref());
    if (!blockArg || blockArg.getOwner() != &func.getBody().front()) {
      continue;
    }

    StencilBounds bounds;
    bounds.lbI = lbI;
    bounds.ubI = ubI;
    bounds.lbJ = lbJ;
    bounds.ubJ = ubJ;
    bounds.outputArgIndex = blockArg.getArgNumber();
    return bounds;
  }

  return std::nullopt;
}

static std::optional<ApplyStencilInfo>
inferStencilFromApply(mlir::ModuleOp module, mlir::func::FuncOp func) {
  mlir::ModuleOp clone = module.clone();
  auto clonedFunc =
      clone.lookupSymbol<mlir::func::FuncOp>(func.getSymName());
  if (!clonedFunc) {
    return std::nullopt;
  }

  mlir::PassManager pm(clone.getContext());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::Neptune::NeptuneIR::createSCFToNeptuneIRPass());
  if (mlir::failed(pm.run(clone))) {
    return std::nullopt;
  }

  mlir::Neptune::NeptuneIR::ApplyOp apply;
  bool multiple = false;
  clonedFunc.walk([&](mlir::Neptune::NeptuneIR::ApplyOp op) {
    if (apply) {
      multiple = true;
      return;
    }
    apply = op;
  });
  if (!apply || multiple) {
    return std::nullopt;
  }

  auto bounds = apply.getBounds();
  auto lb = bounds.getLb().asArrayRef();
  auto ub = bounds.getUb().asArrayRef();
  if (lb.empty() || lb.size() != ub.size()) {
    return std::nullopt;
  }

  ApplyStencilInfo info;
  info.lb.assign(lb.begin(), lb.end());
  info.ub.assign(ub.begin(), ub.end());

  if (auto radius = apply.getRadius()) {
    info.radius.assign(radius->begin(), radius->end());
    info.radiusSource = "attr";
  } else {
    info.radius.assign(lb.size(), 0);
    bool ok = true;
    bool sawAccess = false;
    apply.walk([&](mlir::Neptune::NeptuneIR::AccessOp acc) {
      sawAccess = true;
      auto offsets = acc.getOffsets();
      if (offsets.size() != info.radius.size()) {
        ok = false;
        return;
      }
      for (size_t i = 0; i < offsets.size(); ++i) {
        int64_t v = offsets[i];
        if (v < 0)
          v = -v;
        if (v > info.radius[i])
          info.radius[i] = v;
      }
    });
    if (!ok || !sawAccess) {
      info.radius.assign(lb.size(), 1);
      info.radiusSource = "default";
    } else {
      info.radiusSource = "derived";
    }
  }

  if (info.radius.size() != info.lb.size()) {
    return std::nullopt;
  }

  return info;
}

static bool validateStencilBounds(llvm::ArrayRef<int64_t> outMin,
                                  llvm::ArrayRef<int64_t> outExtent,
                                  llvm::ArrayRef<int64_t> radius,
                                  llvm::ArrayRef<int64_t> inShape,
                                  llvm::StringRef tag) {
  if (outMin.size() != inShape.size() ||
      outExtent.size() != inShape.size() ||
      radius.size() != inShape.size()) {
    llvm::errs() << "neptune-cc: stencil bounds rank mismatch for kernel '"
                 << tag << "'\n";
    return false;
  }

  for (size_t d = 0; d < inShape.size(); ++d) {
    if (outExtent[d] <= 0) {
      llvm::errs() << "neptune-cc: invalid extent for kernel '" << tag
                   << "'\n";
      return false;
    }
    if (outMin[d] < radius[d]) {
      llvm::errs() << "neptune-cc: outMin < radius for kernel '" << tag
                   << "'\n";
      return false;
    }
    int64_t upper = outMin[d] + outExtent[d] + radius[d];
    if (upper > inShape[d]) {
      llvm::errs() << "neptune-cc: bounds exceed input shape for kernel '"
                   << tag << "'\n";
      return false;
    }
  }
  return true;
}

static void printI64Array(llvm::raw_ostream &os,
                          llvm::ArrayRef<int64_t> values) {
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << "]";
}

static std::string joinPath(llvm::StringRef base,
                            std::initializer_list<llvm::StringRef> parts) {
  llvm::SmallString<256> path(base);
  for (llvm::StringRef part : parts) {
    llvm::sys::path::append(path, part);
  }
  return path.str().str();
}

static bool ensureDir(llvm::StringRef dir, llvm::StringRef label) {
  std::error_code ec = llvm::sys::fs::create_directories(dir);
  if (ec) {
    llvm::errs() << "neptune-cc: failed to create " << label << " dir '"
                 << dir << "': " << ec.message() << "\n";
    return false;
  }
  return true;
}

static bool parsePortMapEntry(llvm::StringRef entry, PortInfo &out,
                              llvm::raw_ostream &errs) {
  auto parts = entry.split('=');
  if (parts.first.empty() || parts.second.empty()) {
    errs << "neptune-cc: invalid port_map entry '" << entry << "'\n";
    return false;
  }

  out.name = parts.first.trim().str();
  llvm::SmallVector<llvm::StringRef, 4> fields;
  parts.second.split(fields, ':');
  if (fields.size() < 2) {
    errs << "neptune-cc: invalid port_map entry '" << entry << "'\n";
    return false;
  }

  llvm::StringRef role = fields[0].trim();
  if (role.starts_with("in")) {
    out.isInput = true;
    llvm::StringRef indexStr = role.drop_front(2);
    if (indexStr.getAsInteger(10, out.roleIndex)) {
      errs << "neptune-cc: invalid port_map role '" << role << "'\n";
      return false;
    }
  } else if (role.starts_with("out")) {
    out.isInput = false;
    llvm::StringRef indexStr = role.drop_front(3);
    if (indexStr.getAsInteger(10, out.roleIndex)) {
      errs << "neptune-cc: invalid port_map role '" << role << "'\n";
      return false;
    }
  } else {
    errs << "neptune-cc: invalid port_map role '" << role << "'\n";
    return false;
  }

  if (fields.size() >= 3) {
    llvm::StringRef storage = fields[1].trim();
    if (storage == "ghosted") {
      out.isGhosted = true;
    }
  }

  llvm::StringRef argPart = fields.back().trim();
  if (!argPart.starts_with("arg")) {
    errs << "neptune-cc: invalid port_map arg '" << argPart << "'\n";
    return false;
  }
  llvm::StringRef argIndexStr = argPart.drop_front(3);
  if (argIndexStr.getAsInteger(10, out.argIndex)) {
    errs << "neptune-cc: invalid port_map arg '" << argPart << "'\n";
    return false;
  }

  return true;
}

static bool collectKernelInfos(const EventDB &db,
                               llvm::SmallVectorImpl<KernelInfo> &out) {
  if (!db.kernelModule) {
    return true;
  }

  llvm::StringSet<> seenTags;
  mlir::ModuleOp module = db.kernelModule.get();
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    auto tagAttr =
        func->getAttrOfType<mlir::StringAttr>("neptunecc.tag");
    if (!tagAttr) {
      continue;
    }
    llvm::StringRef tag = tagAttr.getValue();
    if (tag.empty()) {
      continue;
    }
    if (!seenTags.insert(tag).second) {
      llvm::errs() << "neptune-cc: duplicate kernel tag '" << tag << "'\n";
      return false;
    }

    auto portMapAttr =
        func->getAttrOfType<mlir::ArrayAttr>("neptunecc.port_map");
    if (!portMapAttr) {
      llvm::errs() << "neptune-cc: missing neptunecc.port_map for kernel '"
                   << tag << "'\n";
      return false;
    }

    KernelInfo info;
    info.tag = tag.str();
    for (auto attr : portMapAttr) {
      auto strAttr = llvm::dyn_cast<mlir::StringAttr>(attr);
      if (!strAttr) {
        llvm::errs() << "neptune-cc: invalid port_map entry for kernel '"
                     << tag << "'\n";
        return false;
      }
      PortInfo port;
      if (!parsePortMapEntry(strAttr.getValue(), port, llvm::errs())) {
        return false;
      }
      if (port.argIndex >= func.getNumArguments()) {
        llvm::errs() << "neptune-cc: port_map arg out of range for kernel '"
                     << tag << "'\n";
        return false;
      }
      auto memrefTy = llvm::dyn_cast<mlir::MemRefType>(
          func.getArgument(port.argIndex).getType());
      if (!memrefTy) {
        llvm::errs() << "neptune-cc: expected memref argument for kernel '"
                     << tag << "'\n";
        return false;
      }
      if (!memrefTy.hasStaticShape()) {
        llvm::errs()
            << "neptune-cc: dynamic memref shape not supported for kernel '"
            << tag << "'\n";
        return false;
      }
      auto elemTy = memrefTy.getElementType();
      auto intTy = llvm::dyn_cast<mlir::IntegerType>(elemTy);
      if (!intTy || intTy.getWidth() != 32) {
        llvm::errs() << "neptune-cc: only i32 memrefs supported for kernel '"
                     << tag << "'\n";
        return false;
      }
      port.rank = memrefTy.getRank();
      if (port.rank == 0) {
        llvm::errs() << "neptune-cc: scalar memrefs not supported for kernel '"
                     << tag << "'\n";
        return false;
      }
      port.shape.assign(memrefTy.getShape().begin(),
                        memrefTy.getShape().end());
      if (port.isInput) {
        ++info.numInputs;
      } else {
        ++info.numOutputs;
      }
      info.ports.push_back(std::move(port));
    }

    info.halideCompatible = false;
    const PortInfo *inPort = nullptr;
    const PortInfo *outPort = nullptr;
    if (info.numInputs == 1 && info.numOutputs == 1 && info.ports.size() == 2) {
      for (const auto &port : info.ports) {
        if (port.isInput) {
          inPort = &port;
        } else {
          outPort = &port;
        }
      }
      if (inPort && outPort && inPort->rank == outPort->rank &&
          inPort->rank > 0) {
        info.halideCompatible = true;
      }
    }

    if (info.halideCompatible && inPort && outPort) {
      bool boundsFromApply = false;
      bool boundsFromScf = false;
      llvm::SmallVector<int64_t, 4> radius;
      std::string radiusSource;

      auto applyInfo = inferStencilFromApply(module, func);
      if (applyInfo) {
        if (applyInfo->lb.size() == outPort->rank &&
            applyInfo->ub.size() == outPort->rank) {
          bool ok = true;
          for (size_t d = 0; d < applyInfo->lb.size(); ++d) {
            if (applyInfo->ub[d] <= applyInfo->lb[d]) {
              ok = false;
              break;
            }
          }
          if (ok) {
            info.hasOutputBounds = true;
            info.outMin.assign(applyInfo->lb.begin(), applyInfo->lb.end());
            info.outExtent.assign(outPort->rank, 0);
            for (size_t d = 0; d < outPort->rank; ++d) {
              info.outExtent[d] = applyInfo->ub[d] - applyInfo->lb[d];
            }
            radius.assign(applyInfo->radius.begin(), applyInfo->radius.end());
            radiusSource = applyInfo->radiusSource;
            boundsFromApply = true;
          }
        }
      }

      if (!boundsFromApply && outPort->rank == 2) {
        auto bounds = findStencilBounds(func);
        if (bounds && outPort->argIndex == bounds->outputArgIndex &&
            outPort->shape.size() == 2) {
          int64_t lbI = bounds->lbI;
          int64_t ubI = bounds->ubI;
          int64_t lbJ = bounds->lbJ;
          int64_t ubJ = bounds->ubJ;
          if (lbI >= 0 && lbJ >= 0 && ubI > lbI && ubJ > lbJ) {
            info.hasOutputBounds = true;
            info.outMin.assign({lbI, lbJ});
            info.outExtent.assign({ubI - lbI, ubJ - lbJ});
            radius.assign(outPort->rank, 1);
            radiusSource = "default";
            boundsFromScf = true;
          }
        }
      }

      if (info.hasOutputBounds) {
        if (radius.size() != outPort->rank) {
          llvm::errs()
              << "neptune-cc: missing radius for kernel '" << tag << "'\n";
          info.hasOutputBounds = false;
        }
      }

      if (info.hasOutputBounds) {
        if (!validateStencilBounds(
                info.outMin, info.outExtent, radius, inPort->shape, tag)) {
          info.hasOutputBounds = false;
        } else {
          info.boundsValid = true;
          info.radius = std::move(radius);
          info.radiusSource = radiusSource;
          info.boundsSource = boundsFromApply ? "neptune.ir.apply" : "scf.for";
          if (boundsFromApply) {
            llvm::outs() << "neptune-cc: bounds from neptune.ir.apply for '"
                         << tag << "'\n";
          } else if (boundsFromScf) {
            llvm::outs() << "neptune-cc: fallback bounds from scf.for for '"
                         << tag << "'\n";
          }
        }
      }

      if (!info.boundsValid) {
        llvm::errs()
            << "neptune-cc: no valid stencil bounds for kernel '" << tag
            << "', skipping glue/rewrite\n";
      }
    }

    std::stable_sort(info.ports.begin(), info.ports.end(),
                     [](const PortInfo &a, const PortInfo &b) {
                       if (a.isInput != b.isInput) {
                         return a.isInput && !b.isInput;
                       }
                       if (a.roleIndex != b.roleIndex) {
                         return a.roleIndex < b.roleIndex;
                       }
                       return a.name < b.name;
                     });

    out.push_back(std::move(info));
  }

  return true;
}

static std::string formatArrayArg(llvm::StringRef name, unsigned rank) {
  if (rank <= 1) {
    return name.str();
  }
  std::string out = "&";
  out.append(name.str());
  for (unsigned i = 0; i < rank; ++i) {
    out.append("[0]");
  }
  return out;
}

static std::string buildKernelCall(const KernelInfo &info) {
  std::string call = "neptunecc::";
  call.append(info.tag);
  call.push_back('(');
  for (size_t i = 0; i < info.ports.size(); ++i) {
    if (i > 0) {
      call.append(", ");
    }
    const auto &port = info.ports[i];
    call.append(formatArrayArg(port.name, port.rank));
  }
  call.append(");");
  return call;
}

static size_t findLineStart(llvm::StringRef content, size_t offset) {
  size_t pos = std::min(offset, content.size());
  while (pos > 0 && content[pos - 1] != '\n') {
    --pos;
  }
  return pos;
}

static size_t findLineEnd(llvm::StringRef content, size_t offset) {
  size_t pos = std::min(offset, content.size());
  while (pos < content.size() && content[pos] != '\n') {
    ++pos;
  }
  if (pos < content.size()) {
    ++pos;
  }
  return pos;
}

static std::string detectNewline(llvm::StringRef content) {
  if (content.find("\r\n") != llvm::StringRef::npos) {
    return "\r\n";
  }
  return "\n";
}

static void ensureKernelInclude(std::string &content) {
  llvm::StringRef contentRef(content);
  if (contentRef.find("#include \"neptunecc_kernels.h\"") !=
      llvm::StringRef::npos) {
    return;
  }

  std::string newline = detectNewline(contentRef);
  size_t insertPos = 0;
  if (contentRef.starts_with("\xEF\xBB\xBF")) {
    insertPos = 3;
  }

  bool inBlockComment = false;
  size_t pos = insertPos;
  while (pos < content.size()) {
    size_t lineEnd = content.find('\n', pos);
    if (lineEnd == std::string::npos) {
      lineEnd = content.size();
    }
    llvm::StringRef line(content.data() + pos, lineEnd - pos);
    llvm::StringRef trimmed = line.trim();

    if (inBlockComment) {
      if (trimmed.find("*/") != llvm::StringRef::npos) {
        inBlockComment = false;
      }
      insertPos = (lineEnd < content.size()) ? lineEnd + 1 : lineEnd;
      pos = insertPos;
      continue;
    }

    if (trimmed.empty()) {
      insertPos = (lineEnd < content.size()) ? lineEnd + 1 : lineEnd;
      pos = insertPos;
      continue;
    }

    if (trimmed.starts_with("/*")) {
      if (trimmed.find("*/") == llvm::StringRef::npos) {
        inBlockComment = true;
      }
      insertPos = (lineEnd < content.size()) ? lineEnd + 1 : lineEnd;
      pos = insertPos;
      continue;
    }

    if (trimmed.starts_with("//")) {
      insertPos = (lineEnd < content.size()) ? lineEnd + 1 : lineEnd;
      pos = insertPos;
      continue;
    }

    if (trimmed.starts_with("#include")) {
      insertPos = (lineEnd < content.size()) ? lineEnd + 1 : lineEnd;
      pos = insertPos;
      continue;
    }

    break;
  }

  std::string includeLine =
      std::string("#include \"neptunecc_kernels.h\"") + newline;
  content.insert(insertPos, includeLine);
}

static bool writeGlueHeader(llvm::StringRef glueDir,
                            llvm::ArrayRef<KernelInfo> kernels) {
  std::string headerPath = joinPath(glueDir, {"neptunecc_kernels.h"});
  std::ofstream os(headerPath);
  if (!os.is_open()) {
    llvm::errs() << "neptune-cc: failed to open glue header '"
                 << headerPath << "'\n";
    return false;
  }

  os << "#pragma once\n";
  os << "#include <cstdint>\n";
  os << "namespace neptunecc {\n";
  for (const auto &kernel : kernels) {
    if (!kernel.halideCompatible || !kernel.boundsValid) {
      if (!kernel.halideCompatible) {
        llvm::errs() << "neptune-cc: skipping glue header for kernel '"
                     << kernel.tag
                     << "' (requires single input/output with matching rank)\n";
      } else {
        llvm::errs() << "neptune-cc: skipping glue header for kernel '"
                     << kernel.tag << "' (no valid stencil bounds)\n";
      }
      continue;
    }
    os << "int " << kernel.tag << "(";
    for (size_t i = 0; i < kernel.ports.size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      os << "int32_t* " << kernel.ports[i].name;
    }
    os << ");\n";
  }
  os << "} // namespace neptunecc\n";
  return true;
}

static bool writeGlueCpp(llvm::StringRef glueDir,
                         llvm::ArrayRef<KernelInfo> kernels) {
  std::string cppPath = joinPath(glueDir, {"neptunecc_kernels.cpp"});
  std::ofstream os(cppPath);
  if (!os.is_open()) {
    llvm::errs() << "neptune-cc: failed to open glue source '"
                 << cppPath << "'\n";
    return false;
  }

  os << "#include \"neptunecc_kernels.h\"\n";
  os << "#include \"HalideRuntime.h\"\n";
  for (const auto &kernel : kernels) {
    if (!kernel.halideCompatible || !kernel.boundsValid) {
      continue;
    }
    os << "#include \"../halide/" << kernel.tag << ".h\"\n";
  }
  os << "\nnamespace neptunecc {\n\n";

  os << "static inline halide_type_t i32_type() {\n";
  os << "  halide_type_t t;\n";
  os << "  t.code = halide_type_int;\n";
  os << "  t.bits = 32;\n";
  os << "  t.lanes = 1;\n";
  os << "  return t;\n";
  os << "}\n\n";
  os << "static const halide_type_t k_i32_type = i32_type();\n\n";
  os << "// dim0 corresponds to the last IR index (fastest varying),\n";
  os << "// dim1 corresponds to the first IR index. offset_elems uses\n";
  os << "// row-major: offset = sum(outMin[d] * stride[d]).\n\n";
  const char *ghostedEnv = std::getenv("NEPTUNECC_GHOSTED_BASE");
  const bool allowGhostedPolicy =
      ghostedEnv && ghostedEnv[0] != '\0' && ghostedEnv[0] != '0';
  if (allowGhostedPolicy) {
    llvm::outs() << "neptune-cc: ghosted input policy enabled (P1)\n";
  }

  for (const auto &kernel : kernels) {
    if (!kernel.halideCompatible || !kernel.boundsValid) {
      continue;
    }
    const PortInfo *inPort = nullptr;
    const PortInfo *outPort = nullptr;
    for (const auto &port : kernel.ports) {
      if (port.isInput) {
        inPort = &port;
      } else {
        outPort = &port;
      }
    }
    if (!inPort || !outPort) {
      continue;
    }
    size_t rank = inPort->rank;
    if (kernel.outMin.size() != rank || kernel.outExtent.size() != rank ||
        kernel.radius.size() != rank) {
      llvm::errs() << "neptune-cc: mismatched stencil metadata for kernel '"
                   << kernel.tag << "'\n";
      continue;
    }
    llvm::outs() << "neptune-cc: glue kernel '" << kernel.tag << "'\n";
    llvm::outs() << "neptune-cc:   rank=" << rank << " shape=";
    printI64Array(llvm::outs(), inPort->shape);
    if (!kernel.boundsSource.empty()) {
      llvm::outs() << " bounds=" << kernel.boundsSource;
    }
    if (!kernel.radiusSource.empty()) {
      llvm::outs() << " radius_source=" << kernel.radiusSource;
    }
    llvm::outs() << "\n";
    llvm::outs() << "neptune-cc:   outMin=";
    printI64Array(llvm::outs(), kernel.outMin);
    llvm::outs() << " outExtent=";
    printI64Array(llvm::outs(), kernel.outExtent);
    llvm::outs() << " radius=";
    printI64Array(llvm::outs(), kernel.radius);
    llvm::outs() << "\n";

    for (const auto &port : kernel.ports) {
      const char *prefix = port.isInput ? "in" : "out";
      llvm::SmallVector<int64_t, 4> strides(rank, 1);
      for (size_t i = rank; i-- > 1;) {
        strides[i - 1] = strides[i] * port.shape[i];
      }

      llvm::SmallVector<int64_t, 4> mins(rank, 0);
      llvm::SmallVector<int64_t, 4> extents(rank, 0);
      int64_t offsetElems = 0;
      if (port.isInput) {
        bool useGhosted = port.isGhosted && allowGhostedPolicy;
        const char *policy = useGhosted ? "P1" : "P0";
        for (size_t d = 0; d < rank; ++d) {
          if (useGhosted) {
            mins[d] = -kernel.radius[d];
          } else {
            mins[d] = -kernel.outMin[d];
          }
          extents[d] = port.shape[d];
        }
        if (useGhosted) {
          for (size_t d = 0; d < rank; ++d) {
            offsetElems += (kernel.outMin[d] - kernel.radius[d]) * strides[d];
          }
        }
        llvm::outs() << "neptune-cc:   " << prefix << port.roleIndex
                     << " policy=" << policy;
        if (port.isGhosted) {
          if (useGhosted) {
            llvm::outs() << "(ghosted)";
          } else {
            llvm::outs() << "(ghosted->P0)";
          }
        }
        llvm::outs() << " mins=";
        printI64Array(llvm::outs(), mins);
        llvm::outs() << " host_offset_elems=" << offsetElems << "\n";
      } else {
        for (size_t d = 0; d < rank; ++d) {
          mins[d] = 0;
          extents[d] = kernel.outExtent[d];
          offsetElems += kernel.outMin[d] * strides[d];
        }
        llvm::outs() << "neptune-cc:   " << prefix << port.roleIndex
                     << " mins=";
        printI64Array(llvm::outs(), mins);
        llvm::outs() << " host_offset_elems=" << offsetElems << "\n";
      }

      os << "static const halide_dimension_t " << kernel.tag << "_" << prefix
         << port.roleIndex << "_dims[" << rank << "] = {\n";
      for (size_t i = 0; i < rank; ++i) {
        size_t d = rank - 1 - i;
        os << "  {" << static_cast<long long>(mins[d]) << ", "
           << static_cast<long long>(extents[d]) << ", "
           << static_cast<long long>(strides[d]) << ", 0}";
        if (i + 1 < rank) {
          os << ",";
        }
        os << "\n";
      }
      os << "};\n";
      os << "static const halide_buffer_t " << kernel.tag << "_" << prefix
         << port.roleIndex << "_templ = {\n";
      os << "  0,\n";
      os << "  nullptr,\n";
      os << "  nullptr,\n";
      os << "  0,\n";
      os << "  k_i32_type,\n";
      os << "  " << rank << ",\n";
      os << "  const_cast<halide_dimension_t*>(" << kernel.tag << "_" << prefix
         << port.roleIndex << "_dims),\n";
      os << "  nullptr\n";
      os << "};\n\n";

      os << "static inline void init_" << kernel.tag << "_" << prefix
         << port.roleIndex << "(halide_buffer_t &buf, int32_t *base) {\n";
      os << "  buf = " << kernel.tag << "_" << prefix << port.roleIndex
         << "_templ;\n";
      os << "  buf.host = reinterpret_cast<uint8_t*>(base + "
         << static_cast<long long>(offsetElems) << ");\n";
      os << "}\n\n";
    }

    os << "int " << kernel.tag << "(";
    for (size_t i = 0; i < kernel.ports.size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      os << "int32_t* " << kernel.ports[i].name;
    }
    os << ") {\n";

    for (const auto &port : kernel.ports) {
      const char *prefix = port.isInput ? "in" : "out";
      os << "  halide_buffer_t " << prefix << port.roleIndex << "_buf;\n";
      os << "  init_" << kernel.tag << "_" << prefix << port.roleIndex << "("
         << prefix << port.roleIndex << "_buf, " << port.name << ");\n";
    }

    os << "\n  return ::" << kernel.tag << "(";
    for (size_t i = 0; i < kernel.ports.size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      const auto &port = kernel.ports[i];
      const char *prefix = port.isInput ? "in" : "out";
      os << "&" << prefix << port.roleIndex << "_buf";
    }
    os << ");\n";
    os << "}\n\n";
  }

  os << "} // namespace neptunecc\n";
  return true;
}

static bool writeGlueCMake(llvm::StringRef glueDir,
                           llvm::ArrayRef<KernelInfo> kernels) {
  std::string cmakePath = joinPath(glueDir, {"neptunecc_generated.cmake"});
  std::ofstream os(cmakePath);
  if (!os.is_open()) {
    llvm::errs() << "neptune-cc: failed to open glue cmake '"
                 << cmakePath << "'\n";
    return false;
  }

  os << "set(NEPTUNECC_GENERATED_DIR \"${CMAKE_CURRENT_LIST_DIR}\")\n";
  os << "set(NEPTUNECC_HALIDE_DIR \"${CMAKE_CURRENT_LIST_DIR}/../halide\")\n\n";
  os << "add_library(neptunecc_glue STATIC\n";
  os << "  \"${NEPTUNECC_GENERATED_DIR}/neptunecc_kernels.cpp\"\n";
  os << ")\n\n";
  os << "target_include_directories(neptunecc_glue PUBLIC\n";
  os << "  \"${NEPTUNECC_GENERATED_DIR}\"\n";
  os << "  \"${NEPTUNECC_HALIDE_DIR}\"\n";
  os << ")\n\n";
  os << "# Halide kernel libraries (AOT)\n";
  for (const auto &kernel : kernels) {
    if (!kernel.halideCompatible || !kernel.boundsValid) {
      if (!kernel.halideCompatible) {
        llvm::errs() << "neptune-cc: skipping CMake target for kernel '"
                     << kernel.tag
                     << "' (requires single input/output with matching rank)\n";
      } else {
        llvm::errs() << "neptune-cc: skipping CMake target for kernel '"
                     << kernel.tag << "' (no valid stencil bounds)\n";
      }
      continue;
    }
    os << "add_library(neptunecc_" << kernel.tag
       << " STATIC IMPORTED GLOBAL)\n";
    os << "set(_neptunecc_" << kernel.tag << "_lib \"${NEPTUNECC_HALIDE_DIR}/lib"
       << kernel.tag << ".a\")\n";
    os << "if(NOT EXISTS \"${_neptunecc_" << kernel.tag << "_lib}\")\n";
    os << "  set(_neptunecc_" << kernel.tag
       << "_lib \"${NEPTUNECC_HALIDE_DIR}/" << kernel.tag << ".a\")\n";
    os << "endif()\n";
    os << "set_target_properties(neptunecc_" << kernel.tag
       << " PROPERTIES\n";
    os << "  IMPORTED_LOCATION \"${_neptunecc_" << kernel.tag << "_lib}\"\n";
    os << ")\n";
    os << "target_link_libraries(neptunecc_glue PUBLIC neptunecc_" << kernel.tag
       << ")\n";
  }
  os << "\n# If needed, link Halide runtime here.\n";
  return true;
}

static bool writeHalideHelperFile(llvm::StringRef halideDir) {
  std::string helperPath =
      joinPath(halideDir, {"NeptuneHalideHelpers.h"});
  std::ofstream os(helperPath);
  if (!os.is_open()) {
    llvm::errs() << "neptune-cc: failed to open Halide helper header '"
                 << helperPath << "'\n";
    return false;
  }

  os << "#pragma once\n";
  os << "#include \"Halide.h\"\n";
  os << "#include <string>\n";
  os << "#include <vector>\n";
  os << "\nnamespace neptune_halide {\n\n";
  os << "inline void push_arg(std::vector<Halide::Argument> &args,\n";
  os << "                     const Halide::Argument &arg) {\n";
  os << "  args.push_back(arg);\n";
  os << "}\n\n";
  os << "inline void assign(Halide::FuncRef &ref, const Halide::Expr &expr) {\n";
  os << "  ref = expr;\n";
  os << "}\n\n";
  os << "inline void compile(Halide::Func &func, const std::string &prefix,\n";
  os << "                    const std::vector<Halide::Argument> &args,\n";
  os << "                    const std::string &fn, const Halide::Target &target) {\n";
  os << "  func.compile_to_static_library(prefix, args, fn, target);\n";
  os << "}\n\n";
  os << "} // namespace neptune_halide\n";
  return true;
}

} // namespace

bool writeGlue(const EventDB &db, llvm::StringRef outDir) {
  llvm::SmallVector<KernelInfo, 8> kernels;
  if (!collectKernelInfos(db, kernels)) {
    return false;
  }

  std::string glueDir = joinPath(outDir, {"glue"});
  if (!ensureDir(glueDir, "glue")) {
    return false;
  }

  if (!writeGlueHeader(glueDir, kernels)) {
    return false;
  }
  if (!writeGlueCpp(glueDir, kernels)) {
    return false;
  }
  if (!writeGlueCMake(glueDir, kernels)) {
    return false;
  }

  return true;
}

bool writeHalideHelper(llvm::StringRef outDir) {
  std::string halideDir = joinPath(outDir, {"halide"});
  if (!ensureDir(halideDir, "halide")) {
    return false;
  }
  return writeHalideHelperFile(halideDir);
}

bool rewriteKernelSources(const EventDB &db, llvm::StringRef outDir) {
  llvm::SmallVector<KernelInfo, 8> kernels;
  if (!collectKernelInfos(db, kernels)) {
    return false;
  }

  llvm::StringMap<const KernelInfo *> kernelByTag;
  for (const auto &kernel : kernels) {
    kernelByTag[kernel.tag] = &kernel;
  }

  llvm::StringMap<llvm::SmallVector<Replacement, 4>> replacementsByFile;
  llvm::StringSet<> filesToRewrite;
  for (const auto &kernel : db.kernels) {
    if (!kernel.blockBegin.isValid() || !kernel.blockEnd.isValid()) {
      continue;
    }
    llvm::StringRef tag = kernel.begin.tag;
    if (tag.empty()) {
      continue;
    }
    llvm::StringRef filePath = kernel.begin.filePath;
    if (!filePath.empty()) {
      filesToRewrite.insert(filePath);
    }

    auto it = kernelByTag.find(tag);
    if (it == kernelByTag.end()) {
      llvm::errs() << "neptune-cc: no kernel metadata for tag '" << tag
                   << "'\n";
      continue;
    }
    if (!it->second->halideCompatible || !it->second->boundsValid) {
      if (!it->second->halideCompatible) {
        llvm::errs() << "neptune-cc: skipping rewrite for kernel '" << tag
                     << "' (requires single input/output with matching rank)\n";
      } else {
        llvm::errs() << "neptune-cc: skipping rewrite for kernel '" << tag
                     << "' (no valid stencil bounds)\n";
      }
      continue;
    }
    if (filePath.empty()) {
      llvm::errs() << "neptune-cc: missing source path for kernel '" << tag
                   << "'\n";
      continue;
    }

    auto bufferOrErr = llvm::MemoryBuffer::getFile(filePath);
    if (!bufferOrErr) {
      llvm::errs() << "neptune-cc: failed to read source '" << filePath
                   << "': " << bufferOrErr.getError().message() << "\n";
      continue;
    }
    llvm::StringRef contentRef = bufferOrErr.get()->getBuffer();

    size_t beginOffset = static_cast<size_t>(kernel.begin.fileOffset);
    size_t endOffset = static_cast<size_t>(kernel.end.fileOffset);
    if (beginOffset > contentRef.size() || endOffset > contentRef.size()) {
      llvm::errs() << "neptune-cc: kernel offsets out of range for '" << tag
                   << "'\n";
      continue;
    }

    size_t start = findLineStart(contentRef, beginOffset);
    size_t end = findLineEnd(contentRef, endOffset);
    if (start >= end) {
      llvm::errs() << "neptune-cc: invalid kernel range for '" << tag << "'\n";
      continue;
    }

    size_t indentEnd = start;
    while (indentEnd < contentRef.size() &&
           (contentRef[indentEnd] == ' ' || contentRef[indentEnd] == '\t')) {
      ++indentEnd;
    }
    llvm::StringRef indent =
        contentRef.slice(start, std::min(indentEnd, contentRef.size()));
    std::string newline = detectNewline(contentRef);

    Replacement repl;
    repl.start = start;
    repl.end = end;
    repl.text = indent.str();
    repl.text.append(buildKernelCall(*it->second));
    repl.text.append(newline);
    replacementsByFile[filePath].push_back(std::move(repl));
  }

  if (filesToRewrite.empty()) {
    return true;
  }

  std::string rewriteDir = joinPath(outDir, {"rewritten"});
  if (!ensureDir(rewriteDir, "rewritten")) {
    return false;
  }

  for (auto &entry : filesToRewrite) {
    llvm::StringRef filePath = entry.getKey();
    auto bufferOrErr = llvm::MemoryBuffer::getFile(filePath);
    if (!bufferOrErr) {
      llvm::errs() << "neptune-cc: failed to read source '" << filePath
                   << "': " << bufferOrErr.getError().message() << "\n";
      continue;
    }

    std::string content = bufferOrErr.get()->getBuffer().str();
    auto replIt = replacementsByFile.find(filePath);
    if (replIt != replacementsByFile.end()) {
      auto &repls = replIt->second;
      std::stable_sort(repls.begin(), repls.end(),
                       [](const Replacement &a, const Replacement &b) {
                         return a.start > b.start;
                       });

      for (const auto &repl : repls) {
        if (repl.end > content.size()) {
          llvm::errs()
              << "neptune-cc: replacement out of range for '" << filePath
              << "'\n";
          return false;
        }
        content.replace(repl.start, repl.end - repl.start, repl.text);
      }

      ensureKernelInclude(content);
    }

    llvm::SmallString<256> outPath(rewriteDir);
    llvm::sys::path::append(outPath, llvm::sys::path::filename(filePath));
    std::ofstream os(outPath.str().str());
    if (!os.is_open()) {
      llvm::errs() << "neptune-cc: failed to write rewritten source '"
                   << outPath << "'\n";
      return false;
    }
    os << content;
  }

  return true;
}

bool writeHalideGenerators(const EventDB &db, llvm::StringRef outDir,
                           bool emitEmitcMLIR, bool emitHalideCpp) {
  if (!emitEmitcMLIR && !emitHalideCpp) {
    return true;
  }
  if (!db.kernelModule) {
    return true;
  }

  std::string halideDir = joinPath(outDir, {"halide"});
  if (!ensureDir(halideDir, "halide")) {
    return false;
  }

  mlir::ModuleOp module = db.kernelModule.get().clone();
  mlir::MLIRContext *ctx = module.getContext();

  mlir::PassManager pm(ctx);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::Neptune::NeptuneIR::createSCFToNeptuneIRPass());
  mlir::Neptune::NeptuneIR::NeptuneIREmitCHalidePassOptions opts;
  opts.outFile = "halide_kernels";
  pm.addPass(mlir::Neptune::NeptuneIR::createNeptuneIREmitCHalidePass(opts));

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "neptune-cc: halide pass pipeline failed\n";
    return false;
  }

  if (emitEmitcMLIR) {
    std::string emitcPath = joinPath(halideDir, {"emitc.mlir"});
    std::ofstream emitcFile(emitcPath);
    if (!emitcFile.is_open()) {
      llvm::errs() << "neptune-cc: failed to open emitc mlir '"
                   << emitcPath << "'\n";
      return false;
    }
    llvm::raw_os_ostream os(emitcFile);
    module.print(os);
    os << "\n";
  }

  if (emitHalideCpp) {
    std::string cppPath = joinPath(halideDir, {"halide_kernels.cpp"});
    std::ofstream cppFile(cppPath);
    if (!cppFile.is_open()) {
      llvm::errs() << "neptune-cc: failed to open halide cpp '" << cppPath
                   << "'\n";
      return false;
    }
    llvm::raw_os_ostream os(cppFile);
    if (mlir::failed(
            mlir::emitc::translateToCpp(module, os, false, "halide_kernels"))) {
      llvm::errs() << "neptune-cc: failed to translate emitc to C++\n";
      return false;
    }
  }

  return true;
}

} // namespace neptune
