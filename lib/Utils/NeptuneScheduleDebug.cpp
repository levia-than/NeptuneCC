// Schedule debug record + emitters for reproducible dumps.
#include "Utils/NeptuneScheduleDebug.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

namespace neptune {
namespace {

static bool isEnvEnabled(llvm::StringRef name) {
  if (auto val = llvm::sys::Process::GetEnv(name)) {
    return !val->empty() && *val != "0";
  }
  return false;
}

static std::string joinPath(llvm::StringRef base,
                            std::initializer_list<llvm::StringRef> parts) {
  llvm::SmallString<256> path(base);
  for (llvm::StringRef part : parts) {
    llvm::sys::path::append(path, part);
  }
  return path.str().str();
}

static llvm::json::Array toJsonArray(const std::vector<int64_t> &vals) {
  llvm::json::Array out;
  out.reserve(vals.size());
  for (int64_t v : vals) {
    out.push_back(v);
  }
  return out;
}

static llvm::json::Array toJsonArrayStr(const std::vector<std::string> &vals) {
  llvm::json::Array out;
  out.reserve(vals.size());
  for (const auto &v : vals) {
    out.push_back(v);
  }
  return out;
}

}  // namespace

bool scheduleDumpEnabled() { return isEnvEnabled("NEPTUNECC_DUMP_SCHEDULE"); }

bool scheduleDotEnabled() {
  return isEnvEnabled("NEPTUNECC_DUMP_SCHEDULE_DOT");
}

ScheduleEmitter::ScheduleEmitter(llvm::StringRef outDir, bool enableJson,
                                 bool enableTxt, bool enableDot)
    : outDir(outDir.str()),
      enableJson(enableJson),
      enableTxt(enableTxt),
      enableDot(enableDot) {}

void ScheduleEmitter::recordKernel(const KernelScheduleRecord &rec) {
  records.push_back(rec);
}

bool ScheduleEmitter::flush() {
  if (!enableJson && !enableTxt && !enableDot) {
    return true;
  }

  std::error_code ec = llvm::sys::fs::create_directories(
      joinPath(outDir, {"schedule"}));
  if (ec) {
    llvm::errs() << "neptune-cc: failed to create schedule dir: "
                 << ec.message() << "\n";
    return false;
  }

  std::sort(records.begin(), records.end(),
            [](const KernelScheduleRecord &a, const KernelScheduleRecord &b) {
              return a.tag < b.tag;
            });

  if (enableJson && !writeJson())
    return false;
  if (enableTxt && !writeTxt())
    return false;
  if (enableDot && !writeDot())
    return false;
  return true;
}

bool ScheduleEmitter::writeJson() {
  std::string path = joinPath(outDir, {"schedule", "schedule.json"});
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "neptune-cc: failed to write schedule.json: "
                 << ec.message() << "\n";
    return false;
  }

  llvm::json::Array kernels;
  for (const auto &rec : records) {
    llvm::json::Object obj;
    obj["kernel"] = rec.tag;
    obj["name"] = rec.name;
    obj["rank"] = rec.rank;
    obj["shape"] = toJsonArray(rec.shape);

    llvm::json::Object bounds;
    bounds["lb"] = toJsonArray(rec.outMin);
    std::vector<int64_t> ub;
    ub.reserve(rec.outMin.size());
    for (size_t i = 0; i < rec.outMin.size(); ++i) {
      ub.push_back(rec.outMin[i] + rec.outExtent[i]);
    }
    bounds["ub"] = toJsonArray(ub);
    obj["bounds"] = std::move(bounds);
    obj["radius"] = toJsonArray(rec.radius);

    llvm::json::Array ports;
    for (const auto &p : rec.ports) {
      llvm::json::Object pobj;
      pobj["name"] = p.name;
      pobj["dir"] = p.direction;
      pobj["qualifier"] = p.qualifier;
      pobj["ghosted"] = p.ghosted;
      pobj["role_index"] = static_cast<int64_t>(p.roleIndex);
      pobj["arg_index"] = static_cast<int64_t>(p.argIndex);
      ports.push_back(std::move(pobj));
    }
    obj["ports"] = std::move(ports);

    llvm::json::Object sched;
    sched["valid"] = rec.schedule.valid;
    if (rec.schedule.valid) {
      sched["tile"] = toJsonArray(rec.schedule.tile);
      sched["vectorize"] = llvm::json::Object{
          {"dim", rec.schedule.vecDim},
          {"factor", rec.schedule.vec},
          {"enabled", rec.schedule.vec > 1},
          {"reason", rec.schedule.vecReason}};
      sched["unroll"] = llvm::json::Object{
          {"dim", rec.schedule.unrollDim},
          {"factor", rec.schedule.unroll},
          {"enabled", rec.schedule.unroll > 1},
          {"reason", rec.schedule.unrollReason}};
      sched["parallel"] = llvm::json::Object{
          {"dim", rec.schedule.parDim},
          {"enabled", !rec.schedule.parDim.empty() &&
                           rec.schedule.parDim != "none"},
          {"threads", rec.schedule.threads},
          {"reason", rec.schedule.parReason}};
      sched["reorder"] = toJsonArrayStr(rec.schedule.reorder);
      sched["why"] = llvm::json::Object{
          {"l1_bytes", rec.schedule.l1Bytes},
          {"alpha", rec.schedule.alpha},
          {"footprint_bytes", rec.schedule.footprint},
          {"cache_fit", rec.schedule.cacheFit}};
    }
    obj["schedule"] = std::move(sched);

    llvm::json::Object overlap;
    overlap["supported"] = rec.overlap.supported;
    overlap["enabled"] = rec.overlap.enabled;
    overlap["reason"] = rec.overlap.reason;
    overlap["interior_lb"] = toJsonArray(rec.overlap.interiorLb);
    overlap["interior_ub"] = toJsonArray(rec.overlap.interiorUb);
    llvm::json::Array faces;
    for (const auto &face : rec.overlap.faces) {
      llvm::json::Object f;
      f["name"] = face.name;
      f["lb"] = toJsonArray(face.lb);
      f["ub"] = toJsonArray(face.ub);
      faces.push_back(std::move(f));
    }
    overlap["faces"] = std::move(faces);
    obj["overlap"] = std::move(overlap);

    llvm::json::Object rewrite;
    rewrite["mode"] = rec.rewrite.mode;
    rewrite["reason"] = rec.rewrite.reason;
    rewrite["boundary_copy_mode"] = rec.rewrite.boundaryCopyMode;
    obj["rewrite"] = std::move(rewrite);

    kernels.push_back(std::move(obj));
  }

  llvm::json::Object root;
  root["version"] = 1;
  root["kernels"] = std::move(kernels);
  os << llvm::json::Value(std::move(root)) << "\n";
  return true;
}

bool ScheduleEmitter::writeTxt() {
  std::string path = joinPath(outDir, {"schedule", "schedule.txt"});
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "neptune-cc: failed to write schedule.txt: "
                 << ec.message() << "\n";
    return false;
  }

  for (const auto &rec : records) {
    os << "neptune-cc: schedule kernel=" << rec.tag << "\n";
    os << "  rank=" << rec.rank << " shape=";
    for (size_t i = 0; i < rec.shape.size(); ++i) {
      if (i)
        os << ',';
      os << rec.shape[i];
    }
    os << "\n  bounds.lb=";
    for (size_t i = 0; i < rec.outMin.size(); ++i) {
      if (i)
        os << ',';
      os << rec.outMin[i];
    }
    os << " bounds.ub=";
    for (size_t i = 0; i < rec.outMin.size(); ++i) {
      if (i)
        os << ',';
      os << rec.outMin[i] + rec.outExtent[i];
    }
    os << " radius=";
    for (size_t i = 0; i < rec.radius.size(); ++i) {
      if (i)
        os << ',';
      os << rec.radius[i];
    }
    os << "\n";

    if (rec.schedule.valid) {
      os << "  tile=";
      for (size_t i = 0; i < rec.schedule.tile.size(); ++i) {
        if (i)
          os << ',';
        os << rec.schedule.tile[i];
      }
      os << "\n";
      os << "  vectorize={dim=" << rec.schedule.vecDim
         << ",factor=" << rec.schedule.vec
         << ",enabled=" << (rec.schedule.vec > 1) << "}";
      if (!rec.schedule.vecReason.empty())
        os << " reason=\"" << rec.schedule.vecReason << "\"";
      os << "\n";
      os << "  unroll={dim=" << rec.schedule.unrollDim
         << ",factor=" << rec.schedule.unroll
         << ",enabled=" << (rec.schedule.unroll > 1) << "}";
      if (!rec.schedule.unrollReason.empty())
        os << " reason=\"" << rec.schedule.unrollReason << "\"";
      os << "\n";
      os << "  parallel={dim=" << rec.schedule.parDim
         << ",threads=" << rec.schedule.threads
         << ",enabled="
         << (!rec.schedule.parDim.empty() && rec.schedule.parDim != "none")
         << "}";
      if (!rec.schedule.parReason.empty())
        os << " reason=\"" << rec.schedule.parReason << "\"";
      os << "\n";
      os << "  cache: L1=" << rec.schedule.l1Bytes
         << " alpha=" << rec.schedule.alpha
         << " footprint=" << rec.schedule.footprint
         << " fit=" << (rec.schedule.cacheFit ? "true" : "false")
         << "\n";
    } else {
      os << "  schedule: <missing>\n";
    }

    os << "  overlap: supported=" << (rec.overlap.supported ? "true" : "false")
       << " enabled=" << (rec.overlap.enabled ? "true" : "false")
       << " reason=\"" << rec.overlap.reason << "\"\n";

    os << "  rewrite: mode=" << rec.rewrite.mode
       << " reason=\"" << rec.rewrite.reason << "\"";
    if (!rec.rewrite.boundaryCopyMode.empty()) {
      os << " boundary_copy=" << rec.rewrite.boundaryCopyMode;
    }
    os << "\n\n";
  }

  return true;
}

bool ScheduleEmitter::writeDot() {
  for (const auto &rec : records) {
    if (!rec.overlap.enabled)
      continue;
    std::string path =
        joinPath(outDir, {"schedule", "overlap_" + rec.tag + ".dot"});
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "neptune-cc: failed to write overlap dot: "
                   << ec.message() << "\n";
      return false;
    }

    os << "digraph overlap_" << rec.tag << " {\n";
    os << "  hb [label=\"halo_begin\"];\n";
    os << "  interior [label=\"interior\"];\n";
    os << "  he [label=\"halo_end\"];\n";
    for (size_t i = 0; i < rec.overlap.faces.size(); ++i) {
      os << "  b" << i << " [label=\"" << rec.overlap.faces[i].name
         << "\"];\n";
    }
    os << "  hb -> interior -> he";
    for (size_t i = 0; i < rec.overlap.faces.size(); ++i) {
      os << " -> b" << i;
    }
    os << ";\n";
    os << "}\n";
  }
  return true;
}

}  // namespace neptune
