// Schedule debug record + emitters for reproducible dumps.
#pragma once

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace neptune {

struct PortRecord {
  std::string name;
  std::string direction;
  std::string qualifier;
  bool ghosted = false;
  unsigned roleIndex = 0;
  unsigned argIndex = 0;
};

struct ScheduleDecision {
  bool valid = false;
  int64_t rank = 0;
  std::vector<int64_t> tile;
  int64_t vec = 1;
  std::string vecDim;
  std::string vecReason;
  int64_t unroll = 1;
  std::string unrollDim;
  std::string unrollReason;
  std::string parDim;
  std::string parReason;
  int64_t threads = 1;
  int64_t l1Bytes = 0;
  double alpha = 0.0;
  int64_t footprint = 0;
  bool cacheFit = false;
  std::vector<std::string> reorder;
};

struct OverlapRegion {
  std::string name;
  std::vector<int64_t> lb;
  std::vector<int64_t> ub;
};

struct OverlapPlan {
  bool supported = false;
  bool enabled = false;
  std::string reason;
  std::vector<int64_t> interiorLb;
  std::vector<int64_t> interiorUb;
  std::vector<OverlapRegion> faces;
};

struct RewriteDecision {
  std::string mode;
  std::string reason;
  std::string boundaryCopyMode;
};

struct KernelScheduleRecord {
  std::string tag;
  std::string name;
  int64_t rank = 0;
  std::vector<int64_t> shape;
  std::vector<int64_t> outMin;
  std::vector<int64_t> outExtent;
  std::vector<int64_t> radius;
  std::vector<PortRecord> ports;
  ScheduleDecision schedule;
  OverlapPlan overlap;
  RewriteDecision rewrite;
};

bool scheduleDumpEnabled();
bool scheduleDotEnabled();

class ScheduleEmitter {
 public:
  ScheduleEmitter(llvm::StringRef outDir, bool enableJson, bool enableTxt,
                  bool enableDot);

  void recordKernel(const KernelScheduleRecord &rec);
  bool flush();

 private:
  bool writeJson();
  bool writeTxt();
  bool writeDot();

  std::string outDir;
  bool enableJson = false;
  bool enableTxt = false;
  bool enableDot = false;
  std::vector<KernelScheduleRecord> records;
};

}  // namespace neptune
// Defines schedule debug records and dump interfaces for reproducibility.
