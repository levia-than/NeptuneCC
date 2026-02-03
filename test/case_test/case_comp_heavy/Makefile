REPO_ROOT := $(abspath ../../..)
OUT := $(CURDIR)/out
HALIDE_ROOT ?= /home/wyx/project/stencil-solver-playground/halide/Halide-21.0.0-x86-64-linux
CLANG ?= clang++
CLANG_RESOURCE_DIR ?= $(shell $(CLANG) -print-resource-dir 2>/dev/null)
NEPTUNE_CC_EXTRA_ARGS ?= $(if $(CLANG_RESOURCE_DIR),--extra-arg=-resource-dir=$(CLANG_RESOURCE_DIR),)

NEPTUNE_CC ?= $(firstword \
  $(wildcard $(REPO_ROOT)/build/project-build/tools/neptune-cc/neptune-cc) \
  $(wildcard $(REPO_ROOT)/build/project-build/bin/neptune-cc))
SRC := case_comp_heavy.cpp
NEPTUNE_CC_FLAGS ?= --emit-halide

ifneq ($(filter clean,$(MAKECMDGOALS)),clean)
  ifeq ($(strip $(NEPTUNE_CC)),)
    $(error NEPTUNE_CC not found. Build NeptuneCC or set NEPTUNE_CC=/path/to/neptune-cc)
  endif
endif

.PHONY: all clean run compare time

all: compare

$(OUT)/compile_commands.json: $(SRC)
	@mkdir -p $(OUT)
	@echo "[neptune-cc] generate compile_commands.json"
	@printf '[{"directory": "%s", "command": "clang++ -std=c++17 -I%s -c %s", "file": "%s"}]\n' "$(CURDIR)" "$(REPO_ROOT)/include" "$(CURDIR)/$(SRC)" "$(CURDIR)/$(SRC)" > $@

$(OUT)/manifest.json: $(SRC) $(OUT)/compile_commands.json
	@mkdir -p $(OUT)
	@echo "[neptune-cc] $(NEPTUNE_CC) $(NEPTUNE_CC_EXTRA_ARGS) $(NEPTUNE_CC_FLAGS) -p $(OUT) $(SRC) --out-dir=$(OUT)"
	NEPTUNECC_BW_NET=1e12 NEPTUNECC_NET_LAT=1e-9 \
	NEPTUNECC_CALL_OVERHEAD_NS=5000 NEPTUNECC_BOUNDARY_OVERHEAD_PER_ELEM=1e-9 \
	$(NEPTUNE_CC) $(NEPTUNE_CC_EXTRA_ARGS) $(NEPTUNE_CC_FLAGS) -p $(OUT) $(SRC) --out-dir=$(OUT)

$(OUT)/halide/halide_kernels.cpp: $(OUT)/manifest.json
	@echo "[check] expected $@"
	@test -f $@

$(OUT)/halide/halide_build: $(OUT)/halide/halide_kernels.cpp
	@echo "[halide] build generator"
	@printf '%s\n' \
	  '#include <cstdio>' \
	  'void neptuneir_build_halide_kernels();' \
	  'int main() {' \
	  '  neptuneir_build_halide_kernels();' \
	  '  std::printf("halide kernels built\\n");' \
	  '  return 0;' \
	  '}' | \
	clang++ -O2 -std=c++17 -x c++ - \
	  -I"$(HALIDE_ROOT)/include" -I"$(REPO_ROOT)/include" -I"$(OUT)/halide" \
	  $(OUT)/halide/halide_kernels.cpp \
	  -L"$(HALIDE_ROOT)/lib" -lHalide -Wl,-rpath,"$(HALIDE_ROOT)/lib" \
	  -o $@

$(OUT)/halide/k0.h $(OUT)/halide/k0.a: $(OUT)/halide/halide_build
	@echo "[halide] run generator -> k0.h / k0.a"
	cd $(OUT)/halide && LD_LIBRARY_PATH="$(HALIDE_ROOT)/lib" ./halide_build

$(OUT)/baseline: $(SRC)
	@echo "[baseline] clang++ -O2 -DNDEBUG $(SRC) -o $@"
	clang++ -O2 -DNDEBUG -I"$(REPO_ROOT)/include" $(SRC) -o $@

$(OUT)/generated: $(OUT)/rewritten/$(SRC) $(OUT)/glue/neptunecc_kernels.cpp $(OUT)/halide/k0.a
	@echo "[generated] clang++ -O2 -DNDEBUG (rewritten + glue + k0.a)"
	clang++ -O2 -DNDEBUG \
	  -I"$(OUT)/glue" -I"$(OUT)/halide" -I"$(HALIDE_ROOT)/include" -I"$(REPO_ROOT)/include" \
	  $(OUT)/rewritten/$(SRC) $(OUT)/glue/neptunecc_kernels.cpp $(OUT)/halide/k0.a \
	  -o $@

$(OUT)/baseline.out: $(OUT)/baseline
	$(OUT)/baseline > $@

$(OUT)/generated.out: $(OUT)/generated
	$(OUT)/generated > $@

compare: $(OUT)/halide/k0.h $(OUT)/generated $(OUT)/baseline.out $(OUT)/generated.out
	@echo "[compare] python3 compare.py"
	python3 compare.py $(OUT)/baseline.out $(OUT)/generated.out

run: compare

time: $(OUT)/baseline $(OUT)/generated
	@bash -c 'echo "baseline:"; time -p $(OUT)/baseline >/dev/null'
	@bash -c 'echo "generated:"; time -p $(OUT)/generated >/dev/null'

clean:
	rm -rf $(OUT)
