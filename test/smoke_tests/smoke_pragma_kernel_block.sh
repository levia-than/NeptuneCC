#!/usr/bin/env bash
set -euo pipefail

NEPTUNE_CC=${NEPTUNE_CC:-/home/wyx/project/NeptuneCC/build/project-build/tools/neptune-cc/neptune-cc}
if [ ! -x "$NEPTUNE_CC" ]; then
  NEPTUNE_CC=/home/wyx/project/NeptuneCC/build/project-build/bin/neptune-cc
fi
INPUT_FILE=${1:-/home/wyx/project/NeptuneCC/test/smoke_tests/pragma_kernel_block.cpp}

WORKDIR=${WORKDIR:-/tmp/neptune_pragma_smoke}
OUTDIR="$WORKDIR/out"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

"$NEPTUNE_CC" "$INPUT_FILE" --out-dir="$OUTDIR"

MANIFEST="$OUTDIR/manifest.json"
test -f "$MANIFEST"

python3 - "$MANIFEST" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

assert data.get("version") == 1, "version"
kernels = data.get("kernels")
assert isinstance(kernels, list), "kernels list"
assert len(kernels) == 1, "kernels size"

k0 = kernels[0]
assert k0.get("tag") == "k0", "tag"
pb = k0.get("pragma_begin_offset", 0)
pe = k0.get("pragma_end_offset", 0)
bb = k0.get("block_begin_offset", 0)
be = k0.get("block_end_offset", 0)

assert bb != 0 and be != 0, "block offsets"
assert pb <= bb <= be <= pe, "block within pragma range"
PY

KERNELS="$OUTDIR/kernels.mlir"
test -f "$KERNELS"
grep -Eq 'func.func @k0\(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: memref<4x4xi32>, %arg3: memref<4x4xi32>\)' "$KERNELS"
test "$(grep -c "scf.for" "$KERNELS" || true)" -ge 2
grep -q "memref.load" "$KERNELS"
grep -q "memref.store" "$KERNELS"
! grep -q "memref.alloca" "$KERNELS"
! grep -Eq 'scf\.for .*neptunecc\.' "$KERNELS"
grep -q 'neptunecc.tag = "k0"' "$KERNELS"
grep -q 'neptunecc.dm = "da"' "$KERNELS"
grep -q 'neptunecc.block_begin_offset' "$KERNELS"
grep -q 'neptunecc.port_map' "$KERNELS"
grep -q 'a=in0:ghosted:arg0' "$KERNELS"
grep -q 'x=in1:ghosted:arg2' "$KERNELS"
grep -q 'b=out0:owned:arg1' "$KERNELS"
grep -q 'y=out1:owned:arg3' "$KERNELS"
