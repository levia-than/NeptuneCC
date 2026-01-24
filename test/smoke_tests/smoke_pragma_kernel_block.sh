#!/usr/bin/env bash
set -euo pipefail

NEPTUNE_CC=${NEPTUNE_CC:-/home/wyx/project/neptune-pde-solver/build/project-build/bin/neptune-cc}
INPUT_FILE=${1:-/home/wyx/project/neptune-pde-solver/test/smoke_tests/pragma_kernel_block.cpp}

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
