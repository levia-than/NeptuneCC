import json
import pathlib
import sys

if len(sys.argv) != 3:
    print("usage: compare.py baseline.out generated.out")
    sys.exit(2)

baseline_path = pathlib.Path(sys.argv[1])
generated_path = pathlib.Path(sys.argv[2])

baseline = baseline_path.read_text()
generated = generated_path.read_text()
if baseline != generated:
    print("outputs differ")
    sys.exit(1)

out_dir = baseline_path.parent
strategy_json = out_dir / "strategy.json"
if not strategy_json.is_file():
    print(f"missing {strategy_json}")
    sys.exit(1)

data = json.loads(strategy_json.read_text())
entries = data.get("kernels", [])
mode = None
for entry in entries:
    if entry.get("kernel") == "k0":
        mode = entry.get("strategy", {}).get("mode")
        break
if mode != "overlap_split":
    print(f"expected overlap_split, got {mode}")
    sys.exit(1)

print("outputs match")
