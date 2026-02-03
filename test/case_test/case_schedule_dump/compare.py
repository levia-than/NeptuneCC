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
sched_json = out_dir / "schedule" / "schedule.json"
sched_txt = out_dir / "schedule" / "schedule.txt"

if not sched_json.is_file():
    print(f"missing {sched_json}")
    sys.exit(1)
if not sched_txt.is_file():
    print(f"missing {sched_txt}")
    sys.exit(1)

try:
    data = json.loads(sched_json.read_text())
except json.JSONDecodeError as exc:
    print(f"invalid schedule.json: {exc}")
    sys.exit(1)

kernels = data.get("kernels", [])
if not any(k.get("kernel") == "k0" for k in kernels):
    print("schedule.json missing kernel k0")
    sys.exit(1)

print("outputs match")
