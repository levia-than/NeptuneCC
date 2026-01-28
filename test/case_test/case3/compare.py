import sys

if len(sys.argv) != 3:
    print("usage: compare.py <baseline> <generated>")
    raise SystemExit(2)

with open(sys.argv[1], "r", encoding="utf-8") as f:
    base = f.read().strip()
with open(sys.argv[2], "r", encoding="utf-8") as f:
    gen = f.read().strip()

if base != gen:
    print("mismatch:\n--- baseline ---\n" + base + "\n--- generated ---\n" + gen)
    raise SystemExit(1)

print("outputs match")
