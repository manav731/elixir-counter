"""Lists any class in models/card_classifier.pth that has no entry in the elixir cost table.
Run after (re)training. Works without torch installed."""
import os, pickletools, sys, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elixir_tracker import lookup_cost

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "card_classifier.pth")
z = zipfile.ZipFile(path)
data = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])
strs = [a for op, a, _ in pickletools.genops(data)
        if op.name in ("BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8")]
classes = [s for s in strs[strs.index("classes") + 1:] if not s.startswith(("collections", "torch", "_"))]
missing = [c for c in classes if c != "background" and not lookup_cost(c)[1]]
print(f"{len(classes)} classes in checkpoint, {len(missing)} without a cost entry")
for c in missing:
    print("  MISSING:", c)
sys.exit(1 if missing else 0)
