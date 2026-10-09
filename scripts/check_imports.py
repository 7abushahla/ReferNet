"""Fail fast if the environment cannot import what training needs."""
import importlib
import sys

MODULES = ["torch", "torchvision", "mmcv", "mmseg", "opencd", "cv2", "numpy", "pandas", "sklearn", "yaml",
           "refer_or_clear"]
missing = []
for m in MODULES:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{m}: {e}")
if missing:
    print("[check_imports] FAILED\n  " + "\n  ".join(missing))
    sys.exit(1)
print(f"[check_imports] ok: {', '.join(MODULES)}")
