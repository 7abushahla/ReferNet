"""Run the bounded, two-arm direct-reference diagnostic in an allocation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refer_or_clear.direct_reference_diagnostic import main


if __name__ == "__main__":
    raise SystemExit(0 if main()["passed"] else 1)
