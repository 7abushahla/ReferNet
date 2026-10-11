"""Run bounded diagnostics or validate a selected checkpoint. Never launch long training."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refer_or_clear.pilots import check_inference_contract, check_pilot_gates, fresh_process_replay, run_pilot, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pretrain", "selected"), required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir
    options = vars(args).copy()
    phase = options.pop("phase")
    options.pop("run_dir")
    options["config_path"] = options.pop("config")
    if phase == "pretrain":
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("diagnostic run directory must be empty")
        run_dir.mkdir(parents=True, exist_ok=True)
    try:
        if phase == "pretrain":
            for stage in ("overfit", "smoke"):
                result = run_pilot(**options, run_dir=run_dir / stage, phase=stage)
                if not result.get("passed"):
                    gate = check_pilot_gates(run_dir, phase)
                    write_json(run_dir / "gate.json", gate)
                    return gate
            fresh_process_replay(**options, run_dir=run_dir / "smoke")
        else:
            fresh_process_replay(**options, run_dir=run_dir)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        gate = check_pilot_gates(run_dir, phase)
        gate["passed"] = False
        gate["failures"].append(str(error))
        write_json(run_dir / "gate.json", gate)
        return gate
    gate = check_pilot_gates(run_dir, phase)
    write_json(run_dir / "gate.json", gate)
    return gate


if __name__ == "__main__":
    raise SystemExit(0 if main()["passed"] else 1)
