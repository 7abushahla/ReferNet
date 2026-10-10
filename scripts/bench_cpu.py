"""CPU inference timing for a trained checkpoint (E10).

Measures, at batch size one on CPU with a fixed thread count:
  pre-processing  decode, rectification by homography, resize, zone-mask lookup
  network         query plus reference forward; and with reference features cached per design
  end to end      decode to referral score
Reports median and p95 over --n documents after --warmup warm-ups, FPS (documents per second) from the
median end-to-end and network-only times, plus parameter count and file size.
Writes results/timing/<tag>-<host>.json. Quantization is out of scope here.
"""
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--onnx", action="store_true", help="export to ONNX and time with onnxruntime as well")
    ap.parse_args()
    raise NotImplementedError("implemented once a checkpoint exists; see experiments/E10_cpu_timing/NOTES.md")

if __name__ == "__main__":
    main()
