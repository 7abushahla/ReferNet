# E10: CPU inference time

Why: the first line has seconds per document, so the detector must run inside that budget on the counter computer with the scanner as the sensor. This is a reported result (RQ5), not future work.

    python scripts/bench_cpu.py --checkpoint results/runs/<tag>/best.pt --n 200 --warmup 20 --threads 4

Reports median and p95 for: pre-processing (decode, rectify, resize, zone lookup), network (query+reference; and reference features cached per design), end to end; FPS (documents per second) from the median end-to-end time and network-only; parameters and file size. Run on a cluster CPU node and on one laptop/desktop; record CPU model, threads, torch and onnxruntime versions. Quantization and on-sensor execution are future work.
