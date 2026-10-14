# Per-design score decomposition behind the 14 Oct manuscript rewrite

`diag_greece_copy.py` reads the locked prediction files only (`results/diagnostics/locked-predictions/protocol-b/*-v2noaux-fdb7351` and `results/diagnostics/protocol-c-idnet/*/fold0_*_q75_predictions.csv`). It trains nothing and changes nothing. It was run as Slurm job 38878, and `diag_greece_copy_output.txt` is its output.

The manuscript's Sections 5.3 to 5.6 use these numbers:
- **Protocol B, per held-out design, mean over the groups that hold it out:**
  - AUC overall, for inpaint-and-rewrite and for copy-and-replace;
  - genuine documents referred at the fixed 0.5 rule;
  - attacks caught at a 10% genuine referral rate;
  - mean query logit and correction logit by class.
- **Protocol C, deployment model (fold 0), equalized files:** the same decomposition per IDNet design, plus genuine correction quantiles.

Key readings:
- **Greece is weak under IDNet's rendering only.** Held out in Protocol B (SIDTD rendering), its ON AUC is 0.92. On IDNet's Greece renders, the ON correction is already +3.2 on genuine images (median +5.4), close to inpainting (+5.4) and copy-and-replace (+6.6). On the other seven IDNet designs, the genuine mean is -2.9 to -0.1.
- **Copy-and-replace in Protocol B.** The correction stays near its genuine level on copy-and-replace (+2 to +9, against -2 to +7 on genuine documents) but rises to +8 to +23 on inpaint-and-rewrite.
