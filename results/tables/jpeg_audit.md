# JPEG quality audit (estimated from quantization tables)

| dataset | folder | label | n | median QF | min | max |
|---|---|---|---|---|---|---|
| fantasyid | test/attack | 1 | 1085 | 95.0 | 75.0 | 95.0 |
| fantasyid | test/bonafide | 0 | 300 | 93.5 | 93.5 | 93.5 |
| fantasyid | train/attack | 1 | 1266 | 75.0 | 75.0 | 75.0 |
| fantasyid | train/bonafide | 0 | 633 | 95.0 | 95.0 | 95.0 |
| sidtd | templates/Images | 0 | 1000 | 92.0 | 92.0 | 92.0 |
| sidtd | templates/Images | 1 | 1222 | 75.0 | 75.0 | 75.0 |

## Shortcut check per split (AUC of 'lower QF means altered'; 1.0 = compression alone separates the classes)

| protocol | role | subset | n | median QF genuine | median QF altered | AUC(QF) | acc QF<85 |
|---|---|---|---|---|---|---|---|
| protocol_a | test | (all) | 894 | 92.0 | 75.0 | 1.000 | 1.000 |
| protocol_a | train | (all) | 1184 | 92.0 | 75.0 | 1.000 | 1.000 |
| protocol_a | val | (all) | 144 | 92.0 | 75.0 | 1.000 | 1.000 |
| protocol_b | test | heldout | 599 | 93.5 | 75.0 | 1.000 | 1.000 |
| protocol_b | train | (all) | 1981 | 92.0 | 75.0 | 1.000 | 1.000 |
| protocol_b | val | (all) | 241 | 92.0 | 75.0 | 1.000 | 1.000 |

## Pixel-level trace, Protocol B held-out test (n = 599)

| input | AUC of 8-px blockiness |
|---|---|
| raw files (native resolution, native QF) | 0.971 |
| canvases (letterbox 512x800, re-encoded QF 95) | 0.513 |
