// site/js/heatmap.js
const C = 32, H = 64, W = 100, N = H * W;

/** M = sum_c (w_mean_c / |overlap| + w_max_c) F_c inside the overlap, clipped at 0, scaled by its maximum. */
export function differenceMap(local, overlapSmall, head) {
  let area = 0;
  for (let i = 0; i < N; i++) area += overlapSmall[i] ? 1 : 0;
  const w = head.w_mean.map((m, c) => m / Math.max(area, 1) + head.w_max[c]);
  const out = new Float32Array(N);
  let max = 0;
  for (let i = 0; i < N; i++) {
    if (!overlapSmall[i]) continue;
    let s = 0;
    for (let c = 0; c < C; c++) s += w[c] * local[c * N + i];
    out[i] = Math.max(s, 0);
    if (out[i] > max) max = out[i];
  }
  if (max > 0) for (let i = 0; i < N; i++) out[i] /= max;
  return out;
}

export function downsampleOverlap(overlap) {
  const out = new Uint8Array(N);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) out[y * W + x] = overlap[y * 8 * 800 + x * 8] > 0.5 ? 1 : 0;
  return out;
}
