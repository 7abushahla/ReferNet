// site/js/tensors.js
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

export function rgbToNormalizedCHW(rgba, w, h) {
  const n = w * h, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 3; c++) out[c * n + i] = (rgba[4 * i + c] / 255 - MEAN[c]) / STD[c];
  return out;
}

export function buildQuery(queryRgba, masksRgba, w, h) {
  const n = w * h, out = new Float32Array(4 * n);
  out.set(rgbToNormalizedCHW(queryRgba, w, h), 0);
  for (let i = 0; i < n; i++) out[3 * n + i] = masksRgba[4 * i] > 127 ? 1 : 0;
  return out;
}

export function buildOverlap(masksRgba, w, h) {
  const n = w * h, out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = masksRgba[4 * i + 1] > 127 ? 1 : 0;
  return out;
}

/** Exact PNG decode with UPNG (no canvas, so no color management or fingerprint noise): 8-bit RGBA. */
export function decodePng(buffer, UPNG) {
  const img = UPNG.decode(buffer);
  return { rgba: new Uint8Array(UPNG.toRGBA8(img)[0]), width: img.width, height: img.height };
}
