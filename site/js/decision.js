// site/js/decision.js
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));
export const decide = (p, threshold = 0.5) => (p > threshold ? "REFER" : "CLEAR");
export const decideMismatch = () => ({ decision: "REFER", reason: "no reference context, sent to the second line" });

export function percent(p) {
  if (p < 0.005) return "<1%";
  if (p > 0.995) return ">99%";
  return `${Math.round(p * 100)}%`;
}
