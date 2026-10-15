// site/js/library.js
export function documentOptions(manifest, designId) {
  const design = manifest.designs.find((d) => d.id === designId);
  const own = manifest.samples
    .filter((s) => s.design === designId)
    .map((s) => ({ value: s.id, text: `${s.family_name} · #${s.n}` }));
  const others = manifest.mismatch
    .filter((m) => !m.sample.startsWith(`${designId}__`))
    .map((m) => ({ value: `mismatch:${m.sample}`, text: m.label }));
  return [{ group: design.name, items: own }, { group: "Another design (no reference match)", items: others }];
}

export function findSample(manifest, value) {
  if (value.startsWith("mismatch:")) {
    const id = value.slice("mismatch:".length);
    return { kind: "mismatch", entry: manifest.mismatch.find((m) => m.sample === id),
             sample: manifest.samples.find((s) => s.id === id) };
  }
  return { kind: "sample", sample: manifest.samples.find((s) => s.id === value) };
}
