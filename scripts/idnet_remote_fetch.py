"""Fetch a sample of one IDNet design from its Zenodo zip without downloading the archive.

Zenodo serves the archives with HTTP range support, so the zip central directory (at the end of the file) and the
individual members can be read on demand.  A design archive is 5 to 45 GB; the Protocol C harness needs about
400 genuine images, 300 attacks per family and the annotation JSON files, roughly 400 MB.

Usage:
  python scripts/idnet_remote_fetch.py --record 13852734 --design FIN --out data/raw/idnet/FIN [--list]
      [--n-genuine 450 --n-per-family 350 --families fraud1_copy_and_move,fraud5_inpaint_and_rewrite,fraud6_crop_and_replace]
      [--seed 20261014]

The genuine sample is seeded; attacks are chosen among those whose identity stem is in the genuine sample's
complement (the harness later excludes support identities itself), so the chosen attacks cover the sampled identities.
Everything under meta/ is fetched.  Re-running resumes: members already on disk are skipped.
"""
import argparse, io, json, random, sys, time, zipfile
from pathlib import Path
from urllib.request import Request, urlopen

SEED = 20261014
SPAN = 24 << 20   # max bytes per range request
GAP = 512 << 10   # max unused bytes between two members fetched in one span
DEFAULT_FAMILIES = ("fraud1_copy_and_move", "fraud5_inpaint_and_rewrite", "fraud6_crop_and_replace",
                    "fraud2_face_morphing", "fraud3_face_replacement")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (block cache for the central directory)."""

    def __init__(self, url, block=4 << 20):
        self.url, self.block = url, block
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"]); self.url = r.geturl()
        self.pos = 0; self.cache = {}; self.bytes = 0; self.requests = 0

    def _fetch(self, start, end):
        for attempt in range(6):
            try:
                req = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urlopen(req, timeout=120) as r:
                    data = r.read(); self.bytes += len(data); self.requests += 1; return data
            except Exception as e:  # noqa: BLE001
                time.sleep(2 ** attempt); err = e
        raise err

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos)); out = bytearray()
        while n:
            b = self.pos // self.block
            if b not in self.cache:
                s = b * self.block; self.cache[b] = self._fetch(s, min(s + self.block, self.size) - 1)
            chunk = self.cache[b][self.pos - b * self.block:][:n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def read_span(self, start, length):
        """Uncached direct read used for member payloads."""
        return self._fetch(start, start + length - 1)


def member_bytes(f, zf, info):
    """Read one member (local header + payload) with a single range request and decompress it."""
    f.seek(info.header_offset); lh = f.read(30)
    if lh[:4] != b"PK\x03\x04": raise RuntimeError(f"bad local header for {info.filename}")
    n_name, n_extra = int.from_bytes(lh[26:28], "little"), int.from_bytes(lh[28:30], "little")
    start = info.header_offset + 30 + n_name + n_extra
    raw = f.read_span(start, info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED: return raw
    if info.compress_type == zipfile.ZIP_DEFLATED:
        import zlib; return zlib.decompress(raw, -15)
    raise RuntimeError(f"unsupported compression {info.compress_type} for {info.filename}")


def stem_of(name):
    s = Path(name).stem; return s.split("_fake_")[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--record", required=True); ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--list", action="store_true")
    ap.add_argument("--n-genuine", type=int, default=450); ap.add_argument("--n-per-family", type=int, default=350)
    ap.add_argument("--families", default=",".join(DEFAULT_FAMILIES)); ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    url = f"https://zenodo.org/api/records/{a.record}/files/{a.design}.zip/content"
    f = HttpRangeFile(url); print(f"{a.design}.zip {f.size / 1e9:.2f} GB at {f.url}", flush=True)
    zf = zipfile.ZipFile(f); infos = [i for i in zf.infolist() if not i.is_dir()]
    print(f"central directory read: {len(infos)} members, {f.bytes / 1e6:.1f} MB in {f.requests} requests", flush=True)
    # layout: <DESIGN>/positive/x.png, <DESIGN>/fraudK_name/x.png, <DESIGN>/meta/...
    by_dir = {}
    for i in infos:
        parts = Path(i.filename).parts
        if len(parts) < 2: continue
        rel = Path(*parts[1:]) if parts[0] == a.design else Path(i.filename)
        by_dir.setdefault(rel.parts[0], []).append((rel, i))
    for d, items in sorted(by_dir.items()):
        print(f"  {d:32s} {len(items):6d} members {sum(i.file_size for _, i in items) / 1e6:9.1f} MB", flush=True)
    if a.list: return
    rng = random.Random(a.seed)
    pos = sorted(by_dir.get("positive", []), key=lambda t: t[0].name)
    genuine = pos[:]; rng.shuffle(genuine); genuine = genuine[: a.n_genuine]
    chosen = list(by_dir.get("meta", [])) + genuine
    for d, items in by_dir.items():  # small auxiliary folders (e.g. SIDTD_Generated) are taken whole
        if d not in ("meta", "positive") and not d.startswith("fraud") and len(items) < 200: chosen += items
    stems = {stem_of(r.name) for r, _ in genuine}
    for fam in a.families.split(","):
        items = sorted(by_dir.get(fam, []), key=lambda t: t[0].name)
        if not items: print(f"  family {fam} absent", flush=True); continue
        pref = [t for t in items if stem_of(t[0].name) in stems]; rest = [t for t in items if stem_of(t[0].name) not in stems]
        rng.shuffle(pref); rng.shuffle(rest); chosen += (pref + rest)[: a.n_per_family]
    out = Path(a.out); todo = [(r, i) for r, i in chosen if not (out / r).exists()]
    total = sum(i.file_size for _, i in todo)
    print(f"fetching {len(todo)} members, {total / 1e6:.1f} MB ({len(chosen) - len(todo)} already present)", flush=True)
    t0 = time.time(); done = 0; k = 0
    # Members are fetched in batches of one contiguous byte span each (small files such as the meta JSONs sit next to
    # each other in the archive), so the request count is tied to bytes, not to the number of files.
    todo.sort(key=lambda t: t[1].header_offset); batches = []; cur = []
    for rel, info in todo:
        end = info.header_offset + 30 + len(info.filename.encode("utf-8")) + 64 + info.compress_size
        if cur and (end - cur[0][1].header_offset > SPAN or info.header_offset - cur[-1][2] > GAP): batches.append(cur); cur = []
        cur.append((rel, info, end))
    if cur: batches.append(cur)
    print(f"{len(batches)} range requests", flush=True)
    for batch in batches:
        start = batch[0][1].header_offset; span = f.read_span(start, batch[-1][2] - start)
        for rel, info, _ in batch:
            o = info.header_offset - start; lh = span[o:o + 30]
            if lh[:4] != b"PK\x03\x04": raise RuntimeError(f"bad local header for {info.filename}")
            n_name, n_extra = int.from_bytes(lh[26:28], "little"), int.from_bytes(lh[28:30], "little")
            raw = span[o + 30 + n_name + n_extra:o + 30 + n_name + n_extra + info.compress_size]
            if len(raw) != info.compress_size: raw = f.read_span(start + o + 30 + n_name + n_extra, info.compress_size)
            if info.compress_type == zipfile.ZIP_DEFLATED:
                import zlib; data = zlib.decompress(raw, -15)
            elif info.compress_type == zipfile.ZIP_STORED: data = raw
            else: raise RuntimeError(f"unsupported compression {info.compress_type} for {info.filename}")
            if len(data) != info.file_size: raise RuntimeError(f"size mismatch for {info.filename}")
            p = out / rel; p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(p.suffix + ".part"); tmp.write_bytes(data); tmp.rename(p)
            done += len(data); k += 1
        el = time.time() - t0; print(f"  {k}/{len(todo)} {done / 1e6:.1f} MB {done / el / 1e3:.0f} KB/s eta {(total - done) / max(done / el, 1) / 60:.0f} min", flush=True)
    manifest = dict(record=a.record, design=a.design, url=f.url, seed=a.seed, n_genuine=a.n_genuine, n_per_family=a.n_per_family,
                    members=sorted(str(r) for r, _ in chosen))
    (out / "fetch_manifest.json").write_text(json.dumps(manifest, indent=1)); print("done", flush=True)


if __name__ == "__main__":
    main()
