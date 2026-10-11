"""Authorized, cross-fitted support products, independent of legacy preprocessing.

Membership is frozen by the manifest. Builds capture current authorized bytes in
their own provenance, so isolation experiments can reuse membership after source
mutations. Whole annotation-file hashes are archival metadata, not bank inputs.
External blanks require specimen_root/provenance.json (external_blank_v1) with a
specimens mapping of design IDs to path, sha256, source and independent=true.
The complete records inventory is metadata-only specimen-exclusion authority.
Declared image/identity/source/donor/component IDs and supplied raw_sha256 metadata
must not overlap that inventory. No excluded image bytes are read to establish it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .records import provenance_components
from .rectify import CANVAS_H, CANVAS_W, to_canvas
from .references import ZONE_DILATE_PX, fold_of
from .register import MIN_INLIERS, N_FEATURES, RANSAC_PX, RATIO, _features, estimate_similarity

SCHEMA_VERSION = "anchor_enrollment_v1"
GEOMETRY_VERSION = "bank_similarity_v1"
BANKS = ("train_half0", "train_half1", "enrollment")


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(root: Path, value: str, dataset: str | None = None) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("missing relative filename")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"unsafe relative path {value!r}")
    if dataset is not None and relative.parts[0] != dataset:
        raise ValueError(f"dataset {dataset} does not match filename {value!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"filename escapes root {value!r}")
    return resolved


def _design_path(design: str) -> Path:
    if not isinstance(design, str) or not design or Path(design).is_absolute():
        raise ValueError(f"invalid design {design!r}")
    if any(part in ("", ".", "..") for part in design.split("/")) or "\\" in design:
        raise ValueError(f"invalid design {design!r}")
    return Path(design)


def _layout(root: Path, contributor: dict) -> tuple[dict, list[list], Path]:
    path = _relative_path(root, contributor["annotation"], contributor["dataset"])
    document = json.loads(path.read_text())
    if contributor["dataset"] == "sidtd":
        expected = contributor["identity_id"].rsplit("_", 1)[-1] + ".jpg"
        metadata = document.get("_via_img_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{contributor['image_id']}: missing VIA metadata")
        entries = [entry for entry in metadata.values() if entry.get("filename") == expected]
        if len(entries) != 1:
            raise ValueError(f"{contributor['image_id']}: expected exactly one genuine filename {expected}")
        entry = entries[0]
    elif contributor["dataset"] == "fantasyid":
        entry = document
        if entry.get("filename", Path(contributor["path"]).name) != Path(contributor["path"]).name:
            raise ValueError(f"{contributor['image_id']}: layout filename mismatch")
    else:
        raise ValueError(f"unsupported genuine layout dataset {contributor['dataset']}")
    regions = entry.get("regions")
    if isinstance(regions, dict):
        regions = list(regions.values())
    if not isinstance(regions, list) or not regions:
        raise ValueError(f"{contributor['image_id']}: empty genuine layout")
    rectangles = []
    for region in regions:
        shape = region.get("shape_attributes", {})
        attributes = region.get("region_attributes", {})
        if attributes.get("region_provenance") == "altered":
            raise ValueError(f"{contributor['image_id']}: altered annotation in genuine layout")
        if shape.get("name") != "rect":
            raise ValueError(f"{contributor['image_id']}: unsupported genuine field coordinates")
        try:
            left, top, width, height = [float(shape[key]) for key in ("x", "y", "width", "height")]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{contributor['image_id']}: malformed field coordinates") from error
        if (not np.isfinite([left, top, width, height]).all() or left < 0 or top < 0 or width <= 0 or height <= 0
                or left + width > contributor["width"] or top + height > contributor["height"]):
            raise ValueError(f"{contributor['image_id']}: field coordinates outside native image")
        rectangles.append([str(attributes.get("field_name", "")), left, top, width, height])
    return entry, rectangles, path


def _contributor(root: Path, metadata: dict) -> tuple[dict, list[list], Path]:
    path = _relative_path(root, metadata["path"], metadata["dataset"])
    entry, rectangles, annotation = _layout(root, metadata)
    current = {key: metadata[key] for key in ("image_id", "dataset", "design_id", "identity_id", "component_id",
                                            "path", "annotation", "width", "height")}
    current["raw_sha256"] = _file_hash(path)
    current["layout_sha256"] = _digest(entry)
    return current, rectangles, annotation


def _source() -> dict:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                                  check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {"revision": revision, "implementation_sha256": _file_hash(Path(__file__)),
            "helpers": {name: _file_hash(Path(__file__).with_name(name))
                        for name in ("records.py", "rectify.py", "references.py", "register.py")}}


def _table_hash(frame: pd.DataFrame) -> str:
    return _digest(frame.fillna("").sort_values("image_id").reindex(sorted(frame.columns), axis=1).to_dict("records"))


def make_enrollment_manifest(records: pd.DataFrame, split: pd.DataFrame, fold: int, raw_root: Path) -> dict:
    """Audit official membership and authorize genuine training support only."""
    root = Path(raw_root).resolve(strict=True)
    if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < 10:
        raise ValueError("official fold must be an integer in 0..9")
    required = {"image_id", "dataset", "design_id", "identity_id", "source_id", "donor_id", "component_id",
                "path", "annotation", "width", "height", "label"}
    if required - set(records) or {"image_id", "role", "component_id", "design_id", "label", "protocol"} - set(split):
        raise ValueError("records/split missing required protocol, provenance or filename columns")
    if not {"ref_group", "fold"} & set(split):
        raise ValueError("official split requires explicit ref_group or fold binding")
    for frame in (records, split):
        if frame.image_id.isna().any() or not frame.image_id.is_unique or not len(frame):
            raise ValueError("records/split image IDs must be unique and nonempty")
    declared_roots = [records.attrs["root"]] if "root" in records.attrs else []
    if "root" in records:
        declared_roots.extend(records["root"].dropna().unique())
    if any(Path(str(declared)).resolve() != root for declared in declared_roots):
        raise ValueError("records.root differs from raw_root")
    record_frame = records.fillna("").copy()
    split_frame = split.fillna("").copy()
    for key in ("identity_id", "component_id"):
        if not record_frame[key].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f"invalid {key} in provenance graph")
    for key in ("source_id", "donor_id"):
        if not record_frame[key].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"invalid {key} in provenance graph")
    graph = provenance_components(record_frame)
    if ((graph.groupby(record_frame.component_id).nunique() != 1).any()
            or (record_frame.component_id.groupby(graph).nunique() != 1).any()):
        raise ValueError("frozen component IDs do not match source/donor provenance graph")
    if not set(split_frame.role) <= {"train", "val", "test"}:
        raise ValueError("official split roles must be train/val/test")
    if set(split_frame.protocol) != {"A"}:
        raise ValueError("enrollment requires an official Protocol A split")
    if "ref_group" in split_frame and set(split_frame.ref_group) != {f"it{fold}"}:
        raise ValueError("official split ref_group differs from requested fold")
    if "fold" in split_frame and not (split_frame["fold"] == fold).all():
        raise ValueError("official split fold differs from requested fold")
    if not set(split_frame.image_id) <= set(record_frame.image_id):
        raise ValueError("split image IDs missing from records")
    indexed = record_frame.set_index("image_id")
    for row in split_frame.itertuples(index=False):
        record = indexed.loc[row.image_id]
        if any(record[key] != getattr(row, key) for key in ("component_id", "design_id", "label")):
            raise ValueError(f"{row.image_id}: records/split membership discrepancy")
    if set(record_frame[record_frame.dataset == "sidtd"].image_id) != set(split_frame.image_id):
        raise ValueError("official split must retain every SIDTD classifier image")
    for row in record_frame.itertuples(index=False):
        if row.label not in (0, 1) or row.dataset not in ("sidtd", "fantasyid", "idnet"):
            raise ValueError(f"{row.image_id}: invalid dataset/label")
        _design_path(row.design_id)
        if row.design_id.split("/", 1)[0] != row.dataset:
            raise ValueError(f"{row.image_id}: design/dataset mismatch")
        _relative_path(root, row.path, row.dataset)
        if row.annotation:
            _relative_path(root, row.annotation, row.dataset)
        if row.label == 0 and row.dataset == "sidtd" and Path(row.path).stem != row.identity_id.split("/")[-1]:
            raise ValueError(f"{row.image_id}: genuine filename/identity mismatch")
    inventory_columns = ["image_id", "dataset", "design_id", "identity_id", "source_id", "donor_id",
                         "component_id", "path", "annotation", "label"]
    if "raw_sha256" in record_frame:
        inventory_columns.append("raw_sha256")
        for value in record_frame.raw_sha256:
            if (not isinstance(value, str)
                    or (value and (len(value) != 64 or any(character not in "0123456789abcdef" for character in value)))):
                raise ValueError("invalid authoritative raw_sha256 metadata")
    roles = split_frame.set_index("image_id").role.to_dict()
    inventory_entries = []
    for row in record_frame.sort_values("image_id").to_dict("records"):
        entry = {key: row[key] for key in inventory_columns}
        entry["role"] = roles.get(row["image_id"], row.get("role") or "inventory_only")
        inventory_entries.append(entry)
    inventory = {"schema_version": "dataset_inventory_v1", "purpose": "specimen_exclusion_only_not_bank_dependency",
                 "records_sha256": _table_hash(record_frame), "entries": inventory_entries,
                 "entries_sha256": _digest(inventory_entries)}
    forbidden = set(split_frame.loc[split_frame.role.isin(["val", "test"]), "component_id"])
    training = set(split_frame.loc[split_frame.role == "train", "image_id"])
    eligible = record_frame[(record_frame.label == 0) & record_frame.image_id.isin(training)
                            & ~record_frame.component_id.isin(forbidden)].sort_values("image_id")
    contributors, archives, banks = {}, {}, {}
    for row in eligible.to_dict("records"):
        if (not np.isfinite([row["width"], row["height"]]).all()
                or row["width"] <= 0 or row["height"] <= 0
                or int(row["width"]) != row["width"] or int(row["height"]) != row["height"]):
            raise ValueError(f"{row['image_id']}: invalid native coordinates")
        current, _, annotation = _contributor(root, row)
        contributors[row["image_id"]] = current
        archives[row["annotation"]] = _file_hash(annotation)
    for design in sorted(split_frame.design_id.unique()):
        support = eligible[eligible.design_id == design]
        banks[design] = {}
        for bank in BANKS:
            members = support if bank == "enrollment" else support[support.component_id.map(fold_of) == int(bank[-1])]
            if not len(members):
                raise ValueError(f"design {design}, bank {bank}: empty authorized support")
            banks[design][bank] = {"bank": bank, "design_id": design, "fold": fold,
                                   "geometry_version": GEOMETRY_VERSION, "support_ids": members.image_id.tolist(),
                                   "component_ids": sorted(members.component_id.unique())}
    return {"schema_version": SCHEMA_VERSION, "geometry_version": GEOMETRY_VERSION, "fold": fold,
            "raw_root": str(root), "official_split_sha256": _table_hash(split_frame),
            "official_fold_binding": {"protocol": "A", "fold": fold, "ref_group": f"it{fold}"},
            "dataset_inventory": inventory,
            "official_split_hash_format": "canonical_json_rows_v1",
            "provenance_graph_sha256": _table_hash(record_frame[["image_id", "identity_id", "source_id", "donor_id", "component_id"]]),
            "classifier_counts": {role: int((split_frame.role == role).sum()) for role in ("train", "val", "test")},
            "forbidden_components": sorted(forbidden), "eligible_ids": eligible.image_id.tolist(),
            "contributors": contributors, "archive_metadata": archives, "banks": banks, "source": _source()}


def resolve_enrollment_bank(manifest: dict, design_id: str, role: str, training_component: str | None = None) -> dict:
    """Resolve evaluation by design/fold only; train uses the opposite component half."""
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("geometry_version") != GEOMETRY_VERSION:
        raise ValueError("unsupported enrollment schema/geometry version")
    if role == "train":
        if not isinstance(training_component, str) or not training_component:
            raise ValueError("train lookup requires a training component")
        bank = f"train_half{1 - fold_of(training_component)}"
    elif role in ("val", "test", "evaluation"):
        if training_component is not None:
            raise ValueError("evaluation bank lookup forbids query component provenance")
        bank = "enrollment"
    else:
        raise ValueError(f"unknown enrollment role {role!r}")
    try:
        selected = manifest["banks"][design_id][bank]
    except KeyError as error:
        raise ValueError(f"missing design/bank {design_id}/{bank}; no global fallback") from error
    if (not selected.get("support_ids") or selected.get("fold") != manifest["fold"]
            or selected.get("geometry_version") != GEOMETRY_VERSION):
        raise ValueError(f"empty or inconsistent support bank {design_id}/{bank}")
    if role == "train" and training_component in selected["component_ids"]:
        raise ValueError("training component present in its selected support bank")
    return copy.deepcopy(selected)


def _valid_median(images: list[np.ndarray], valid: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not images or len(images) != len(valid):
        raise ValueError("empty or inconsistent median support")
    stack = np.stack(images).astype(np.float32)
    observations = np.stack(valid) > 0
    stack[~observations] = np.nan
    coverage = np.minimum(observations.sum(axis=0), 255).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
    return np.nan_to_num(median, nan=0).astype(np.uint8), coverage


def _similarity(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("invalid similarity transform")
    linear = matrix[:2, :2]
    gram = linear.T @ linear
    if np.linalg.det(linear) <= 0 or not np.allclose(gram, np.eye(2) * gram[0, 0], atol=1e-6, rtol=1e-6):
        raise ValueError("invalid similarity transform axes")
    return matrix


def _polygon_mask(rectangles: list[list], transform: np.ndarray) -> np.ndarray:
    matrix = _similarity(transform)
    mask = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    for _, left, top, width, height in rectangles:
        vertices = np.array([[left, top, 1], [left + width, top, 1],
                             [left + width, top + height, 1], [left, top + height, 1]], dtype=np.float64)
        mapped = (matrix @ vertices.T).T[:, :2]
        if not np.isfinite(mapped).all() or np.max(np.abs(mapped)) > 1e7:
            raise ValueError("invalid transformed polygon coordinates")
        cv2.fillPoly(mask, [np.round(mapped * 256).astype(np.int32)], 255, shift=8)
    return mask


def _warp(image: np.ndarray, transform: np.ndarray, mask: bool = False) -> np.ndarray:
    return cv2.warpAffine(image, _similarity(transform)[:2], (CANVAS_W, CANVAS_H),
                          flags=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _fit(source, target, seed: int) -> tuple[np.ndarray | None, int]:
    cv2.setRNGSeed(seed)
    transform, inliers = estimate_similarity(source, target)
    if transform is None or inliers < MIN_INLIERS:
        return None, int(inliers)
    try:
        return _similarity(transform), int(inliers)
    except ValueError:
        return None, int(inliers)


def _seed(fold: int, design: str, bank: str, stage: str, image_id: str = "") -> int:
    return int(_digest([GEOMETRY_VERSION, fold, design, bank, stage, image_id])[:8], 16) & 0x7fffffff


def _write_image(path: Path, image: np.ndarray) -> dict:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"cannot write product {path}")
    return {"path": str(path), "sha256": _file_hash(path)}


def _write_json(path: Path, value) -> dict:
    path.write_bytes(_json_bytes(value))
    return {"path": str(path), "sha256": _file_hash(path)}


def _external_blanks(specimen_root: Path | None, designs, inventory: dict | None, raw_root: Path) -> dict:
    if specimen_root is None:
        return {}
    if (not isinstance(inventory, dict) or inventory.get("schema_version") != "dataset_inventory_v1"
            or not inventory.get("entries") or inventory.get("entries_sha256") != _digest(inventory["entries"])):
        raise ValueError("external blank requires frozen complete dataset inventory authority")
    entries = inventory["entries"]
    dataset_paths = {_relative_path(raw_root, entry["path"], entry["dataset"]) for entry in entries}
    dataset_file_ids = set()
    for dataset_path in dataset_paths:
        try:
            metadata = dataset_path.stat()
        except FileNotFoundError:
            continue
        dataset_file_ids.add((metadata.st_dev, metadata.st_ino))
    identifier_keys = ("image_id", "identity_id", "source_id", "donor_id", "component_id")
    identifiers = {entry[key] for entry in entries for key in identifier_keys if entry[key]}
    raw_hashes = {entry["raw_sha256"] for entry in entries if entry.get("raw_sha256")}
    root = Path(specimen_root).resolve(strict=True)
    document = json.loads((root / "provenance.json").read_text())
    if document.get("schema_version") != "external_blank_v1":
        raise ValueError("external blank requires declared external_blank_v1 provenance")
    blanks = {}
    for design in designs:
        declaration = document.get("specimens", {}).get(design)
        if not declaration or declaration.get("independent") is not True or not declaration.get("source"):
            raise ValueError(f"{design}: missing independent blank provenance")
        path = _relative_path(root, declaration["path"])
        if (path in dataset_paths or any(declaration.get(key) in identifiers for key in identifier_keys)
                or declaration.get("sha256") in raw_hashes):
            raise ValueError(f"{design}: external blank overlaps authoritative dataset inventory; not independent")
        metadata = path.stat()
        if (metadata.st_dev, metadata.st_ino) in dataset_file_ids:
            raise ValueError(f"{design}: external blank aliases authoritative dataset inventory; not independent")
        if _file_hash(path) != declaration.get("sha256"):
            raise ValueError(f"{design}: external blank source/hash is not immutable and independent")
        blanks[design] = dict(declaration, resolved_path=str(path), provenance_path=str(root / "provenance.json"),
                             declaration_sha256=_digest(declaration))
    return blanks


def _build_bank(manifest: dict, selected: dict, root: Path, directory: Path, blank: dict | None, archives: dict) -> dict:
    design, bank, fold = selected["design_id"], selected["bank"], selected["fold"]
    directory.mkdir(parents=True)
    images, validity, native_transforms, features, layouts, contributors = {}, {}, {}, {}, {}, {}
    for image_id in selected["support_ids"]:
        metadata = manifest["contributors"][image_id]
        current, rectangles, annotation = _contributor(root, metadata)
        archives[metadata["annotation"]] = _file_hash(annotation)
        contributors[image_id], layouts[image_id] = current, rectangles
        image = cv2.imread(str(_relative_path(root, metadata["path"], metadata["dataset"])), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (metadata["height"], metadata["width"]):
            raise ValueError(f"{image_id}: unreadable raw image or native coordinate dimensions mismatch")
        images[image_id], validity[image_id], native_transforms[image_id] = to_canvas(image)
        features[image_id] = _features(images[image_id])
    anchor = selected["support_ids"][0]
    first_images, first_valid, registrations = [], [], {}
    for image_id in selected["support_ids"]:
        if image_id == anchor:
            transform, inliers = np.eye(3), None
        else:
            transform, inliers = _fit(features[image_id], features[anchor], _seed(fold, design, bank, "anchor", image_id))
        registrations[image_id] = {"anchor_applied": transform is not None, "anchor_inliers": inliers,
                                    "anchor_transform": None if transform is None else transform.tolist()}
        if transform is not None:
            first_images.append(_warp(images[image_id], transform))
            first_valid.append(_warp(validity[image_id], transform, mask=True))
    target, target_coverage = _valid_median(first_images, first_valid)
    target_valid = (target_coverage > 0).astype(np.uint8) * 255
    target_features = _features(target)
    final_images, final_valid, used_ids = [], [], []
    zone = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    for image_id in selected["support_ids"]:
        transform, inliers = _fit(features[image_id], target_features, _seed(fold, design, bank, "target", image_id))
        registration = registrations[image_id]
        registration.update(target_applied=transform is not None, target_inliers=inliers,
                            canvas_to_bank=None if transform is None else transform.tolist(),
                            native_to_bank=None if transform is None else (transform @ native_transforms[image_id]).tolist())
        if transform is None:
            registration["failure"] = "insufficient_inliers_or_invalid_similarity"
            continue
        valid = cv2.bitwise_and(_warp(validity[image_id], transform, mask=True), target_valid)
        if not valid.any():
            registration.update(target_applied=False, failure="empty_valid_overlap", canvas_to_bank=None, native_to_bank=None)
            continue
        final_images.append(_warp(images[image_id], transform))
        final_valid.append(valid)
        used_ids.append(image_id)
        zone |= cv2.bitwise_and(_polygon_mask(layouts[image_id], transform @ native_transforms[image_id]), valid)
    if not used_ids:
        raise ValueError(f"{design}/{bank}: empty successfully registered support; context unavailable")
    median, coverage = _valid_median(final_images, final_valid)
    valid = (coverage > 0).astype(np.uint8) * 255
    zone = cv2.dilate(zone, np.ones((2 * ZONE_DILATE_PX + 1,) * 2, np.uint8))
    zone = cv2.bitwise_and(zone, valid)
    products = {name: _write_image(directory / f"{name}.png", image) for name, image in (
        ("target", target), ("target_coverage", target_coverage), ("target_valid", target_valid),
        ("median", median), ("coverage", coverage), ("valid", valid), ("zone", zone))}
    geometry = {"schema_version": SCHEMA_VERSION, "geometry_version": GEOMETRY_VERSION,
                "design_id": design, "fold": fold, "bank": bank, "anchor_id": anchor,
                "axes": "x=column,y=row; native_to_bank=canvas_to_bank@native_to_canvas",
                "canvas_hw": [CANVAS_H, CANVAS_W], "support_ids": selected["support_ids"],
                "registered_ids": used_ids, "registrations": registrations,
                "valid_interpolation": "nearest", "rgb_interpolation": "linear", "zone_dilate_px": ZONE_DILATE_PX,
                "registration": {"min_inliers": MIN_INLIERS, "n_features": N_FEATURES, "ratio": RATIO,
                                 "ransac_px": RANSAC_PX, "seed": _seed(fold, design, bank, "bank"),
                                 "seed_derivation": "sha256(version,fold,design,bank,stage,image_id) low31",
                                 "opencv_version": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
                                 "opencl": False},
                "contributors": contributors, "source": manifest["source"],
                "official_split_sha256": manifest["official_split_sha256"]}
    if blank is not None:
        path = Path(blank["resolved_path"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{design}: unreadable external blank")
        specimen, specimen_valid, letterbox = to_canvas(image)
        transform, inliers = _fit(_features(specimen), target_features, _seed(fold, design, bank, "specimen"))
        if transform is None:
            raise ValueError(f"{design}/{bank}: specimen registration failed; context unavailable ({inliers} inliers)")
        overlap = cv2.bitwise_and(_warp(specimen_valid, transform, mask=True), valid)
        if not overlap.any():
            raise ValueError(f"{design}/{bank}: specimen has empty valid overlap")
        products["specimen"] = _write_image(directory / "specimen.png", _warp(specimen, transform))
        products["specimen_valid"] = _write_image(directory / "specimen_valid.png", overlap)
        geometry["specimen"] = {"provenance": blank, "native_to_bank": (transform @ letterbox).tolist(),
                                "canvas_to_bank": transform.tolist(), "inliers": inliers,
                                "alignment_target": "target.png"}
    products["geometry"] = _write_json(directory / "geometry.json", geometry)
    dependency = dict(geometry, output_sha256={name: item["sha256"] for name, item in products.items()})
    products["manifest"] = _write_json(directory / "manifest.json", dependency)
    return dict(selected, contributors=contributors, registered_ids=used_ids, products=products,
                counts={"authorized": len(selected["support_ids"]), "registered": len(used_ids),
                        "failed": len(selected["support_ids"]) - len(used_ids)})


def build_enrollment_products(manifest: dict, raw_root: Path, output_root: Path, specimen_root: Path | None = None) -> dict:
    """Build every named bank independently, without opening any query pixels."""
    root, output = Path(raw_root).resolve(strict=True), Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new versioned directory: {output}")
    if str(root) != manifest.get("raw_root"):
        raise ValueError("raw_root differs from frozen enrollment root")
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output must be separate from raw_root")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("geometry_version") != GEOMETRY_VERSION:
        raise ValueError("unsupported enrollment manifest version")
    if manifest.get("official_fold_binding") != {"protocol": "A", "fold": manifest["fold"],
                                                "ref_group": f"it{manifest['fold']}"}:
        raise ValueError("missing or inconsistent official fold binding")
    if not manifest.get("banks") or not manifest.get("eligible_ids"):
        raise ValueError("empty authorized support")
    authorized = set(manifest["eligible_ids"])
    forbidden = set(manifest["forbidden_components"])
    if set(manifest["contributors"]) != authorized:
        raise ValueError("contributor IDs differ from frozen eligible support")
    for design, banks in manifest["banks"].items():
        _design_path(design)
        if set(banks) != set(BANKS):
            raise ValueError(f"{design}: missing named support bank")
        design_ids = {image_id for image_id, row in manifest["contributors"].items() if row["design_id"] == design}
        for bank, selected in banks.items():
            members = selected.get("support_ids", [])
            if not members or members != sorted(set(members)) or not set(members) <= authorized:
                raise ValueError(f"{design}/{bank}: empty or unauthorized support")
            expected = design_ids if bank == "enrollment" else {
                image_id for image_id in design_ids if fold_of(manifest["contributors"][image_id]["component_id"]) == int(bank[-1])}
            if set(members) != expected or selected.get("bank") != bank or selected.get("design_id") != design:
                raise ValueError(f"{design}/{bank}: membership differs from frozen authorized bank")
            components = {manifest["contributors"][image_id]["component_id"] for image_id in members}
            if (components & forbidden or components != set(selected.get("component_ids", []))
                    or selected.get("fold") != manifest["fold"] or selected.get("geometry_version") != GEOMETRY_VERSION):
                raise ValueError(f"{design}/{bank}: forbidden or inconsistent support components")
    blanks = _external_blanks(specimen_root, manifest["banks"], manifest.get("dataset_inventory"), root)
    build_manifest = copy.deepcopy(manifest)
    current_source = _source()
    frozen_source = manifest["source"]
    if any(frozen_source.get(key) != current_source.get(key) for key in ("implementation_sha256", "helpers")):
        build_manifest["source"] = current_source
    previous_threads, previous_opencl = cv2.getNumThreads(), cv2.ocl.useOpenCL()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    output.mkdir(parents=True, exist_ok=False)
    built, archives = {}, {}
    try:
        for design in sorted(manifest["banks"]):
            built[design] = {}
            for bank in BANKS:
                directory = output / f"fold{manifest['fold']}" / _design_path(design) / bank
                built[design][bank] = _build_bank(build_manifest, manifest["banks"][design][bank], root, directory,
                                                  blanks.get(design), archives)
        result = copy.deepcopy(build_manifest)
        result.update(banks=built, archive_metadata=archives, output_root=str(output),
                      counts={"designs": len(built), "banks": len(built) * len(BANKS), "eligible": len(authorized)})
        result["contributors"] = {image_id: contributor for banks in built.values()
                                  for image_id, contributor in banks["enrollment"]["contributors"].items()}
        result["manifest_creation_source"] = manifest["source"]
        result["execution_source"] = current_source
        _write_json(output / "archive_metadata.json", {"purpose": "archival_only_not_bank_dependency",
                                                      "annotation_files": archives, "execution_source": current_source})
        _write_json(output / "manifest.json", result)
        return result
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)
