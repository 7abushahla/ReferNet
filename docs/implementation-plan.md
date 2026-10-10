# Implementation plan: training stack (10 October 2026)

Status before this plan: data stage done and tested (`scripts/prepare_data.py`); every other module is a stub. This plan fixes the interfaces so that the modules can be written in parallel and assembled into one smoke test, then into the Protocol A sweep. Conventions come from `docs/conventions.md`; the reference findings of the Codex review (10 Oct) are folded in where they touch Protocol A.

## 0. Tensor and file conventions (binding)

- Canvas 512 x 800 BGR JPEG under `data/processed/canvases/<image_id>.jpg`; `valid/<image_id>.png` exists only when padded or warped (255 = valid); `masks/<image_id>.png` for altered images (255 = altered); `zones/<design_tag>.png` (255 = field); references `references/<design_tag>_fold{0,1}.jpg`, `_all.jpg`, `_specimen.jpg` (design_tag = design_id with `/` replaced by `_`).
- Model input: float tensor `(B, 8, H, W)` = `[query RGB (3), query zone (1), reference RGB (3), reference zone (1)]`, RGB order, ImageNet mean and std on RGB channels, zone in {0, 1} unchanged. H, W = 512, 800 in training (random crop 448 x 704 allowed in training; full canvas at test).
- Label convention inside the code: `label = 1` means altered. Export at evaluation: `score_bonafide = 1 - sigmoid(logit)` so that 1 = genuine (DeepID convention).
- Model output: dict `{"logit": (B,), "map": (B, 1, H/4, W/4)}`; the map is upsampled to full size by the loss and by the evaluator.
- Ground-truth mask tensor `(B, 1, H, W)` float in {0, 1}; genuine images have an all-zero mask. Valid-region tensor `(B, 1, H, W)` in {0, 1} masks the loss and the localization metrics.
- Metrics file: `results/runs/<tag>/metrics.json` written last and only on success; schema in Section 5.

## 1. `data/pairs.py`

```python
def build_pairs(split_df, records_df, processed_root, reference="median", wrong_design_fraction=0.1,
                role="train", seed=0) -> list[dict]
```
One dict per sample: `image_id, design_id, path (canvas), valid_path or None, mask_path or None, zone_path, ref_path, label (0/1), subset, attack_family, component_id, wrong_design (bool)`.
- `reference="median"`: `references/<tag>_fold{other}.jpg` where `other = 1 - fold_of_component(component_id)`; raise if the file is missing (no `_all` fallback). `reference="specimen"`: `references/<tag>_specimen.jpg`, raise if missing. `reference="single"`: one genuine canvas of the same design from another component, drawn once per sample with `seed`.
- Wrong-design negatives (train only): for a fraction of genuine samples, replace `ref_path` by the reference of another design, set `label=1`, `mask_path=None` with an all-ones mask flag `wrong_design=True` (the map target is the full valid region).

## 2. `data/dataset.py`

```python
class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, train, crop=(448, 704), jpeg=(60, 100), affine_px=8, jitter_px=2, jitter_p=0.25, photometric=True, equalize_qf=None)
    def __getitem__(self, i) -> dict(x=(8,H,W) float, mask=(1,H,W) float, valid=(1,H,W) float, label=float, meta=dict)
```
- Geometry drawn once per pair (crop, small affine up to `affine_px`), applied to query, reference, zone, mask, valid. Alignment jitter (query only, up to `jitter_px`, probability `jitter_p`).
- Photometric jitter and JPEG re-encoding drawn independently for query and reference (cv2 encode/decode).
- `equalize_qf=75` at test: both inputs re-encoded at that quality (the "equalized" column).
- No flips. Test: no crop, no augmentation.
- Zone channel from `zone_path`; reference zone channel is the same mask.

## 3. `models/`

`models/build.py`:
```python
def build_model(arch="siamese_r18", fusion="difference", init="imagenet", in_channels=4) -> nn.Module
```
- `arch="siamese_r18"` (plain PyTorch, torchvision ResNet-18, stem widened to `in_channels` with the extra channel zero-initialised; shared encoder for query and reference; features at strides 4, 8, 16, 32).
- `fusion`: `single` (query branch only; reference channels ignored, same capacity), `stacked` (query and reference concatenated as 8 channels into one encoder), `difference` (`|f_q - f_r|` concatenated with `f_q` per stage), `attention` (cross-attention from query to reference features at stride 16 and 32, 6 heads, key dim 32, zone used as additive logit bias). Implemented in `models/fusion.py`.
- Decoder: per-stage 1x1 to 128, bilinear to stride 4, concat, 3x3 conv, 1x1 to one map logit (`models/heads.py: MapHead`). `ReferralHead`: GAP of the deepest fused feature concatenated with `max` of the map logit, two dense layers, one logit.
- `arch="changerex_r18"` reserved for the Open-CD adapter (not built in this pass; raise with a clear message).
- `models/build.py` also exposes `count_params(model)`.

## 4. `losses.py`, `metrics.py`, `policy_metrics.py`

- `referral_loss(logit, map_logit, label, mask, valid, mask_weight=3.0)` = BCE(logit, label) + mask_weight * (BCE + Dice) on the map inside `valid`, map upsampled to mask size. Wrong-design samples use the valid region as the mask target.
- `metrics.py` (pure numpy/sklearn, bona fide score high):
  `auc, eer, far_threshold(bf_scores_bonafide, bf_scores_attack, target_apcer)` (bob rule: lowest acceptance threshold whose realised APCER is at most the target; ties accepted as bona fide), `apcer_bpcer(scores, labels, thr)`, `hter`, `bpcer_at_apcer(scores, labels, [0.01, 0.05, 0.10])` on the evaluation set (descriptive), `transported(scores, labels, thr_from_source)`, `deepid_f1(scores, labels)` at 0.5, `localization_f1(pred_maps, gt_masks, valid)` with DeepID polarity, `false_positive_map_area(pred_maps on genuine, zones)` inside zones / outside zones.
- `policy_metrics.py` keeps its altered-high convention; `evaluate.py` converts explicitly.

## 5. `train.py` and `evaluate.py`

`python -m refer_or_clear.train --config configs/protocol_a.yaml --arch siamese_r18 --fusion difference --reference median --init imagenet --seed 42 --data-root data/processed --run-dir results/runs/<tag> [--epochs N --limit K --device cpu|cuda --amp]`
- Seeds everything; AdamW (pretrained 1e-4, new 1e-3, wd 0.05), linear warm-up 500 iterations then cosine; AMP on CUDA; batch from config; early stopping on validation AUC (patience 8); best checkpoint `checkpoints/<tag>/best.pt`; per-epoch CSV log `<run-dir>/log.csv` (epoch, train loss, val loss, val AUC, lr, seconds); `train.json` with the resolved configuration, git hash, parameter count, device.
- On finish calls `evaluate.run(...)` for the test split(s) and writes `metrics.json`:
```json
{"tag":..., "protocol":"A", "arch":..., "fusion":..., "reference":..., "seed":42, "n_params":...,
 "val": {"auc":..., "threshold_miss5": thr, "threshold_miss1": thr},
 "test": {"native": {...}, "equalized_qf75": {...}},
 "subsets": {"<subset>": {"native": {...}}},
 "timing": {"train_seconds":..., "epochs":...}}
```
each `{...}` = `auc, eer, deepid_acc, deepid_f1, bpcer_at_apcer_1/5/10, transported_miss5, transported_referral5, transported_miss1, transported_referral1, hter5, loc_f1, fp_area_in_zone, fp_area_out_zone, n_genuine, n_altered`.
- `evaluate.py` is also runnable alone on a checkpoint.
- `--limit K` subsamples K pairs per split for smoke tests.

## 6. Smoke test (`scripts/smoke_test.sh`)

CPU on the Mac: `--limit 16 --epochs 1 --device cpu`, batch 2, crop 256 x 400; asserts a finite loss, a backward pass, a metrics.json. GPU on the 3080 (`aric@192.168.1.202`): full 512 x 800, batch 8, `--limit 64 --epochs 2`, prints peak memory. Then the cluster `verify_gpu.sh` runs the same.

## 7. Reference-isolation fixes for Protocol A (from the Codex review)

- `references.py`: fold by provenance component (`fold_of(component_id)`), not identity; validity-aware median (invalid pixels excluded, written coverage map); `reference_path` raises instead of falling back to `_all.jpg`; specimen registered to the fold median of the support, written per fold.
- `prepare_data.py`: registration first-pass anchor restricted to genuine images; references built per fold from the fold's own support (already the case for the RGB median; the registration target is shared and documented as such).

## 8. Compute plan (Protocol A first)

See `docs/compute-plan.md`.

## 9. ChangerEx arm (built and tested 10 October 2026)

`src/refer_or_clear/models/changerex.py` wraps the two Open-CD modules of `changer_ex_r18_512x512_40k_levircd.py` as a plain `nn.Module` with the interface of Section 0. No mmengine Runner, no registered dataset, no edits to Open-CD. `IA_ResNetV1c(depth=18, in_channels=4, interaction_cfg=(None, SpatialExchange(1/2), ChannelExchange(1/2), ChannelExchange(1/2)), norm_cfg=BN, contract_dilation=True)` is the shared two-image backbone (per stage it returns `cat([f_query, f_ref])` at strides 4, 8, 16, 32). A one-line subclass of the `Changer` head (`in_channels=[64,128,256,512], channels=128, num_classes=2, out_channels=1`) returns the 128-channel fused feature after FDAF and MixFFN together with the one-channel stride-4 map logit. `ReferralHead` is GAP of that feature concatenated with `max` of the map logit, two dense layers (129 to 128 to 1). Output `{"logit": (B,), "map": (B, 1, H/4, W/4)}`.

- Entry point `build_changerex(fusion, init, in_channels=4, ckpt_path=None)`. `models/build.py` should dispatch `arch == "changerex_r18"` to it (not done here; build.py belongs to the other pass).
- `fusion="difference"` is stock ChangerEx. `fusion="single"` feeds `x[:, :4]` to both branches, so the reference channels are never read; it is the matched single-image control for this arm (same parameters, same capacity, same exchange and FDAF modules). Verified: the map is bit-identical when the reference channels are replaced by noise.
- `init="scratch"` applies Open-CD's own `init_cfg` through `init_weights()` (Kaiming convs, zero-init residual BN, N(0, 0.01) `conv_seg`), which mmengine would otherwise only run inside a Runner. `init="levir"` loads `checkpoints/pretrained/ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth` (override with `ROC_LEVIR_CKPT`) with `strict=False` after two surgeries. The stem `backbone.stem.0.weight` is zero-padded from (32, 3, 3, 3) to (32, 4, 3, 3), so the zone channel contributes nothing at step 0. The two-class `conv_seg` is folded to one logit as `w[1] - w[0]`, `b[1] - b[0]`, which equals `logit[:, 1] - logit[:, 0]` of the released softmax head (verified to 1.9e-5 in float32). The load prints the matched-tensor count and the missing and unexpected keys; expected is 173 matched, 4 missing (the referral head), 0 unexpected.
- The map logit is normalized nowhere inside the model. The LEVIR head produces logits around -14 on random input (confident "unchanged"), so the map loss of Section 4 starts from that prior.
- If the Open-CD stack is not importable the module still imports; `build_changerex` raises an `ImportError` that names the exact install recipe below.
- `scripts/cluster/get_levir_ckpt.sh` downloads the checkpoint with `curl -L` from the Google Drive usercontent URL (`confirm=t`) into `checkpoints/pretrained/` and checks 136,880,383 bytes and md5 `7cd73048926efa8f0535317f5dde87e8` (md5 computed from the audited copy); re-running skips a verified file.

Test (lab laptop `aric@192.168.1.202`, CPU only, the NVIDIA driver has an NVML mismatch; conda env `roc`, files copied to `~/roc-test/`, standalone script `test_changerex.py`, input `torch.randn(2, 8, 256, 400)`, train mode, BCE on the logit plus BCE on the map, backward):

| fusion | init | params (total / backbone / head / referral) | loss | stem grad, RGB vs zone | checks |
|---|---|---|---|---|---|
| difference | scratch | 11,407,874 / 11,196,032 / 195,073 / 16,769 | 1.371 | 4.2 / 1.4 | finite, no NaN grads, every parameter has a grad |
| difference | levir | same | 2.133 | 80.8 / 28.0 | 173 matched, 4 missing (referral), 0 unexpected; folded head equals logit1 - logit0 (max 1.9e-5) |
| single | scratch | same | 1.389 | 0.31 / 0.12 | map invariant to the reference channels (max diff 0) |
| single | levir | same | 1.494 | 37.2 / 14.6 | as above, both checks |

Map shape (2, 1, 64, 100). At the training resolution (2, 8, 512, 800) the eval forward takes 0.5 s and train forward plus backward 1.6 s on 20 CPU threads, map (2, 1, 128, 200), finite gradients. The parameter counts are of `nn.Parameter` tensors; the audit's 11,401,853 counts state-dict entries including BN buffers and the 2-class `conv_seg`, so the two figures agree once the 288 zone-channel weights and the referral head are added.

Environment recipe that worked on the laptop (Python 3.10, exact order; every step as a plain `pip install`):

```
conda create -y -n roc python=3.10
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.2 torchvision==0.16.2
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmengine==0.10.4 mmsegmentation==1.2.2 mmdet==3.3.0
git clone https://github.com/likyoo/open-cd.git ~/open-cd && (cd ~/open-cd && pip install -v -e .)   # HEAD 790a1972
pip install opencv-python-headless "numpy<2" pandas scikit-learn pyyaml tqdm
pip install ftfy regex            # mmseg 1.2.2 tokenizer imports them at package import
pip install mmpretrain==1.2.0     # opencd/models/backbones/__init__.py imports vit_sam, which needs mmpretrain
```
Resulting pins: torch 2.1.2+cu121, torchvision 0.16.2+cu121, mmcv 2.1.0 (prebuilt wheel, `mmcv.ops` imports), mmengine 0.10.4, mmsegmentation 1.2.2, mmdet 3.3.0, mmpretrain 1.2.0, opencd 1.1.0, numpy 1.26.4, ftfy 6.3.1, regex 2026.9.29. The mmcv wheel installed without a build.

Differences to fold into `scripts/cluster/setup-env.sh` and `requirements-train.txt` (not edited here): add `mmpretrain==1.2.0`, `ftfy` and `regex` to `requirements-train.txt` (the audit called mmpretrain optional; at Open-CD HEAD it is a hard import of `opencd.models`). Two cosmetic effects of the order: the mmcv step pulls mmengine 0.10.7 and the torch step pulls numpy 2.2.6, both downgraded by the later pinned steps, so `numpy<2` could be installed together with torch to avoid the double install; `pip install -e .` of Open-CD pulls `opencv-python` 5.0 and `opencv-contrib-python` 4.11 next to the headless build, harmless but three OpenCV wheels. `scripts/cluster/get_levir_ckpt.sh` should be called from `stage-data.sh` or before `verify_gpu.sh`.
