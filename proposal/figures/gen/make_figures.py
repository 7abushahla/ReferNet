"""Figures for the forgery-triage proposal.

fig_attacks.{pdf,png}     three fraud examples from the public datasets (reproduced panels)
fig_templates.{pdf,png}   template diversity: IDNet template sheet and FantasyID Arabic design
fig_referral.{pdf,png}    illustrative referral operating-point chart (synthetic score model, labelled as such)

Run with SANA-FE/.venv/bin/python from this folder.
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
OUT = os.path.join(os.path.dirname(__file__), "..")
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})


def load(name):
    im = cv2.imread(os.path.join(SRC, name), cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 4:
        a = im[:, :, 3:4] / 255.0
        im = (im[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def save(fig, stem):
    fig.savefig(os.path.join(OUT, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


# 1. Attack examples -----------------------------------------------------------
morph = load("idnet_morph.png")          # left two cards: genuine and morphed
h, w = morph.shape[:2]
morph_pair = morph[:, : int(w * 0.615)]  # drop the JSON panel
sidtd = load("sidtd_cropreplace.png")
sidtd_core = sidtd[30:, : int(sidtd.shape[1] * 0.76)]  # drop the side callouts, keep real/fake columns
fant = load("fantasy_usa_altered.png")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6), gridspec_kw={"width_ratios": [1.45, 1.35, 1.0]})
for ax, im in zip(axes, [morph_pair, sidtd_core, fant]):
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
fig.subplots_adjust(wspace=0.10, bottom=0.24, top=0.99)
labels = ["(a) IDNet: genuine (left)\nand face-morphed (right)", "(b) SIDTD: real (left) and\ncrop-and-replace forgery (right)", "(c) FantasyID:\naltered fields boxed"]
fig.canvas.draw()
for ax, lab in zip(axes, labels):
    bb = ax.get_position()
    fig.text((bb.x0 + bb.x1) / 2, bb.y0 - 0.03, lab, ha="center", va="top", fontsize=8.5)
save(fig, "fig_attacks")

# 2. Template diversity ----------------------------------------------------------
tmpl = load("idnet_templates.png")
arab = load("fantasy_arabic.png")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.9, 1.0]})
axes[0].imshow(tmpl); axes[1].imshow(arab)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
fig.subplots_adjust(wspace=0.03, bottom=0.1)
fig.canvas.draw()
for ax, lab in zip(axes, ["(a) IDNet: ten US and ten EU document designs", "(b) FantasyID: Arabic design, printed card"]):
    bb = ax.get_position()
    fig.text((bb.x0 + bb.x1) / 2, 0.02, lab, ha="center", va="bottom", fontsize=8.5)
save(fig, "fig_templates")

# 3. Referral operating point (illustrative) --------------------------------------
# Score model: genuine ~ N(0,1); altered ~ N(d,1). Referral rate at threshold t with prevalence p:
#   R(t) = (1-p) * P(genuine > t) + p * P(altered > t);  miss(t) = P(altered < t).
from math import erf, sqrt


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


p = 0.044  # GDRFA hit rate among referred documents, 2025 to H1 2026, used only to set the axis
ts = np.linspace(-3, 6, 600)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for d, lab, col in [(1.5, "weak detector (d'=1.5)", "#9c6b4e"), (2.5, "d'=2.5", "#4a6fa5"), (3.5, "strong detector (d'=3.5)", "#2b2b2b")]:
    miss = np.array([Phi(t - d) for t in ts])
    ref = np.array([(1 - p) * (1 - Phi(t)) + p * (1 - Phi(t - d)) for t in ts])
    ax.plot(miss * 100, ref * 100, color=col, lw=1.6, label=lab)
ax.axhline(100, color="#aaaaaa", lw=0.8, ls="--")
ax.text(7.5, 95.5, "refer everything (today)", fontsize=7.5, color="#666666")
ax.axvline(5, color="#aa4632", lw=0.8, ls=":")
ax.text(5.6, 75, "miss budget 5%", fontsize=7.5, color="#aa4632")
ax.set_xlabel("altered documents missed by the first line (%)")
ax.set_ylabel("documents referred to the second line (%)")
ax.set_xlim(0, 40); ax.set_ylim(0, 105)
ax.grid(alpha=0.25)
ax.legend(loc="center right", fontsize=7.5, frameon=False)
save(fig, "fig_referral")
print("done")


# 4. The idea: evaluation gap, the method, and what is reported ------------------------
def card(name, crop=None):
    im = load(name)
    if crop is not None:
        y0, y1, x0, x1 = crop
        im = im[int(y0 * im.shape[0]):int(y1 * im.shape[0]), int(x0 * im.shape[1]):int(x1 * im.shape[1])]
    return im


wv = card("idnet_morph.png", crop=(0.0, 1.0, 0.0, 0.305))
alb = card("sidtd_cropreplace.png", crop=(0.04, 0.50, 0.0, 0.385))
fin = card("sidtd_fin.png")
usa = card("fantasy_usa_nf.png")
per = card("fantasy_persian.png")
ara = card("fantasy_arabic.png")

# a presented (altered) Arabic card: portrait swapped from the Persian design, plus the year of birth changed
pres = ara.copy()
H, W = ara.shape[:2]
py0, py1, px0, px1 = int(0.21 * H), int(0.84 * H), int(0.035 * W), int(0.345 * W)
src = per[int(0.21 * per.shape[0]):int(0.84 * per.shape[0]), int(0.035 * per.shape[1]):int(0.345 * per.shape[1])]
pres[py0:py1, px0:px1] = cv2.resize(src, (px1 - px0, py1 - py0))
# copy the expiry year digits over the birth year digits (a text edit)
yr_src = ara[int(0.875 * H):int(0.925 * H), int(0.47 * W):int(0.56 * W)]
pres[int(0.615 * H):int(0.665 * H), int(0.47 * W):int(0.56 * W)] = yr_src
# deviation map
diff = np.abs(pres.astype(np.int16) - ara.astype(np.int16)).sum(axis=2).astype(np.float32)
diff = cv2.GaussianBlur(diff, (0, 0), 6)
diff = diff / (diff.max() + 1e-6)
heat = (plt.get_cmap("inferno")(diff)[:, :, :3] * 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "arch_presented.png"), cv2.cvtColor(pres, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUT, "arch_reference.png"), cv2.cvtColor(ara, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUT, "arch_mask.png"), cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))
zone = np.zeros(ara.shape[:2], np.uint8)
for (y0, y1, x0, x1) in [(0.21, 0.84, 0.035, 0.345), (0.27, 0.33, 0.36, 0.57), (0.40, 0.46, 0.36, 0.57), (0.51, 0.57, 0.36, 0.95),
                         (0.61, 0.67, 0.36, 0.95), (0.74, 0.80, 0.36, 0.95), (0.87, 0.93, 0.36, 0.95), (0.18, 0.25, 0.62, 0.80), (0.26, 0.55, 0.80, 0.97)]:
    zone[int(y0 * zone.shape[0]):int(y1 * zone.shape[0]), int(x0 * zone.shape[1]):int(x1 * zone.shape[1])] = 255
cv2.imwrite(os.path.join(OUT, "arch_zone.png"), zone)

fig = plt.figure(figsize=(10.5, 8.2))
fig.patch.set_facecolor("white")
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def place(im, x, y, w, h, border=None, lw=1.2):
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(im, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(border is not None)
        if border is not None:
            s_.set_edgecolor(border); s_.set_linewidth(lw)
    return ax


def arrow(x0, y0, x1, y1, color="#555555", text=None, dy=0.012, fs=8):
    fig.patches.append(mpatches.FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                       arrowstyle="-|>", mutation_scale=13, color=color, lw=1.3))
    if text:
        fig.text((x0 + x1) / 2, max(y0, y1) + dy, text, ha="center", va="bottom", fontsize=fs, color=color)


def box(x, y, w, h, title, body, color, fill):
    bx = fig.add_axes([x, y, w, h]); bx.set_xticks([]); bx.set_yticks([])
    for s_ in bx.spines.values():
        s_.set_edgecolor(color); s_.set_linewidth(1.2)
    bx.set_facecolor(fill)
    bx.text(0.5, 0.72, title, ha="center", va="center", fontsize=8.5, weight="bold", color=color, transform=bx.transAxes)
    bx.text(0.5, 0.30, body, ha="center", va="center", fontsize=7, transform=bx.transAxes)
    return bx


def rule(y):
    fig.add_artist(mlines.Line2D([0.02, 0.98], [y, y], color="#cccccc", lw=0.8, transform=fig.transFigure))


grey, blue, red, green = "#777777", "#4a6fa5", "#aa4632", "#3d7a4a"
CW, CH = 0.095, 0.085

# ---- A: evaluation gap
fig.text(0.02, 0.965, "A.  The evaluation gap: published detectors are tested on the designs they were trained on", fontsize=9.5, weight="bold", va="center")
fig.text(0.045, 0.905, "train", fontsize=8, color=grey, ha="center")
place(wv, 0.075, 0.80, CW, CH, border=grey); place(alb, 0.18, 0.80, CW, CH, border=grey)
arrow(0.285, 0.8425, 0.345, 0.8425, grey, "same design")
fig.text(0.40, 0.905, "test", fontsize=8, color=grey, ha="center")
place(wv, 0.355, 0.80, CW, CH, border=grey); place(alb, 0.46, 0.80, CW, CH, border=grey)
fig.text(0.59, 0.865, "reported: 96 to 100 % accurate", fontsize=8.5, color=grey, va="center")
fig.text(0.59, 0.82, "on a design not in training: 51 to 71 % (IDNet),\nabout half of alterations missed (FantasyID)", fontsize=8, color=red, va="center")
rule(0.775)

# ---- B: this study's test
fig.text(0.02, 0.745, "B.  This study tests on a design, a dataset and a script the detector has never seen", fontsize=9.5, weight="bold", va="center")
fig.text(0.045, 0.69, "train", fontsize=8, color=blue, ha="center")
for im, x in zip([wv, alb, fin, usa], [0.075, 0.18, 0.285, 0.39]):
    place(im, x, 0.585, CW, CH, border=blue)
fig.text(0.28, 0.572, "IDNet, SIDTD, FantasyID: 40 designs, 6 alteration types", fontsize=7.5, color=blue, ha="center", va="top")
arrow(0.495, 0.6275, 0.555, 0.6275, blue, "never seen")
fig.text(0.625, 0.69, "test", fontsize=8, color=red, ha="center")
place(pres, 0.565, 0.58, 0.12, 0.095, border=red, lw=1.8)
fig.text(0.625, 0.572, "Arabic design, altered portrait and birth year", fontsize=7.5, color=red, ha="center", va="top")
fig.text(0.72, 0.655, "A plain detector sees only this image.", fontsize=8.5, va="center")
fig.text(0.72, 0.625, "It has never seen the design, so it cannot tell\nthe design's own features from an alteration.", fontsize=8, va="center", color="#444444")
rule(0.545)

# ---- C: the method
fig.text(0.02, 0.515, "C.  The method: give the detector the clean reference of the same design, and score the deviation", fontsize=9.5, weight="bold", va="center")
fig.text(0.095, 0.47, "presented document", fontsize=8, ha="center", color=red)
place(pres, 0.04, 0.345, 0.11, 0.115, border=red, lw=1.6)
fig.text(0.095, 0.335, "from the reader", fontsize=7, ha="center", va="top", color="#555555")
fig.text(0.17, 0.40, "+", fontsize=16, ha="center", va="center", color="#444444")
fig.text(0.245, 0.47, "clean reference, same design", fontsize=8, ha="center", color=green)
place(ara, 0.19, 0.345, 0.11, 0.115, border=green, lw=1.6)
fig.text(0.245, 0.335, "retrieved from the 80-country\ntemplate library the centre already holds", fontsize=7, ha="center", va="top", color="#555555")
arrow(0.31, 0.40, 0.36, 0.40, blue)
fig.text(0.335, 0.385, "align,\ncompare", fontsize=7, color=blue, ha="center", va="top")
box(0.365, 0.35, 0.135, 0.105, "two-input detector", "Siamese network,\ndifference or\nattention fusion", blue, "#eef2f8")
arrow(0.50, 0.40, 0.545, 0.40, blue)
fig.text(0.61, 0.47, "deviation", fontsize=8, ha="center", color=blue)
place(heat, 0.555, 0.345, 0.11, 0.115, border=blue, lw=1.4)
fig.text(0.61, 0.335, "where the presented page\ndeparts from its design", fontsize=7, ha="center", va="top", color="#555555")
arrow(0.67, 0.40, 0.715, 0.40, blue)
box(0.72, 0.35, 0.25, 0.105, "refer  /  clear", "threshold set to a miss budget;\nthe design is an input, not a memory,\nso an unseen design is no longer a failure", blue, "#eef2f8")
rule(0.295)

# ---- D: reported
fig.text(0.02, 0.265, "D.  What is reported, in the second line's own terms", fontsize=9.5, weight="bold", va="center")
items = [("plain vs reference-conditioned", "on every unseen level:\ndoes the reference close the gap?"),
         ("referrals and misses", "share referred at 1 % and 5 %\nmissed, against referring all"),
         ("the gap between protocols", "unseen design within dataset\nvs unseen dataset and script"),
         ("today's base rate", "20,307 referred,\n902 altered (4.4 %)")]
for i_, (h, t) in enumerate(items):
    x = 0.04 + i_ * 0.24
    fig.text(x, 0.21, h, fontsize=8.5, weight="bold", color=blue, va="center")
    fig.text(x, 0.155, t, fontsize=7.8, va="center")

save(fig, "fig_idea")
print("idea done")


# 5. Project at a glance: pipeline figure -----------------------------------------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig = plt.figure(figsize=(14, 4.6))
fig.patch.set_facecolor("white")


def panel(x, y, w, h, color, title):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.018", transform=fig.transFigure,
                       facecolor=color, edgecolor="#333333", linewidth=1.2)
    fig.patches.append(p)
    fig.text(x + w / 2, y + h - 0.06, title, ha="center", va="center", fontsize=11, weight="bold")


def item(x, y, w, h, text, fs=8.2, fc="#fbfbf7"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.012", transform=fig.transFigure,
                       facecolor=fc, edgecolor="#333333", linewidth=0.9)
    fig.patches.append(p)
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold")


def harrow(x0, x1, y):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=16, color="#222222", lw=1.6))


def img(im, x, y, w, h, lab=None, border="#555555"):
    ax = fig.add_axes([x, y, w, h]); ax.imshow(im, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_edgecolor(border); s_.set_linewidth(1.0)
    if lab:
        fig.text(x + w / 2, y - 0.015, lab, ha="center", va="top", fontsize=7.5)


# inputs (left)
fig.text(0.075, 0.95, "Public datasets", ha="center", va="center", fontsize=9.5, weight="bold")
img(fin, 0.02, 0.66, 0.11, 0.22, "MIDV-2020: 10 designs,\nscans and phone photos")
img(alb, 0.02, 0.36, 0.11, 0.22, "SIDTD: forged versions,\nclean source, altered field")
img(usa, 0.02, 0.06, 0.11, 0.22, "FantasyID: 13 designs,\n10 scripts, printed cards")
harrow(0.14, 0.175, 0.47)

# data preparation
panel(0.18, 0.05, 0.19, 0.9, "#fdf3c4", "Data preparation")
item(0.19, 0.70, 0.17, 0.12, "Unified records: design, identity,\nlabel, type, fields, mask", fs=7.6)
item(0.19, 0.55, 0.17, 0.12, "Rectify every page to its\ndesign canvas (512 x 800)", fs=7.6)
item(0.19, 0.40, 0.17, 0.12, "Design median reference\n+ zone mask per design", fs=7.6)
item(0.19, 0.25, 0.17, 0.12, "Extra forgeries (SIDTD generator)\nwith exact masks", fs=7.6)
item(0.19, 0.10, 0.17, 0.12, "Pairs (page, reference, zone);\nidentity-grouped splits", fs=7.6)
harrow(0.375, 0.405, 0.47)

# model
panel(0.41, 0.05, 0.20, 0.9, "#f9c9c4", "Proposed architecture")
item(0.42, 0.72, 0.18, 0.10, "Base: ChangerEx (Open-CD)\nSiamese change detection", fs=7.6)
item(0.42, 0.57, 0.18, 0.11, "Standard: difference fusion\n|f(page) - f(ref)| + f(page) + zone", fs=7.4)
item(0.42, 0.42, 0.18, 0.11, "Improved: attention fusion\nself-attn, cross-attn (Q = page, K,V = ref)", fs=7.2)
item(0.42, 0.27, 0.18, 0.11, "Heads: alteration mask M\nand referral score s", fs=7.6)
item(0.42, 0.10, 0.18, 0.13, "Grid: 4 backbones x 4 fusions\nx 2 references x 4 transfer regimes", fs=7.2)
harrow(0.615, 0.645, 0.47)

# training and evaluation
panel(0.65, 0.05, 0.19, 0.9, "#ddd2f3", "Training and evaluation")
item(0.66, 0.72, 0.17, 0.10, "Loss: BCE(s) + 3 (BCE + Dice)(M)\nAdamW, early stopping, 3 seeds", fs=7.2)
item(0.66, 0.57, 0.17, 0.11, "Augment: misalignment, jitter,\nindependent JPEG on both inputs", fs=7.4)
item(0.66, 0.40, 0.17, 0.13, "Protocol A: unseen design, SIDTD split\nProtocol B: unseen dataset and script,\nFantasyID test (Arabic design alone)", fs=7.0)
item(0.66, 0.25, 0.17, 0.11, "Anchors: SIDTD baselines,\nTruFor zero-shot, same-capacity baselines", fs=7.0)
item(0.66, 0.10, 0.17, 0.11, "Metrics: AUC, EER, APCER@BPCER,\npixel F1; referral rate at miss 1, 5 %", fs=7.0)
harrow(0.845, 0.875, 0.47)

# output (right)
fig.text(0.94, 0.95, "Output", ha="center", va="center", fontsize=9.5, weight="bold")
img(pres, 0.885, 0.66, 0.11, 0.22, "presented page +\ndesign reference")
img(heat, 0.885, 0.36, 0.11, 0.22, "deviation map M")
item(0.885, 0.13, 0.11, 0.12, "refer / clear\nat miss budget", fs=8.5, fc="#eef2f8")
fig.text(0.94, 0.09, "released: data, splits, code", ha="center", va="top", fontsize=7.5)

save(fig, "fig_pipeline")
print("pipeline done")
