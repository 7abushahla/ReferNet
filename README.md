# Refer or Clear

Reference-conditioned detection of altered travel documents on designs never seen. Entry for the GDRFA Dubai Scientific Research Award, second cycle (Digital Innovation Award, artificial intelligence theme). Deadline 15 October 2026.

**Keep this repository private.** The award requires work that is original, unpublished, and not submitted elsewhere, and it screens submissions for originality. Do not make the repository public before the award decision.

## Layout

| Folder | Contents |
|---|---|
| `paper/` | The submission in the award's required structure (title page with Arabic and English titles, both abstracts, eleven sections). LuaLaTeX, Overleaf-ready via `latexmkrc`. `build.sh` makes the PDF, `wordcount.sh` checks the 4,000 to 8,000-word and 250-word abstract rules, `to_docx.sh` makes the Word file. |
| `proposal/` | The full research proposal (`main.tex`, `build.sh`, pdfLaTeX) and the design package `DESIGN.md`: problem, state of the art, novelty, data pipeline, the two protocols, architecture, configuration grid, day plan. Figures and their generator, `figures/gen/make_figures.py`. |
| `evidence/` | The literature and evidence sweeps behind every claim, with source URLs: GDRFA's stated problems, the three novelty sweeps, the dataset engineering brief, the state-of-the-art brief, the two-input architecture brief. |

The implementation code (data loader, references and zone masks, Open-CD models, training and evaluation) goes in `code/` when work starts; datasets and checkpoints stay out of git (see `.gitignore`).

## Build

```
cd paper && bash build.sh        # needs LuaLaTeX, TeX Gyre Termes and Amiri (TeX Live)
cd proposal && bash build.sh     # pdfLaTeX + BibTeX
```

Notes in `proposal/README.md` and `proposal/DESIGN.md` refer to evidence files as `exa-results/<name>.md`; in this repository they are in `evidence/`.

## Authors

Hamza A. Abushahla, Ariel Justine N. Panopio, [researcher 3], Mohamed I. AlHajri. Department of Computer Science and Engineering, American University of Sharjah.
