# GDRFA award submission template

LaTeX template that follows the award's official rules (participation guide, read 10 Oct 2026: https://researchaward.gdrfad.gov.ae/guide-en.html).

## Rules it encodes

| Rule (from the guide) | Where it is handled |
|---|---|
| First page: title in Arabic and English, researcher name(s), workplace, job title, academic degree, email, contact number | `main.tex`, title page table (edit the `\titleEN`, `\titleAR` macros once) |
| Arabic abstract within 250 words; English abstract within 250 words | sections 2 and 3; checked by `wordcount.sh` |
| Structure: title page, Arabic abstract, English abstract, Introduction, Research Problem Significance and Objectives, Theoretical and Conceptual Framework and Literature Review, Methodology and Procedures (study population, samples, tools), Results Discussion and Analysis, Conclusions and Recommendations, References, Appendices | section headings in that order, with the guide's titles |
| 4,000 to 8,000 words including body, tables, figures, appendices and references | `wordcount.sh` counts the built PDF, title page excluded |
| Upload MS Word plus PDF | `build.sh` (PDF), `to_docx.sh` (Word) |
| Evaluated on eight criteria (scientific value, originality, theoretical framework, language, methodology, integrity, analysis and results, applied impact) | guidance notes in each section point to the criterion they serve |

## Use

```
bash build.sh       # LuaLaTeX + BibTeX -> main.pdf (babel handles Arabic; fonts TeX Gyre Termes and Amiri)
bash wordcount.sh   # totals against the rules
bash to_docx.sh     # pandoc -> main.docx with IEEE citations
```

- Grey italic `\todo{...}` notes are guidance. Set `\showguidancefalse` in the preamble before the final build; `to_docx.sh` hides them automatically.
- Figures are read from `../figures/`. References are in `references.bib` in this folder, copied from the proposal on 10 Oct 2026; add new entries here.
- TikZ drawings (architecture, system, protocols, Gantt in `../main.tex`) do not convert to Word: export them to PDF or PNG and include them with `\includegraphics`.
- Results tables for Protocol A and B are pre-filled with the published baseline rows; our rows are placeholders.

## To fill in

- Researcher 3 (name, workplace, email) and, for all four researchers, job title, academic degree and contact number. Order on the title page: Abushahla, Panopio, researcher 3, AlHajri.
- The Arabic title has been drafted, but a native speaker must check it, along with the Arabic abstract.
- The section numbers run 1 to 6 from the Introduction; the guide's 1 to 11 numbering includes the title page and abstracts. The guide allows sections to be adjusted.

## Overleaf

Upload `../GDRFA-submission-overleaf.zip` with New Project > Upload Project. `latexmkrc` makes Overleaf compile with LuaLaTeX; if it does not, set Menu > Compiler > LuaLaTeX. The shell scripts do not run on Overleaf: use Menu > Word Count there (it skips the bibliography, so add the references by hand) and download the PDF; produce the Word file locally with `to_docx.sh`.

## Where the manuscript lives

This folder is the master copy. The paper-only repository `7abushahla/GDRFA-Award-Paper` is what Overleaf syncs with; push to it with `bash scripts/sync_paper_repo.sh "message"` from the repository root, then in Overleaf use Menu > GitHub > Pull. Edits made in Overleaf come back the other way: pull the paper repository and copy `main.tex` here before editing it again.
