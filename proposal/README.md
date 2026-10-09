# Refer or Clear: reference-conditioned forgery detection on unseen designs (GDRFA award proposal)

Research proposal for the GDRFA Dubai Scientific Research Award, second cycle (deadline 15 October 2026, Digital Innovation Award). Written 7 October 2026 after the event-camera liveness proposal in the sibling folder (2026-10-06 01) was withdrawn: Dubai's touchpoints are staffed and the gates read iris, so the replay and print threat it addressed does not exist there. That folder is kept as a record and is not for submission.

## Status 8 Oct 2026

Design approved in conversation and written to `DESIGN.md` (problem, SOTA, novelty, data pipeline, two protocols, architecture, grid, day plan). The proposal `main.tex` was rewritten to match: no hardware or quantization (the model runs on the counter computer behind the existing scanner), contributions in Section 1, project-at-a-glance pipeline figure (`fig_pipeline`, sideways) and architecture block diagram (`fig:arch`, TikZ with sample images), ChangerEx base framework with standard (difference) and improved (attention) variants, Protocol A (SIDTD nationality hold-out) and Protocol B (FantasyID test, Arabic design alone), MIDV-Holo/DLC-2021 dropped, IDNet optional. Novelty framing: the public measurement and open baseline that no vendor publishes and GDRFA cannot obtain from its own commercial reader.

## The idea in one paragraph

Method delta (added 7 Oct 2026 after a second novelty sweep): the detector is reference-conditioned. It takes the presented page and the clean reference of the same design (IDNet ships blank templates; FantasyID releases its designs; SIDTD bona fides with masked fields; at Dubai, the centre's own 80-country library), registers them, and classifies from the difference of feature maps, with a channel-stacked variant as ablation. No learned ID-alteration detector built this way and tested on unseen designs exists (sweep: `exa-results/reference-conditioned-id-forgery-novelty-2026-10-07.md`; nearest: CheckSim ICDAR-W 2021, Chen TIFS 2022, Neitthoffer ICDAR-W 2023).

GDRFA's Document Examination Centre has published its own numbers: 20,307 referred examinations and 902 irregularities between January 2025 and June 2026 (4.4 percent hit rate), about 1,600 first-line officers feeding roughly 50 experts, templates from 80+ countries, and a statement by its head that electronically altered documents, not paper fakes, are the hard case. The proposal builds a compact tamper detector for the first-line officer's own device (counter reader, gate, or the Click & Travel handheld), evaluates it on document designs never seen in training (including an Arabic-script design), reports the referral rate at a fixed miss budget against the centre's base rate, and measures the quantized model's latency, memory and energy on handheld-class hardware with the group's mixed-precision pipeline. All data are public and openly licensed: IDNet (CC0), SIDTD (CC BY-SA 3.0), FantasyID (CC BY 4.0), MIDV-Holo and DLC-2021 (CC BY-SA 2.5).

## Build

```
bash build.sh
```

pdflatex + bibtex (IEEEtran). Fails on undefined references or citations. Figures are regenerated with:

```
cd figures/gen && ../../../../../../SANA-FE/.venv/bin/python make_figures.py
```

## Arabic title

الإحالة أو التمرير: فرز تزوير الوثائق لضابط الخط الأول بمعزل عن قالب الوثيقة، مُكمَّم للجهاز المحمول

(Refer or Clear: template-agnostic forgery triage for the first-line officer, quantized for the handheld.)

## Evidence files (citation sources)

- `exa-results/gdrfa-problems-evidence-2026-10-07.md`: GDRFA's own stated problems, KPIs, deployments and open data, with URLs (Parts A, B, C).
- `exa-results/id-forgery-triage-novelty-2026-10-07.md`: the 50+ paper novelty sweep for this idea, with the verdict and the reviewer caveats.
- `exa-results/immigration-demand-novelty-2026-10-07.md`: the sweep for the runner-up idea (passport-hall demand forecasting), kept for the record.

## Sources for GDRFA's own problem statements

- Al Najjar, electronic forgeries harder than paper: The National, 3 Jun 2024, https://www.thenationalnews.com/news/uae/2024/06/03/how-dubai-airport-staff-detect-fake-passports-within-seconds/
- Al Najjar, new counterfeiting methods, centre equipment and staff: Gulf News, 28 Jan 2024, https://gulfnews.com/uae/crime/we-will-catch-you-how-dubai-detects-fake-passports-visas-and-ids-1.100690046
- 20,307 examinations, 902 irregularities incl. digitally altered images: Gulf News, 2 Jul 2026, https://gulfnews.com/uae/government/dubai-detects-over-900-forged-documents-using-smart-system-1.500594998 ; Gulf Today, 2 Jul 2026, https://www.gulftoday.ae/news/2026/07/02/gdrfa-detects-902-discrepancies-in-20000-travel-papers-in-18-months
- Clear rise in forged foreign passports: Gulf News, 9 Jul 2025, https://gulfnews.com/uae/government/dubai-attorney-general-tours-gdrfas-document-examination-center-1.500192573
- Three-tier model, ~1,600 first-line officers, in-house training: UAE Journal, 25 Jan 2025, https://uaejournal.com/dubai-stay-seizes-88-fake-travel-documents-last-january/
- First-line officers have "only seconds": Frontex/INTERPOL, https://www.statewatch.org/media/4301/eu-frontex-interpol-doc-library-presentation-wk-15744-23-censored.pdf

## Figures

- `fig_attacks`: panels reproduced from the IDNet (arXiv 2408.01690), SIDTD (arXiv 2401.01858) and FantasyID (arXiv 2507.20808) papers via their arXiv HTML; originals in `figures/src/`. Confirm reuse permission before public publication (CC BY-SA / CC BY for the dataset images; the paper figures themselves are arXiv-licensed per paper).
- `fig_templates`: IDNet template sheet (paper figure, watermarked by its authors) and FantasyID Arabic design (paper figure).
- `fig_pipeline`: generated (matplotlib) in the style of the indoor-environment paper's pipeline figure; `arch_presented/reference/mask.png` are the sample images used inside the TikZ architecture figure.
- `fig_idea`: generated; rows A to D; the presented document in row C is a FantasyID card altered by us (portrait from another FantasyID design, birth year overwritten) and the caption says so.
- `fig_referral`: generated, Gaussian score model, labelled illustrative in the caption. The 4.4 percent base rate sets only the axis.
- TikZ: system placement, three-level protocol, Gantt.

## Submission notes

- 4,000 to 8,000 words, Word + PDF, Arabic and English title, eleven-section template (title page, Arabic abstract, English abstract, introduction, research problem, literature, methodology, results, conclusions and recommendations, references, appendices). The LaTeX note form here is for internal review; the award submission needs reflowing into the template.
- The guide page lists 15 September in two places and the index page 15 October; the index page and the 29 June and 25 September press releases say 15 October.
- Automated originality screening: this text is new and unpublished.

## Open items before submission

- Bib audit applied 7 Oct 2026 (Crossref and arXiv). Still to verify by hand: `chen2022domain` volume and pages (TIFS 2022), `li2026layout` full author names (CVPRW 2026), `zhou2026reveal` author list (arXiv 2605.28459), `ghanmi2021checksim` author list against the Springer page, and the SIDTD few-shot numbers (Sci Data table and GitHub README disagree).
- Confirm the MIDV-Holo and DLC-2021 licence lines against the Zenodo records (CC BY-SA 2.5 as reported by the dataset sweep).
- Decide the second deployment target (Cortex-M via TFLite, as in the spectrum-sensing paper, or a phone NPU) once the model size is known.
- Download on day one: IDNet for four countries (Zenodo multipart, ~40 GB), SIDTD (CVC direct zips), FantasyID (Zenodo 17063366, 2.6 GB). DLC-2021 (~90 GB) only if RQ4 is kept.
- If the professor has a contact at the Document Examination Centre, one sentence confirming the referral bottleneck would strengthen Section 1.

## Rejected on the way (do not re-propose)

Event-camera liveness (staffed touchpoints, iris at gates, no documented incident); passport-hall demand forecasting with gate-eligibility split (novel but validatable only against one annual GDRFA number); document capture-quality gating (no GDRFA source names it); plus the earlier list in the memory file.
