# NeurIPS 2026 Tables — Index

This folder contains LaTeX table files ready for `\input{}` into the main paper.

## File index

| File | Purpose | Status |
|---|---|---|
| `main_results.tex` | **Main paper Table 1** — 4 settings × 5 baselines + DUET\*, success rate only. | ✅ ready |
| `main_results_with_reward.tex` | Extended Table — adds reward_mean column for each setting (mostly redundant with SR on AlfWorld; useful for WebShop partial-credit). | ⚠ 4 cells TBD |

## Required preamble additions

Add these to `neurips_2026.tex` before using these tables:

```latex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage[table]{xcolor}
```

## Usage

In `neurips_2026.tex` body:

```latex
\section{Experiments}
\subsection{Main Results}
% ... narrative ...

\input{tables/main_results.tex}

% ... discussion ...
```

## Data sources

All numbers in these tables originate from `../data/raw_data.md` — that file is the single source of truth. If a number changes, edit `data/raw_data.md` first and propagate to `tables/*.tex`.

## Open TODOs

- [ ] Verify 3B WebShop reward_mean values for OnPolicy/LUFFY/CHORD/SFT+RL baselines (currently `TBD†` in `main_results_with_reward.tex`)
- [ ] Add ablation table (DUET v1 = no BC; DUET no DR3; DUET no SC) — pending decision on which ablations to include
- [ ] Add LUFFY reproducibility analysis table (49.5% paper / 38.0% L20X / 3.5% 4×A100) — could go in Appendix
- [ ] Verify final caption wording with co-authors

## Aesthetic notes

- Best baseline per column: \underline{underlined}
- DUET\* row: light gray shading (`\rowcolor{gray!12}`) for emphasis
- Numbers as percentages (1 decimal place); reward_mean as fractions (3 decimal places)
- All tables use `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`) — no vertical lines
