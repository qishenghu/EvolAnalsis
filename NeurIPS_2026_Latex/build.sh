#!/usr/bin/env bash
# Build neurips_2026.pdf with tectonic.
# Run from the NeurIPS_2026_Latex/ directory.
#
# Usage:
#   bash build.sh           # full build (auto runs bibtex + reruns)
#   bash build.sh clean     # clean intermediates
#   bash build.sh pages     # print main-paper page breakdown
#
# First run downloads ~30 fonts/maps (~1 minute); subsequent runs are seconds.
set -e

cd "$(dirname "$0")"

# Activate the dedicated tex conda env.
source /data/home/qisheng/miniconda3/etc/profile.d/conda.sh
conda activate tex

case "${1:-build}" in
  clean)
    rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz neurips_2026.pdf
    echo "cleaned."
    ;;
  pages)
    if [[ ! -f neurips_2026.pdf ]]; then
      echo "no PDF; run 'bash build.sh' first"; exit 1
    fi
    python3 - <<'PY'
from pypdf import PdfReader
r = PdfReader('neurips_2026.pdf')
print(f'total pages: {len(r.pages)}')
markers = {
    'Abstract':   'Experience replay from',
    '1 Intro':    'Introduction',
    '2 RW':       'Related Work',
    '3 Method':   'DUET: Principled Teacher',
    '4 Exp':      'Experiments',
    '5 Disc':     'Discussion and Limitations',
    '6 Conc':     'Conclusion',
    'References': 'References',
    'App.A':      'LUFFY reproducibility',
    'App.B':      'Hyperparameter sensitivity',
    'App.C':      'Implementation Details',
    'App.D':      'Extended main results',
    'App.E':      'On ablating',
    'Checklist':  'Paper Checklist',
}
seen = set()
for i, p in enumerate(r.pages, 1):
    t = p.extract_text() or ''
    for name, marker in markers.items():
        if name in seen: continue
        if marker in t:
            print(f'  p{i:2d}  {name}')
            seen.add(name)
            break
PY
    ;;
  build|*)
    # tectonic auto-runs bibtex on first pass, reruns tex once. We do a third
    # pass to converge cross-refs. --keep-intermediates so subsequent passes are
    # fast (no font re-download).
    tectonic --keep-intermediates --chatter minimal neurips_2026.tex
    tectonic --keep-intermediates --chatter minimal neurips_2026.tex
    echo
    echo "=== build complete ==="
    ls -la neurips_2026.pdf
    bash "$0" pages
    ;;
esac
