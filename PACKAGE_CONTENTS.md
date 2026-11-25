# Data Analytics final

Self-contained final package created on 2026-11-24.

## Included

- merged analysis code from both group implementations
- UCI RNA-Seq data and labels
- full seed-555 results
- final PowerPoint and PDF presentation
- reusable JGU slide master
- Obsidian project documentation

Old versions, build caches, archives, and exploratory intermediate outputs are
intentionally excluded.

## Reproduce

```bash
python3 -m pip install -r requirements.txt
python3 analysis.py --mode full --output-dir results_reproduced
```

The analysis automatically reads `data/pancan/data.csv` and
`data/pancan/labels.csv`.
