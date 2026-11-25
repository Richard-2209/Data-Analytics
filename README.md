# Cancer type from gene expression

This project combines the strongest parts of the original project code and the
parallel analysis in `Data Analytics Elias`. It evaluates whether a small,
stable gene set is sufficient to distinguish the five cancer types in the UCI
**Gene expression cancer RNA-Seq** dataset.

## Research question

> Can a Random Forest identify a small and stable subset of genes that is
> sufficient to distinguish five cancer types?

Three leakage-safe selection strategies are compared: random-forest importance,
ANOVA F statistics, and L1-logistic coefficients. The main result is a curve of
cross-validated Macro-F1 against the number of selected genes. An additional
all-gene random forest provides an OOB and impurity-importance benchmark.

## Methodological safeguards

- Every supervised feature selector is fitted **inside each training fold**.
- Random-Forest permutation importance is calculated on held-out samples.
- Macro-F1 and Balanced Accuracy are emphasized because class sizes differ.
- Selection frequency across folds is reported as a measure of stability.
- Macro-F1 importance is complemented by log-loss importance because the task
  is close to saturated and class predictions can remain unchanged.
- The single decision tree is descriptive and is not treated as the final model.

## Best-of-both merge

Retained from the original project:

- fold-internal RF and L1 feature selection;
- Macro-F1, balanced accuracy and ROC-AUC evaluation;
- stable-gene overlap and held-out permutation importance;
- presentation-oriented tree and reporting pipeline.

Added from Elias's analysis:

- expression-distribution and variance diagnostics;
- centred, unscaled PCA as the primary RNA-Seq exploration, with standardized
  PCA as a sensitivity view;
- ANOVA + random forest as a third selector/model combination;
- an all-gene random-forest baseline with OOB accuracy and fold-wise impurity
  stability.

## Run

The bundled data directory `data/pancan/` is used by default. From this folder:

```bash
cd "Data Analytics final"
python3 analysis.py --mode quick --output-dir results_quick
```

The quick mode checks the complete workflow with 3 folds and four gene counts.
For the final analysis:

```bash
python3 analysis.py \
  --mode full \
  --output-dir results
```

The full mode is computationally expensive because every fold independently fits
a feature-selection model and a final classifier for each candidate gene count.
The current final run uses `random_state = 555`; the exact configuration is
stored in `results/run_config.json`.

To use another copy of the dataset:

```bash
python3 analysis.py \
  --mode full \
  --data-dir "/path/to/TCGA-PANCAN-HiSeq-801x20531" \
  --output-dir "/path/to/results"
```

## Package structure

- `analysis.py`: merged, executable analysis
- `data/pancan/`: bundled UCI data and labels
- `results/`: verified full run with seed 555
- `presentation/`: final PPTX, PDF, and reusable JGU master
- `documentation/Obsidian/`: project notes

## Main outputs

- `pca_cancer_types.png`
- `pca_cancer_types_standardized.png`
- `pca_scree_plot.png`
- `expression_value_distribution.png`
- `decision_tree_depth3.png`
- `decision_tree_depth2_presentation.png`
- `performance_vs_gene_count.png`
- `cv_summary.csv`
- `all_gene_rf_fold_metrics.csv`
- `all_gene_rf_impurity_importance.csv`
- `gene_selection_stability_k50.csv`
- `stable_gene_overlap_k50.csv`
- `stable_gene_overlap_all_selectors_k50.csv`
- `rf_top_permutation_importance.png`
- `rf_top_permutation_importance_log_loss.png`
- normalized out-of-fold confusion matrices

## Final full-run result (seed 555)

- Random forest, 50 genes: Macro-F1 `0.9958`.
- ANOVA + random forest, 50 genes: Macro-F1 `0.9938`.
- L1 logistic, 50 genes: Macro-F1 `0.9990`.
- All-gene random forest: Macro-F1 `0.9958`, mean OOB accuracy `0.9953`.
- Seven genes are selected in at least three of five folds by all three
  selectors: `gene_15894`, `gene_15895`, `gene_15898`, `gene_220`,
  `gene_7238`, `gene_7964`, and `gene_9175`.

## Final presentation

The submission-ready presentation is stored in `presentation/`:

- `Abschlusspräsentation_Gene_Expression_Cancer_best_of_both_final.pptx`
- `Abschlusspräsentation_Gene_Expression_Cancer_best_of_both_final.pdf`
- `Blanker_Folienmaster_JGU_Rot.pptx`

The deck contains ten visually focused slides for a roughly ten-minute
presentation. It uses the reusable JGU master, editable charts, speaker notes,
the merged leakage-safe evaluation design, the full cross-validation results,
and the main limitations. The title currently names Richard and Elias; add any
further group members before submission if required.

## Limitations

- The dataset contains 801 samples but 20,531 gene-expression features.
- It distinguishes five tumour types; it does not detect cancer versus healthy tissue.
- Gene names are anonymized (`gene_0`, ..., `gene_20530`).
- Results come from one dataset and lack external-cohort validation.
- Choosing the best gene count from the displayed curve and reporting the same
  cross-validation score would be optimistic.  Lock the configuration before a
  final external or fully nested evaluation.

