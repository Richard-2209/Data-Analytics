#!/usr/bin/env python3
"""Cancer-type classification from RNA-Seq gene expression.

This merged analysis combines the strongest parts of both group implementations:
RNA-Seq exploration, an all-gene random-forest baseline, three leakage-safe feature
selectors, selection stability, and held-out permutation importance measured with
both Macro-F1 and log loss.

The file is deliberately split into small functions and can also be opened in an
IDE that supports ``# %%`` cells.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_LOCAL_CACHE = Path(__file__).resolve().parent / ".cache"
_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_LOCAL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


PROJECT_DIR = Path(__file__).resolve().parent
COURSE_DIR = PROJECT_DIR.parent
LOCAL_DATA_DIR = PROJECT_DIR / "data" / "pancan"
ELIAS_DATA_DIR = COURSE_DIR / "Data Analytics Elias" / "data" / "pancan"
DOWNLOAD_DATA_DIR = Path(
    "/Users/richardmohling/Downloads/TCGA-PANCAN-HiSeq-801x20531"
)
DEFAULT_DATA_DIR = (
    LOCAL_DATA_DIR
    if LOCAL_DATA_DIR.exists()
    else ELIAS_DATA_DIR
    if ELIAS_DATA_DIR.exists()
    else DOWNLOAD_DATA_DIR
)

JGU_RED = "#C1002A"
BLUE = "#3E6C8A"
INK = "#242424"
GRAY = "#636363"
GRID = "#D8D8D8"
CLASS_ORDER = ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"]
CLASS_COLORS = dict(
    zip(CLASS_ORDER, ["#C1002A", "#3E6C8A", "#4F8A64", "#D38A20", "#8B5D8F"])
)
CLASS_MARKERS = dict(zip(CLASS_ORDER, ["o", "s", "^", "D", "v"]))

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "axes.edgecolor": GRAY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


@dataclass(frozen=True)
class RunConfig:
    mode: str
    gene_counts: tuple[int, ...]
    folds: int
    rf_selector_trees: int
    rf_model_trees: int
    permutation_repeats: int
    logloss_permutation_repeats: int
    importance_gene_count: int
    random_state: int = 555


def make_config(mode: str) -> RunConfig:
    if mode == "quick":
        return RunConfig(
            mode=mode,
            gene_counts=(3, 10, 50, 100),
            folds=3,
            rf_selector_trees=60,
            rf_model_trees=120,
            permutation_repeats=3,
            logloss_permutation_repeats=3,
            importance_gene_count=50,
        )
    return RunConfig(
        mode=mode,
        gene_counts=(1, 2, 3, 5, 10, 20, 25, 50, 100, 500, 1000),
        folds=5,
        rf_selector_trees=250,
        rf_model_trees=500,
        permutation_repeats=10,
        logloss_permutation_repeats=5,
        importance_gene_count=50,
    )


# %% Data loading and validation
def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    data_path = data_dir / "data.csv"
    labels_path = data_dir / "labels.csv"
    if not data_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Expected data.csv and labels.csv in {data_dir.resolve()}"
        )

    # Passing a scalar dtype would also be applied to the string sample-ID
    # column by some pandas versions.  A per-column mapping avoids that and
    # keeps the 20k-feature matrix at float32 memory usage.
    data_columns = pd.read_csv(data_path, nrows=0).columns
    numeric_dtypes = {column: np.float32 for column in data_columns[1:]}
    X = pd.read_csv(data_path, index_col=0, dtype=numeric_dtypes)
    labels = pd.read_csv(labels_path, index_col=0)
    if "Class" not in labels.columns:
        raise ValueError("labels.csv must contain a 'Class' column")
    y = labels["Class"].astype(str)

    if not X.index.equals(y.index):
        missing_x = y.index.difference(X.index)
        missing_y = X.index.difference(y.index)
        raise ValueError(
            "Sample identifiers do not align. "
            f"Missing in data: {list(missing_x[:5])}; "
            f"missing in labels: {list(missing_y[:5])}"
        )
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Unexpected missing values found")
    if not X.columns.is_unique:
        raise ValueError("Gene columns must be unique")
    return X, y


def save_dataset_summary(
    X: pd.DataFrame, y: pd.Series, output_dir: Path
) -> None:
    counts = y.value_counts().sort_index()
    gene_variance = X.var(axis=0)
    values = X.to_numpy()
    summary = {
        "samples": int(X.shape[0]),
        "genes": int(X.shape[1]),
        "nonconstant_genes": int((gene_variance > 0).sum()),
        "zero_variance_genes": int((gene_variance == 0).sum()),
        "all_zero_genes": int((X == 0).all(axis=0).sum()),
        "classes": int(y.nunique()),
        "missing_values": int(X.isna().sum().sum() + y.isna().sum()),
        "minimum_expression": float(values.min()),
        "maximum_expression": float(values.max()),
        "share_exact_zero": float((values == 0).mean()),
        "class_counts": {k: int(v) for k, v in counts.items()},
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    counts.rename("samples").to_csv(output_dir / "class_distribution.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(counts.index, counts.values, color=JGU_RED)
    ax.bar_label(bars, padding=3)
    ax.set(title="Class distribution", xlabel="Cancer type", ylabel="Samples")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close(fig)

    nonzero = values[values > 0]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(nonzero, bins=100, color=BLUE, edgecolor="white", linewidth=0.25)
    ax.set(
        title="Distribution of non-zero RNA-Seq expression values",
        xlabel="Expression value",
        ylabel="Entries",
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value / 1_000:.0f}k")
    )
    fig.tight_layout()
    fig.savefig(output_dir / "expression_value_distribution.png")
    plt.close(fig)

    variance_summary = gene_variance.quantile(
        [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
    ).rename("variance")
    variance_summary.to_csv(output_dir / "gene_variance_quantiles.csv")


# %% Exploratory plots
def create_pca_plot(
    X: pd.DataFrame, y: pd.Series, output_dir: Path, random_state: int
) -> None:
    # The UCI values are already normalized expression measurements. The primary
    # PCA is therefore centred but not standardized gene by gene; standardizing
    # would give near-constant genes the same weight as high-variance genes.
    variance = X.var(axis=0)
    X_nonconstant = X.loc[:, variance > 0]
    pca = PCA(n_components=4, svd_solver="randomized", random_state=random_state)
    scores = pca.fit_transform(X_nonconstant)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in CLASS_ORDER:
        mask = y.to_numpy() == label
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=28,
            alpha=0.85,
            label=f"{label} (n={mask.sum()})",
            color=CLASS_COLORS[label],
            marker=CLASS_MARKERS[label],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set(
        title="PCA of RNA-Seq expression profiles",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
    )
    ax.legend(title="Cancer type")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_cancer_types.png", dpi=180)
    plt.close(fig)

    pd.DataFrame(
        {
            "sample": X.index,
            "Class": y.to_numpy(),
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "PC3": scores[:, 2],
            "PC4": scores[:, 3],
        }
    ).to_csv(output_dir / "pca_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in CLASS_ORDER:
        mask = y.to_numpy() == label
        ax.scatter(
            scores[mask, 2],
            scores[mask, 3],
            s=28,
            alpha=0.85,
            label=f"{label} (n={mask.sum()})",
            color=CLASS_COLORS[label],
            marker=CLASS_MARKERS[label],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set(
        title="PCA of RNA-Seq expression profiles: PC3 and PC4",
        xlabel=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)",
        ylabel=f"PC4 ({pca.explained_variance_ratio_[3]:.1%} variance)",
    )
    ax.legend(title="Cancer type")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_cancer_types_pc3_pc4.png")
    plt.close(fig)

    # Sensitivity plot: the previous group code standardized every gene. Keeping
    # it as a separate diagnostic makes the preprocessing choice transparent.
    X_standardized = StandardScaler().fit_transform(X_nonconstant)
    pca_scaled = PCA(
        n_components=2, svd_solver="randomized", random_state=random_state
    )
    scaled_scores = pca_scaled.fit_transform(X_standardized)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in CLASS_ORDER:
        mask = y.to_numpy() == label
        ax.scatter(
            scaled_scores[mask, 0],
            scaled_scores[mask, 1],
            s=24,
            alpha=0.75,
            label=label,
            color=CLASS_COLORS[label],
            marker=CLASS_MARKERS[label],
        )
    ax.set(
        title="Sensitivity: PCA after gene-wise standardization",
        xlabel=f"PC1 ({pca_scaled.explained_variance_ratio_[0]:.1%} variance)",
        ylabel=f"PC2 ({pca_scaled.explained_variance_ratio_[1]:.1%} variance)",
    )
    ax.legend(title="Cancer type")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_cancer_types_standardized.png")
    plt.close(fig)

    pca_metadata = {
        "primary_preprocessing": "centred, not gene-wise standardized",
        "primary_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "standardized_explained_variance_ratio": (
            pca_scaled.explained_variance_ratio_.tolist()
        ),
    }
    (output_dir / "pca_metadata.json").write_text(
        json.dumps(pca_metadata, indent=2), encoding="utf-8"
    )


def create_variance_diagnostics(
    X: pd.DataFrame, output_dir: Path, random_state: int
) -> None:
    variance = X.var(axis=0)
    positive = variance[variance > 0].sort_values(ascending=False)
    top = positive.head(100)
    cumulative_share = top.cumsum() / positive.sum() * 100

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.16},
    )
    ranks = np.arange(1, len(top) + 1)
    ax_top.bar(ranks, top / positive.sum() * 100, color=BLUE, width=0.8)
    ax_top.set(
        title="Variance concentration in the 100 most variable genes",
        ylabel="Share per gene (%)",
    )
    ax_bottom.plot(ranks, cumulative_share, color=JGU_RED, linewidth=2)
    ax_bottom.set(xlabel="Gene rank by variance", ylabel="Cumulative share (%)")
    for k in (10, 50, 100):
        ax_bottom.scatter(k, cumulative_share.iloc[k - 1], color=JGU_RED, zorder=3)
        ax_bottom.annotate(
            f"top {k}: {cumulative_share.iloc[k - 1]:.1f}%",
            (k, cumulative_share.iloc[k - 1]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center" if k < 100 else "right",
            fontsize=9,
        )
    fig.savefig(output_dir / "top_variance_genes.png")
    plt.close(fig)

    X_nonconstant = X.loc[:, variance > 0]
    n_components = min(100, X_nonconstant.shape[0] - 1)
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(X_nonconstant)
    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)
    pcs = np.arange(1, n_components + 1)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(8, 6.3), gridspec_kw={"hspace": 0.4}
    )
    ax_top.bar(pcs[:30], explained[:30], color=BLUE)
    ax_top.set(
        title="Scree plot of the first 30 principal components",
        xlabel="Principal component",
        ylabel="Variance explained (%)",
    )
    ax_bottom.plot(pcs, cumulative, color=JGU_RED, linewidth=2)
    ax_bottom.set(
        title="Cumulative variance explained",
        xlabel="Number of principal components",
        ylabel="Cumulative (%)",
    )
    fig.savefig(output_dir / "pca_scree_plot.png")
    plt.close(fig)


def create_tree_plot(
    X: pd.DataFrame, y: pd.Series, output_dir: Path, random_state: int
) -> None:
    variance = VarianceThreshold()
    X_var = variance.fit_transform(X)
    var_genes = X.columns.to_numpy()[variance.get_support()]
    k = min(50, X_var.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_var, y)
    selected_genes = var_genes[selector.get_support()]

    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
    ).fit(X_selected, y)
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(
        tree,
        feature_names=selected_genes,
        class_names=tree.classes_,
        filled=True,
        rounded=True,
        proportion=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Illustrative depth-3 decision tree (descriptive fit on all samples)")
    fig.tight_layout()
    fig.savefig(output_dir / "decision_tree_depth3.png", dpi=180)
    plt.close(fig)

    # Presentation version: only the first two decision levels are shown, with
    # larger labels and a restrained palette.  This is a separately fitted,
    # descriptive tree on the same 50-gene panel (not the evaluated forest).
    presentation_tree = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
    ).fit(X_selected, y)
    class_colors = ["#F3D7DE", "#DCE8EF", "#E8E8E8", "#F5E6C8", "#DDEAD8"]
    fig, ax = plt.subplots(figsize=(16, 6.4), facecolor="white")
    annotations = plot_tree(
        presentation_tree,
        feature_names=selected_genes,
        class_names=presentation_tree.classes_,
        filled=False,
        rounded=True,
        impurity=False,
        proportion=False,
        precision=1,
        fontsize=16,
        ax=ax,
    )
    node_annotations = [
        annotation for annotation in annotations
        if "samples =" in annotation.get_text()
    ]
    for node_index, annotation in zip(
        range(presentation_tree.tree_.node_count), node_annotations
    ):
        predicted_class = int(
            np.argmax(presentation_tree.tree_.value[node_index][0])
        )
        sample_count = int(presentation_tree.tree_.n_node_samples[node_index])
        if presentation_tree.tree_.children_left[node_index] != -1:
            feature_name = selected_genes[
                presentation_tree.tree_.feature[node_index]
            ]
            threshold = presentation_tree.tree_.threshold[node_index]
            annotation.set_text(
                f"{feature_name} ≤ {threshold:.1f}\nn = {sample_count}"
            )
        else:
            annotation.set_text(
                f"n = {sample_count}\n→ {presentation_tree.classes_[predicted_class]}"
            )
        annotation.set_bbox(
            {
                "boxstyle": "round,pad=0.5",
                "facecolor": class_colors[predicted_class],
                "edgecolor": "#C1002A" if node_index == 0 else "#777777",
                "linewidth": 2.2 if node_index == 0 else 1.4,
            }
        )
        annotation.set_color("#242424")
        annotation.set_fontfamily("DejaVu Sans")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.02)
    fig.savefig(
        output_dir / "decision_tree_depth2_presentation.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(fig)


# %% Leakage-safe model construction
def build_rf_pipeline(k: int, cfg: RunConfig) -> Pipeline:
    selector_model = RandomForestClassifier(
        n_estimators=cfg.rf_selector_trees,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    final_model = RandomForestClassifier(
        n_estimators=cfg.rf_model_trees,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=cfg.random_state + 1,
    )
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            (
                "selector",
                SelectFromModel(
                    selector_model,
                    threshold=-np.inf,
                    max_features=k,
                ),
            ),
            ("model", final_model),
        ]
    )


def build_l1_pipeline(k: int, cfg: RunConfig) -> Pipeline:
    selector_model = LogisticRegression(
        penalty="l1",
        # liblinear's one-vs-rest L1 fits are substantially faster than SAGA
        # for this p >> n dataset while still providing sparse coefficients.
        solver="liblinear",
        C=0.1,
        class_weight="balanced",
        max_iter=1500 if cfg.mode == "full" else 400,
        tol=1e-3,
        random_state=cfg.random_state,
    )
    final_model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=1.0,
        class_weight="balanced",
        max_iter=1500,
        random_state=cfg.random_state + 1,
    )
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            (
                "selector",
                SelectFromModel(
                    selector_model,
                    threshold=-np.inf,
                    max_features=k,
                    norm_order=1,
                ),
            ),
            ("model", final_model),
        ]
    )


def build_anova_rf_pipeline(k: int, cfg: RunConfig) -> Pipeline:
    """Univariate ANOVA ranking followed by the same final random forest."""
    final_model = RandomForestClassifier(
        n_estimators=cfg.rf_model_trees,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=cfg.random_state + 1,
    )
    return Pipeline(
        [
            ("variance", VarianceThreshold()),
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            ("model", final_model),
        ]
    )


def selected_gene_names(pipe: Pipeline, all_genes: pd.Index) -> np.ndarray:
    variance_mask = pipe.named_steps["variance"].get_support()
    genes_after_variance = all_genes.to_numpy()[variance_mask]
    selector_mask = pipe.named_steps["selector"].get_support()
    return genes_after_variance[selector_mask]


def fold_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_order: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_roc_auc_ovr": roc_auc_score(
            y_true,
            probabilities,
            labels=class_order,
            multi_class="ovr",
            average="macro",
        ),
    }


# %% Cross-validation, stability, and held-out permutation importance
def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    cfg: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv = StratifiedKFold(
        n_splits=cfg.folds, shuffle=True, random_state=cfg.random_state
    )
    splits = list(cv.split(X, y))
    builders = {
        "Random forest": build_rf_pipeline,
        "ANOVA + random forest": build_anova_rf_pipeline,
        "L1 logistic": build_l1_pipeline,
    }

    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []

    for model_name, builder in builders.items():
        for k in cfg.gene_counts:
            print(f"[{model_name}] evaluating k={k}", flush=True)
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                pipe = builder(k, cfg)
                pipe.fit(X_train, y_train)

                y_pred = pipe.predict(X_test)
                probabilities = pipe.predict_proba(X_test)
                class_order = pipe.named_steps["model"].classes_
                metrics = fold_metrics(
                    y_test, y_pred, probabilities, class_order
                )
                metric_rows.append(
                    {
                        "model": model_name,
                        "genes": k,
                        "fold": fold,
                        **metrics,
                    }
                )

                genes = selected_gene_names(pipe, X.columns)
                for gene in genes:
                    selection_rows.append(
                        {
                            "model": model_name,
                            "genes_in_model": k,
                            "fold": fold,
                            "gene": gene,
                        }
                    )

                if k == cfg.importance_gene_count:
                    for sample, truth, pred in zip(
                        X_test.index, y_test.to_numpy(), y_pred
                    ):
                        oof_rows.append(
                            {
                                "model": model_name,
                                "sample": sample,
                                "truth": truth,
                                "prediction": pred,
                            }
                        )

                    if model_name == "Random forest":
                        # Permute only the selected held-out features. The ranking
                        # and final classifier were both learned without this fold.
                        X_test_selected = pipe[:-1].transform(X_test)
                        importance_specs = (
                            ("macro_f1", "f1_macro", cfg.permutation_repeats),
                            (
                                "log_loss",
                                "neg_log_loss",
                                cfg.logloss_permutation_repeats,
                            ),
                        )
                        for metric, scoring, repeats in importance_specs:
                            perm = permutation_importance(
                                pipe.named_steps["model"],
                                X_test_selected,
                                y_test,
                                scoring=scoring,
                                n_repeats=repeats,
                                random_state=cfg.random_state + fold,
                                n_jobs=1,
                            )
                            for gene, mean, std in zip(
                                genes,
                                perm.importances_mean,
                                perm.importances_std,
                            ):
                                importance_rows.append(
                                    {
                                        "metric": metric,
                                        "fold": fold,
                                        "gene": gene,
                                        "importance_mean": float(mean),
                                        "importance_std": float(std),
                                    }
                                )

    metrics_df = pd.DataFrame(metric_rows)
    selections_df = pd.DataFrame(selection_rows)
    importance_df = pd.DataFrame(importance_rows)
    metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    selections_df.to_csv(output_dir / "selected_genes_by_fold.csv", index=False)
    importance_df.to_csv(
        output_dir / "rf_permutation_importance_by_fold.csv", index=False
    )
    pd.DataFrame(oof_rows).to_csv(
        output_dir / "oof_predictions_k50.csv", index=False
    )
    return metrics_df, selections_df, importance_df


def evaluate_all_gene_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    cfg: RunConfig,
) -> pd.DataFrame:
    """All-gene baseline with OOB scores and fold-wise impurity stability."""
    cv = StratifiedKFold(
        n_splits=cfg.folds, shuffle=True, random_state=cfg.random_state
    )
    metric_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    impurity_folds: dict[str, pd.Series] = {}

    print("[All-gene random forest] evaluating baseline", flush=True)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        variance = VarianceThreshold()
        X_train_var = variance.fit_transform(X_train)
        X_test_var = variance.transform(X_test)
        genes = X.columns.to_numpy()[variance.get_support()]
        model = RandomForestClassifier(
            n_estimators=cfg.rf_model_trees,
            max_features="sqrt",
            class_weight="balanced_subsample",
            oob_score=True,
            n_jobs=-1,
            random_state=cfg.random_state + fold,
        )
        model.fit(X_train_var, y_train)
        prediction = model.predict(X_test_var)
        probabilities = model.predict_proba(X_test_var)
        metrics = fold_metrics(
            y_test, prediction, probabilities, model.classes_
        )
        metric_rows.append(
            {
                "model": "All-gene random forest",
                "genes": len(genes),
                "fold": fold,
                "oob_accuracy": float(model.oob_score_),
                **metrics,
            }
        )
        impurity_folds[f"fold_{fold}"] = pd.Series(
            model.feature_importances_, index=genes
        )
        for sample, truth, pred in zip(
            X_test.index, y_test.to_numpy(), prediction
        ):
            oof_rows.append(
                {
                    "sample": sample,
                    "truth": truth,
                    "prediction": pred,
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "all_gene_rf_fold_metrics.csv", index=False)
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(output_dir / "all_gene_rf_oof_predictions.csv", index=False)

    impurity_df = pd.DataFrame(impurity_folds).fillna(0.0)
    impurity_df["mean"] = impurity_df.mean(axis=1)
    impurity_df["std"] = impurity_df.drop(columns="mean").std(axis=1)
    impurity_df = impurity_df.sort_values("mean", ascending=False)
    impurity_df.to_csv(
        output_dir / "all_gene_rf_impurity_importance.csv"
    )

    top = impurity_df.head(20).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.barh(top.index, top["mean"] * 100, color=BLUE)
    ax.errorbar(
        top["mean"] * 100,
        top.index,
        xerr=top["std"] * 100,
        fmt="none",
        ecolor=INK,
        linewidth=1,
    )
    ax.set(
        title="All-gene random forest: impurity importance stability",
        xlabel="Mean decrease in impurity (% of total, mean +/- SD)",
        ylabel="Gene",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "all_gene_rf_impurity_importance.png")
    plt.close(fig)

    labels = CLASS_ORDER
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        oof_df["truth"],
        oof_df["prediction"],
        labels=labels,
        display_labels=labels,
        cmap="Blues",
        values_format="d",
        ax=ax,
        colorbar=False,
    )
    errors = int((oof_df["truth"] != oof_df["prediction"]).sum())
    ax.set_title(
        f"All-gene random forest: {errors} of {len(oof_df)} misclassified"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "all_gene_rf_confusion_matrix.png")
    plt.close(fig)
    return metrics_df


def summarize_and_plot_results(
    metrics: pd.DataFrame,
    selections: pd.DataFrame,
    importances: pd.DataFrame,
    output_dir: Path,
    cfg: RunConfig,
) -> None:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_roc_auc_ovr",
    ]
    summary = (
        metrics.groupby(["model", "genes"])[metric_names]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    summary.to_csv(output_dir / "cv_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    model_colors = {
        "Random forest": JGU_RED,
        "ANOVA + random forest": BLUE,
        "L1 logistic": "#7A7A7A",
    }
    for model_name, group in summary.groupby("model"):
        group = group.sort_values("genes")
        ax.errorbar(
            group["genes"],
            group["macro_f1_mean"],
            yerr=group["macro_f1_std"],
            marker="o",
            capsize=3,
            linewidth=2,
            label=model_name,
            color=model_colors.get(model_name),
        )
    ax.set_xscale("log")
    ax.set(
        title="Cancer-type classification versus number of selected genes",
        xlabel="Selected genes (log scale)",
        ylabel="Macro-F1 (cross-validation mean +/- SD)",
        ylim=(0.25, 1.03),
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "performance_vs_gene_count.png", dpi=180)
    plt.close(fig)

    central = selections[
        selections["genes_in_model"] == cfg.importance_gene_count
    ]
    stability = (
        central.groupby(["model", "gene"])
        .size()
        .rename("selected_folds")
        .reset_index()
    )
    stability["selection_frequency"] = stability["selected_folds"] / cfg.folds
    stability = stability.sort_values(
        ["model", "selected_folds", "gene"], ascending=[True, False, True]
    )
    stability.to_csv(output_dir / "gene_selection_stability_k50.csv", index=False)

    threshold = int(np.ceil(cfg.folds / 2))
    stable_rf = set(
        stability.loc[
            (stability["model"] == "Random forest")
            & (stability["selected_folds"] >= threshold),
            "gene",
        ]
    )
    stable_l1 = set(
        stability.loc[
            (stability["model"] == "L1 logistic")
            & (stability["selected_folds"] >= threshold),
            "gene",
        ]
    )
    stable_anova = set(
        stability.loc[
            (stability["model"] == "ANOVA + random forest")
            & (stability["selected_folds"] >= threshold),
            "gene",
        ]
    )
    overlap = pd.DataFrame(
        {
            "gene": sorted(stable_rf | stable_l1),
        }
    )
    overlap["stable_random_forest"] = overlap["gene"].isin(stable_rf)
    overlap["stable_l1_logistic"] = overlap["gene"].isin(stable_l1)
    overlap["selected_by_both"] = (
        overlap["stable_random_forest"] & overlap["stable_l1_logistic"]
    )
    overlap.to_csv(output_dir / "stable_gene_overlap_k50.csv", index=False)

    all_genes = sorted(stable_rf | stable_l1 | stable_anova)
    overlap_all = pd.DataFrame({"gene": all_genes})
    overlap_all["stable_random_forest"] = overlap_all["gene"].isin(stable_rf)
    overlap_all["stable_anova_forest"] = overlap_all["gene"].isin(stable_anova)
    overlap_all["stable_l1_logistic"] = overlap_all["gene"].isin(stable_l1)
    overlap_all["selected_by_all_three"] = (
        overlap_all["stable_random_forest"]
        & overlap_all["stable_anova_forest"]
        & overlap_all["stable_l1_logistic"]
    )
    overlap_all.to_csv(
        output_dir / "stable_gene_overlap_all_selectors_k50.csv", index=False
    )

    if not importances.empty:
        perm_summary = (
            importances.groupby(["metric", "gene"])["importance_mean"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        rf_stability = stability.loc[
            stability["model"] == "Random forest",
            ["gene", "selected_folds", "selection_frequency"],
        ]
        perm_summary = perm_summary.merge(rf_stability, on="gene", how="left")
        perm_summary = perm_summary.sort_values(
            ["metric", "selected_folds", "mean"],
            ascending=[True, False, False],
        )
        perm_summary.to_csv(
            output_dir / "rf_permutation_importance_summary.csv", index=False
        )
        for metric, x_label, filename in (
            (
                "macro_f1",
                "Decrease in held-out Macro-F1 after permutation",
                "rf_top_permutation_importance.png",
            ),
            (
                "log_loss",
                "Increase in held-out log loss after permutation",
                "rf_top_permutation_importance_log_loss.png",
            ),
        ):
            top = (
                perm_summary.loc[
                    (perm_summary["metric"] == metric)
                    & (perm_summary["selected_folds"] >= threshold)
                    & (perm_summary["mean"] > 0)
                ]
                .sort_values("mean", ascending=False)
                .head(20)
                .sort_values("mean")
                .copy()
            )
            top["std"] = top["std"].fillna(0.0)
            fig, ax = plt.subplots(figsize=(8, 6.5))
            ax.barh(top["gene"], top["mean"], xerr=top["std"], color=JGU_RED)
            ax.set(
                title=(
                    "Random forest held-out permutation importance "
                    f"(stable genes, k={cfg.importance_gene_count})"
                ),
                xlabel=x_label,
                ylabel="Gene",
            )
            fig.tight_layout()
            fig.savefig(output_dir / filename)
            plt.close(fig)

    oof_path = output_dir / "oof_predictions_k50.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        labels = sorted(set(oof["truth"]) | set(oof["prediction"]))
        for model_name, group in oof.groupby("model"):
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ConfusionMatrixDisplay.from_predictions(
                group["truth"],
                group["prediction"],
                labels=labels,
                display_labels=labels,
                normalize="true",
                cmap="Blues",
                values_format=".2f",
                ax=ax,
                colorbar=False,
            )
            ax.set_title(
                f"{model_name}: out-of-fold confusion matrix "
                f"(k={cfg.importance_gene_count})"
            )
            fig.tight_layout()
            safe_name = model_name.lower().replace(" ", "_")
            fig.savefig(
                output_dir / f"confusion_matrix_{safe_name}_k50.png", dpi=180
            )
            plt.close(fig)


def write_run_notes(output_dir: Path, cfg: RunConfig) -> None:
    note = f"""# Run notes

- Mode: `{cfg.mode}`
- Cross-validation: {cfg.folds}-fold stratified CV
- Gene counts: {', '.join(map(str, cfg.gene_counts))}
- Random-forest selector trees: {cfg.rf_selector_trees}
- Final random-forest trees: {cfg.rf_model_trees}
- Macro-F1 permutation repetitions: {cfg.permutation_repeats}
- Log-loss permutation repetitions: {cfg.logloss_permutation_repeats}
- Random seed: {cfg.random_state}

## Combined model comparison

- Random-forest feature selection followed by a random forest
- ANOVA F-statistic selection followed by the same random forest
- L1-logistic feature selection followed by L2-logistic classification
- All-gene random-forest baseline with OOB accuracy and fold-wise impurity stability

## Interpretation warning

The performance-versus-gene-count curve compares pre-specified gene counts using
the same outer folds.  Selecting the visually best gene count and quoting that same
score as a final unbiased estimate would be optimistic.  For a publication-grade
final estimate, lock the chosen configuration and evaluate it on a separate external
cohort or add a fully nested outer evaluation.

Feature selection is fitted separately inside every training fold. The held-out fold
is used only for evaluation and permutation importance. Macro-F1 permutation
importance remains easy to interpret; log-loss permutation importance is the more
sensitive diagnostic when accuracy is nearly saturated.
"""
    (output_dir / "RUN_NOTES.md").write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing data.csv and labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="quick verifies the workflow; full is intended for final results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_config(args.mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "MPLCONFIGDIR", str((args.output_dir / ".mplconfig").resolve())
    )

    print("Loading data ...", flush=True)
    X, y = load_data(args.data_dir)
    print(
        f"Loaded {X.shape[0]} samples, {X.shape[1]} genes, "
        f"classes={dict(Counter(y))}",
        flush=True,
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    save_dataset_summary(X, y, args.output_dir)
    create_pca_plot(X, y, args.output_dir, cfg.random_state)
    create_variance_diagnostics(X, args.output_dir, cfg.random_state)
    create_tree_plot(X, y, args.output_dir, cfg.random_state)
    evaluate_all_gene_random_forest(X, y, args.output_dir, cfg)
    metrics, selections, importances = evaluate_models(
        X, y, args.output_dir, cfg
    )
    summarize_and_plot_results(
        metrics, selections, importances, args.output_dir, cfg
    )
    write_run_notes(args.output_dir, cfg)
    print(f"Done. Results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

