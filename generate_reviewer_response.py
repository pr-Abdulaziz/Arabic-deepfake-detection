"""
Reviewer Response Generator
===========================
Gathers and computes all materials requested by reviewer:
  1. Computational Cost & Efficiency (training times)
  2. Training & Validation Graphs
  3. Extended statistical metrics: Accuracy, Precision, Recall (Sensitivity),
     Specificity, ROC-AUC, PR-AUC, MCC, F1 (and weighted variants)

Output folder: Output/reviewer_response/
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "Output")
RR   = os.path.join(OUT, "output_for_paper")

def mkd(*parts):
    p = os.path.join(*parts)
    os.makedirs(p, exist_ok=True)
    return p

DIRS = {
    "root":        mkd(RR),
    "comp_cost":   mkd(RR, "1_computational_cost"),
    "train_val":   mkd(RR, "2_training_validation_graphs"),
    "metrics":     mkd(RR, "3_extended_metrics"),
}

# ── Helper ─────────────────────────────────────────────────────────────────────
def csv(platform, name):
    return os.path.join(OUT, platform, "tables", name)

def icv(platform, name):           # important/platform
    return os.path.join(OUT, "important", platform, name)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  COMPUTATIONAL COST
# ══════════════════════════════════════════════════════════════════════════════
print("── 1. Computational Cost ──")

cost_rows = []

for platform in ["twitter", "youtube"]:
    # ML classifiers – training_time column in experiment_results_long.csv
    df_exp = pd.read_csv(csv(platform, "experiment_results_long.csv"))
    df_ml  = df_exp[df_exp["model_family"] == "ML"].copy()
    df_ml["platform"] = platform
    df_ml["model_type"] = "ML"
    cost_rows.append(df_ml[["platform","model_type","model_family",
                              "model_name","embedding_name","preprocessing",
                              "training_time","macro_f1"]].rename(
                                  columns={"training_time":"training_time_sec"}))

    # Transformer – epoch_time_sec summed per model
    df_tr = pd.read_csv(csv(platform, "transformer_training_history.csv"))
    df_tr_grp = df_tr.groupby(["model_shortname","preprocessing"]).agg(
        total_training_time_sec=("epoch_time_sec","sum"),
        epochs=("epoch","max")
    ).reset_index()
    df_tr_grp["platform"]     = platform
    df_tr_grp["model_type"]   = "Transformer"
    df_tr_grp["model_family"] = "Transformer"
    df_tr_grp["model_name"]   = df_tr_grp["model_shortname"]
    df_tr_grp["embedding_name"] = df_tr_grp["model_shortname"]
    df_tr_grp["training_time_sec"] = df_tr_grp["total_training_time_sec"]
    df_tr_grp["macro_f1"]     = np.nan
    cost_rows.append(df_tr_grp[["platform","model_type","model_family",
                                  "model_name","embedding_name","preprocessing",
                                  "training_time_sec","epochs","macro_f1"]])

    # Sequence models – sum epoch time (from sequence_training_history)
    df_seq = pd.read_csv(csv(platform, "sequence_training_history.csv"))
    # No epoch_time here, but we have training_time in experiment_results_long
    df_seq2 = df_exp[df_exp["model_family"] == "Sequence"].copy()
    df_seq2["platform"]   = platform
    df_seq2["model_type"] = "Sequence"
    cost_rows.append(df_seq2[["platform","model_type","model_family",
                                "model_name","embedding_name","preprocessing",
                                "training_time","macro_f1"]].rename(
                                    columns={"training_time":"training_time_sec"}))

df_cost = pd.concat(cost_rows, ignore_index=True, sort=False)
df_cost["training_time_min"] = df_cost["training_time_sec"] / 60
df_cost.to_csv(os.path.join(DIRS["comp_cost"], "computational_cost_all_models.csv"), index=False)

# Summary table: best config per model family/platform
summary_cost = (
    df_cost.groupby(["platform","model_type","model_name","preprocessing"])
    .agg(training_time_sec=("training_time_sec","first"),
         training_time_min=("training_time_min","first"))
    .reset_index()
    .sort_values(["platform","model_type","training_time_sec"])
)
summary_cost.to_csv(os.path.join(DIRS["comp_cost"], "computational_cost_summary.csv"), index=False)

# --- Bar chart: training time by model family (best preprocessing) -----------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
for ax, platform in zip(axes, ["twitter", "youtube"]):
    sub = df_cost[df_cost["platform"] == platform].copy()
    sub["label"] = sub["model_type"] + "\n" + sub["model_name"] + "\n(" + sub["preprocessing"] + ")"
    grp = sub.groupby(["model_type","model_name"])["training_time_sec"].mean().reset_index()
    grp["label"] = grp["model_type"] + " – " + grp["model_name"]
    grp = grp.sort_values("training_time_sec")
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(grp)))
    bars = ax.barh(grp["label"], grp["training_time_sec"], color=colors)
    ax.set_xlabel("Avg. Training Time (seconds)", fontsize=11)
    ax.set_title(f"Computational Cost – {platform.capitalize()}", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.1f s", padding=4, fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(DIRS["comp_cost"], "computational_cost_barplot.png"), dpi=150)
plt.savefig(os.path.join(DIRS["comp_cost"], "computational_cost_barplot.pdf"))
plt.close()
print("  ✓ Computational cost files saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  REGENERATE TRAINING / VALIDATION GRAPHS (WITHOUT NOTES)
# ══════════════════════════════════════════════════════════════════════════════
print("── 2. Training & Validation Graphs ──")

def _clean_text(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none"} else s

def _build_label(row, id_cols):
    parts = []
    for col in id_cols:
        if col in row.index:
            val = _clean_text(row[col])
            if val:
                parts.append(val)
    return " | ".join(parts) if parts else "model"

def regenerate_train_val_plot(platform, csv_stem, *, title, ylabel, xlabel, id_cols):
    src_csv = os.path.join(OUT, platform, "tables", f"{platform}_{csv_stem}.csv")
    if not os.path.exists(src_csv):
        print(f"  ✗ Missing curve source CSV: {src_csv}")
        return

    df = pd.read_csv(src_csv)
    required = {"curve_type", "x_value", "y_value"}
    if not required.issubset(df.columns):
        print(f"  ✗ Missing required columns in {src_csv}")
        return

    df = df.copy()
    df["x_value"] = pd.to_numeric(df["x_value"], errors="coerce")
    df["y_value"] = pd.to_numeric(df["y_value"], errors="coerce")
    df = df.dropna(subset=["curve_type", "x_value", "y_value"]).copy()
    if df.empty:
        print(f"  ✗ No plottable rows in {src_csv}")
        return

    grouping_cols = [c for c in id_cols if c in df.columns]
    if not grouping_cols:
        grouping_cols = ["model_name"] if "model_name" in df.columns else []
    if not grouping_cols:
        df["_label"] = "model"
        grouping_cols = ["_label"]
    else:
        df["_label"] = df[grouping_cols].apply(lambda r: _build_label(r, grouping_cols), axis=1)

    line_groups = []
    for label, grp in df.groupby("_label"):
        for curve_type, sub in grp.groupby("curve_type"):
            line_groups.append((label, str(curve_type).lower(), sub.sort_values("x_value")))

    unique_labels = sorted({label for label, _, _ in line_groups})
    cmap = plt.get_cmap("tab20")
    color_map = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}
    style_map = {
        "train": {"linestyle": "-", "marker": "o"},
        "validation": {"linestyle": "--", "marker": "s"},
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, curve_type, sub in line_groups:
        style = style_map.get(curve_type, {"linestyle": "-", "marker": "o"})
        ax.plot(
            sub["x_value"].values,
            sub["y_value"].values,
            color=color_map[label],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=4.0,
            alpha=0.9,
            label=f"{label} ({'Train' if curve_type == 'train' else 'Validation'})",
        )

    x_min = int(np.nanmin(df["x_value"]))
    x_max = int(np.nanmax(df["x_value"]))
    ax.set_xlim(x_min, x_max)
    if x_max - x_min <= 20:
        ax.set_xticks(range(x_min, x_max + 1))
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="both", linestyle="--", alpha=0.35)
    legend_cols = 2 if len(line_groups) <= 8 else 3 if len(line_groups) <= 12 else 4
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=legend_cols,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    out_stem = os.path.join(DIRS["train_val"], f"{platform}_{csv_stem}")
    fig.savefig(f"{out_stem}.png", dpi=150)
    fig.savefig(f"{out_stem}.pdf")
    plt.close(fig)
    print(f"  ✓ Regenerated {platform}_{csv_stem}.png/.pdf")

plot_specs = [
    ("twitter", "ml_train_val_curves", "Twitter ML: Train vs Validation Macro-F1", "Macro-F1", "Configuration Rank", ["model_name"]),
    ("twitter", "sequence_train_val_loss", "Twitter Sequence: Train vs Validation Loss", "Loss", "Epoch", ["model_name", "preprocessing", "embedding_name"]),
    ("twitter", "transformer_train_val_loss", "Twitter Transformer: Train vs Validation Loss", "Loss", "Epoch", ["model_name", "preprocessing"]),
    ("youtube", "ml_train_val_curves", "YouTube ML: Train vs Validation Macro-F1", "Macro-F1", "Configuration Rank", ["model_name"]),
    ("youtube", "sequence_train_val_loss", "YouTube Sequence: Train vs Validation Loss", "Loss", "Epoch", ["model_name", "preprocessing", "embedding_name"]),
    ("youtube", "transformer_train_val_loss", "YouTube Transformer: Train vs Validation Loss", "Loss", "Epoch", ["model_name", "preprocessing"]),
]
for platform, csv_stem, title, ylabel, xlabel, id_cols in plot_specs:
    regenerate_train_val_plot(
        platform,
        csv_stem,
        title=title,
        ylabel=ylabel,
        xlabel=xlabel,
        id_cols=id_cols,
    )

# ══════════════════════════════════════════════════════════════════════════════
# 3.  EXTENDED STATISTICAL METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("── 3. Extended Metrics ──")

def compute_specificity(y_true, y_pred, pos_label=1):
    """Specificity = TN / (TN + FP) for the negative class."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return np.nan

def compute_mcc(y_true, y_pred):
    try:
        return matthews_corrcoef(y_true, y_pred)
    except Exception:
        return np.nan

def compute_roc_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

def compute_pr_auc(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


all_extended_rows = []

for platform in ["twitter", "youtube"]:
    print(f"  Processing {platform}…")

    # Load raw predictions (in_domain_predictions_long.csv)
    pred_path = os.path.join(OUT, platform, "tables", "in_domain_predictions_long.csv")
    if not os.path.exists(pred_path):
        print(f"    ✗ in_domain_predictions_long.csv not found for {platform}")
        continue

    # This file can be large; load with low-memory
    pred_df = pd.read_csv(pred_path, low_memory=False)
    print(f"    Loaded predictions: {pred_df.shape}")

    # Detect column names
    print(f"    Columns: {list(pred_df.columns[:10])}")

    # Identify key columns
    label_col = next((c for c in pred_df.columns if "true" in c.lower() or c == "label"), None)
    pred_col  = next((c for c in pred_df.columns if "pred" in c.lower() and "proba" not in c.lower()), None)
    prob_col  = next((c for c in pred_df.columns if "proba" in c.lower() or "score" in c.lower()), None)

    print(f"    label_col={label_col}, pred_col={pred_col}, prob_col={prob_col}")

    if label_col is None or pred_col is None:
        print(f"    ✗ Cannot identify label/pred columns. Skipping {platform}")
        continue

    # Group by model config
    group_cols = [c for c in ["model_family","model_name","embedding_name",
                                "preprocessing","config_name"] if c in pred_df.columns]

    for grp_keys, grp in pred_df.groupby(group_cols):
        y_true = grp[label_col].values
        y_pred = grp[pred_col].values

        # Encode labels to 0/1 if needed
        classes = np.unique(y_true)
        if set(classes) <= {"Fake","Real","fake","real","0","1",0,1}:
            if y_true.dtype == object:
                pos_cls = [c for c in classes if str(c).lower() in ("fake","1")][0]
                y_true_bin = (y_true == pos_cls).astype(int)
                y_pred_bin = (y_pred == pos_cls).astype(int)
            else:
                y_true_bin = y_true.astype(int)
                y_pred_bin = y_pred.astype(int)
        else:
            continue

        row = dict(zip(group_cols, grp_keys if isinstance(grp_keys, tuple) else [grp_keys]))
        row["platform"] = platform

        cm = confusion_matrix(y_true_bin, y_pred_bin)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            row["TP"] = int(tp)
            row["TN"] = int(tn)
            row["FP"] = int(fp)
            row["FN"] = int(fn)
            row["accuracy"]    = (tp + tn) / (tp + tn + fp + fn)
            row["precision"]   = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            row["recall_sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            row["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            row["f1"]          = (2 * tp) / (2 * tp + fp + fn) if (2*tp+fp+fn) > 0 else np.nan
            row["MCC"]         = compute_mcc(y_true_bin, y_pred_bin)
        else:
            continue

        if prob_col is not None:
            y_score = grp[prob_col].values.astype(float)
            row["ROC_AUC"] = compute_roc_auc(y_true_bin, y_score)
            row["PR_AUC"]  = compute_pr_auc(y_true_bin, y_score)
        else:
            row["ROC_AUC"] = np.nan
            row["PR_AUC"]  = np.nan

        all_extended_rows.append(row)

if all_extended_rows:
    df_ext = pd.DataFrame(all_extended_rows)
    out_path = os.path.join(DIRS["metrics"], "extended_metrics_all_models.csv")
    df_ext.to_csv(out_path, index=False)
    print(f"  ✓ Extended metrics saved: {out_path}  ({len(df_ext)} rows)")

    # Best model per platform summary
    metric_cols = ["accuracy","precision","recall_sensitivity","specificity",
                   "f1","MCC","ROC_AUC","PR_AUC"]
    existing_mc = [c for c in metric_cols if c in df_ext.columns]
    id_cols     = [c for c in ["platform","model_family","model_name",
                                "embedding_name","preprocessing"] if c in df_ext.columns]

    df_best = (df_ext.sort_values("f1", ascending=False)
                     .groupby(["platform","model_family","model_name"], as_index=False)
                     .first()[id_cols + existing_mc])
    df_best.to_csv(os.path.join(DIRS["metrics"], "extended_metrics_best_per_model.csv"), index=False)

    # ── Heatmap of all metrics for best models ──────────────────────────────
    for platform in ["twitter","youtube"]:
        sub = df_best[df_best["platform"]==platform].copy()
        if sub.empty:
            continue
        sub["label"] = sub["model_family"] + " – " + sub["model_name"]
        sub = sub.set_index("label")[existing_mc]

        fig, ax = plt.subplots(figsize=(max(10, len(existing_mc)*1.4), max(6, len(sub)*0.5+2)))
        im = ax.imshow(sub.values.astype(float), aspect="auto", cmap="RdYlGn",
                       vmin=0.85, vmax=1.0)
        ax.set_xticks(range(len(existing_mc)))
        ax.set_xticklabels(existing_mc, rotation=35, ha="right", fontsize=10)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.index, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03)
        for i in range(len(sub)):
            for j in range(len(existing_mc)):
                val = sub.values[i, j]
                ax.text(j, i, f"{val:.4f}" if not np.isnan(val) else "N/A",
                        ha="center", va="center", fontsize=7, color="black")
        ax.set_title(f"Extended Metrics Heatmap – {platform.capitalize()}",
                     fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(DIRS["metrics"], f"{platform}_extended_metrics_heatmap.png"), dpi=150)
        plt.savefig(os.path.join(DIRS["metrics"], f"{platform}_extended_metrics_heatmap.pdf"))
        plt.close()
        print(f"  ✓ Heatmap saved for {platform}")

    # ── Combined grouped histogram for precision / recall / f1 ──────────────
    family_order = {"ML": 0, "Sequence": 1, "Transformer": 2}
    preprocessing_order = ["original", "manual", "deepfake_aware"]
    preprocessing_labels = {
        "original": "Original",
        "manual": "Manual",
        "deepfake_aware": "Deepfake-aware",
    }
    metric_map = {
        "precision": "Precision",
        "recall_sensitivity": "Recall",
        "f1": "F1-score",
    }
    metric_cols_hist = [m for m in ["precision", "recall_sensitivity", "f1"] if m in df_ext.columns]

    for platform in ["twitter", "youtube"]:
        platform_rows = df_ext[df_ext["platform"] == platform].copy()
        if platform_rows.empty or not metric_cols_hist:
            continue

        best_family_models = (
            platform_rows.sort_values("f1", ascending=False)
            .groupby("model_family", as_index=False)
            .first()[["model_family", "model_name"]]
        )

        selected_rows = platform_rows.merge(
            best_family_models, on=["model_family", "model_name"], how="inner"
        )
        selected_rows = (
            selected_rows.sort_values("f1", ascending=False)
            .groupby(["model_family", "model_name", "preprocessing"], as_index=False)
            .first()
        )
        selected_rows = selected_rows[
            selected_rows["preprocessing"].isin(preprocessing_order)
        ].copy()
        if selected_rows.empty:
            continue

        selected_rows["family_sort"] = selected_rows["model_family"].map(family_order).fillna(99)
        selected_rows["prep_sort"] = selected_rows["preprocessing"].map(
            {k: i for i, k in enumerate(preprocessing_order)}
        ).fillna(99)
        selected_rows = selected_rows.sort_values(["family_sort", "prep_sort", "model_name"]).reset_index(drop=True)
        selected_rows["embedding_name"] = selected_rows["embedding_name"].fillna("").astype(str).str.strip()

        def build_group_label(row):
            prep_label = preprocessing_labels.get(row["preprocessing"], row["preprocessing"])
            emb = row["embedding_name"]
            model_label = f"{row['model_family']} - {row['model_name']}"
            if emb and emb.lower() not in {"nan", "none"} and emb.lower() != str(row["model_name"]).lower():
                model_label = f"{model_label} ({emb})"
            return f"{model_label}\n{prep_label}"

        selected_rows["group_label"] = selected_rows.apply(build_group_label, axis=1)

        group_step = 1.30
        x = np.arange(len(selected_rows)) * group_step
        bar_w = 0.18
        fig_w = max(16, len(selected_rows) * 1.75)
        fig, ax = plt.subplots(figsize=(fig_w, 7))

        min_score = float(selected_rows[metric_cols_hist].min().min())
        y_min = max(0.0, min_score - 0.06)

        metric_offsets = np.linspace(-0.30, 0.30, len(metric_cols_hist))
        for idx, metric in enumerate(metric_cols_hist):
            vals = selected_rows[metric].astype(float).values
            bars = ax.bar(
                x + metric_offsets[idx],
                vals,
                width=bar_w,
                label=metric_map.get(metric, metric),
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.004 + (idx * 0.0025),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.65),
                )

        ax.set_xticks(x)
        ax.set_xticklabels(selected_rows["group_label"], fontsize=9, rotation=22, ha="right")
        ax.set_ylim(y_min, 1.04)
        ax.set_ylabel("Score")
        ax.set_title(
            f"Precision / Recall / F1 Histogram - Best Family Models ({platform.capitalize()})",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(
            os.path.join(DIRS["metrics"], f"{platform}_best_family_models_prf_histogram.png"),
            dpi=150,
        )
        plt.savefig(
            os.path.join(DIRS["metrics"], f"{platform}_best_family_models_prf_histogram.pdf")
        )
        plt.close()

else:
    print("  ✗ No extended metric rows computed – check prediction file columns.")

    # ── Fallback: use experiment_results_long (already has precision/recall/accuracy) ──
    print("  → Falling back to experiment_results_long for available metrics…")
    fallback_rows = []
    for platform in ["twitter","youtube"]:
        df_exp = pd.read_csv(csv(platform, "experiment_results_long.csv"))
        df_exp["platform"] = platform
        fallback_rows.append(df_exp)
    df_fall = pd.concat(fallback_rows, ignore_index=True)
    df_fall.to_csv(os.path.join(DIRS["metrics"],
                                "available_metrics_from_results.csv"), index=False)
    print("  ✓ Fallback metrics saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  COPY ROC & PR CURVE FIGURES (already computed)
# ══════════════════════════════════════════════════════════════════════════════
print("── 4. Copying ROC & PR Curve Figures ──")
for platform in ["twitter", "youtube"]:
    for curve in ["roc", "pr"]:
        for ext in ["pdf", "png"]:
            fname = f"{platform}_combined_{curve}_curve.{ext}"
            src = os.path.join(OUT, platform, "figures", fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(DIRS["metrics"], fname))
                print(f"  ✓ Copied {fname}")

def regenerate_curve_bundle_from_csv(platform, curve_kind):
    csv_path = os.path.join(OUT, platform, "tables", f"{platform}_combined_{curve_kind}_curve.csv")
    if not os.path.exists(csv_path):
        print(f"  ✗ Missing curve CSV: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    required_cols = {"x_value", "y_value"}
    if not required_cols.issubset(df.columns):
        print(f"  ✗ Missing required columns in {csv_path}")
        return
    df = df.copy()
    df["x_value"] = pd.to_numeric(df["x_value"], errors="coerce")
    df["y_value"] = pd.to_numeric(df["y_value"], errors="coerce")
    df = df.dropna(subset=["x_value", "y_value"]).copy()
    if df.empty:
        print(f"  ✗ No plottable rows in {csv_path}")
        return

    for col in ["display_label", "model_family", "model_name", "embedding_name", "score_summary"]:
        if col not in df.columns:
            df[col] = ""
    df["display_label"] = df["display_label"].fillna(df["model_name"]).astype(str)

    family_order = {"ML": 0, "Sequence": 1, "Transformer": 2}
    model_order = {
        "linearsvc": 0, "logisticregression": 1, "randomforest": 2,
        "lstm": 3, "bilstm": 4,
        "arabertv2": 5, "marbertv2": 6, "camelbert": 7,
    }
    embedding_order = {"tfidf": 0, "fasttext": 1, "word2vec_cbow": 2, "random": 3}

    sort_cols = [c for c in ["point_index", "x_value"] if c in df.columns]
    if not sort_cols:
        sort_cols = ["x_value"]

    group_keys = ["display_label", "model_family", "model_name", "embedding_name", "score_summary"]
    grouped = (
        df[group_keys]
        .drop_duplicates()
        .assign(
            _f=lambda x: x["model_family"].map(family_order).fillna(99),
            _m=lambda x: x["model_name"].astype(str).str.lower().map(model_order).fillna(99),
            _e=lambda x: x["embedding_name"].astype(str).str.lower().map(embedding_order).fillna(99),
        )
        .sort_values(["_f", "_m", "_e", "display_label"])
    )

    platform_title = platform.capitalize()
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, ax = plt.subplots(figsize=(12.0, 8.0))
        palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(grouped))))

        if curve_kind == "roc":
            ax.plot(
                [0, 1], [0, 1],
                linestyle="--", linewidth=1.4, color="#8C8C8C", alpha=0.9,
                label="Random (AUC=0.500)"
            )

        for idx, row in enumerate(grouped.itertuples(index=False)):
            label_name = str(row.display_label)
            sub = df[df["display_label"] == label_name].sort_values(sort_cols)
            score_note = str(row.score_summary).strip()
            label = f"{label_name} ({score_note})" if score_note else label_name
            ax.plot(
                sub["x_value"].values,
                sub["y_value"].values,
                linewidth=2.0,
                color=palette[idx],
                label=label,
            )

        if curve_kind == "roc":
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title(f"ROC Curves - {platform_title} Dataset", fontsize=15, pad=12)
        else:
            ax.set_xlabel("Recall", fontsize=12)
            ax.set_ylabel("Precision", fontsize=12)
            ax.set_title(f"PR Curves - {platform_title} Dataset", fontsize=15, pad=12)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.grid(axis="both", linestyle="--", alpha=0.35)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
        fig.tight_layout(rect=(0, 0, 0.78, 1))

    out_stem = os.path.join(DIRS["metrics"], f"{platform}_combined_{curve_kind}_curve")
    fig.savefig(f"{out_stem}.png", dpi=150)
    fig.savefig(f"{out_stem}.pdf")
    plt.close(fig)
    print(f"  ✓ Regenerated {platform}_combined_{curve_kind}_curve.png/.pdf from CSV")

# Force-regenerate Twitter ROC/PR bundles in the same layout style used for YouTube.
for curve in ["roc", "pr"]:
    regenerate_curve_bundle_from_csv("twitter", curve)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  README SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
readme = """# Reviewer Response – Supplementary Materials

Generated automatically from project outputs.

## Folder Structure

### 1_computational_cost/
- **computational_cost_all_models.csv** — training time (seconds) for every model configuration
- **computational_cost_summary.csv**    — aggregated summary per model type
- **computational_cost_barplot.png/pdf** — horizontal bar chart of avg training time

### 2_training_validation_graphs/
Regenerated from platform training-history CSV tables (clean plots, no short notes):
- `*_ml_train_val_curves.*`            — Train vs. Validation F1/Accuracy per epoch (ML)
- `*_sequence_train_val_loss.*`        — Train vs. Validation Loss (LSTM/BiLSTM)
- `*_transformer_train_val_loss.*`     — Train vs. Validation Loss (Transformers)

### 3_extended_metrics/
- **extended_metrics_all_models.csv** — full per-config metrics table including:
    Accuracy, Precision, Recall (Sensitivity), Specificity, F1, MCC, ROC-AUC, PR-AUC
- **extended_metrics_best_per_model.csv** — one best row per model family
- **{platform}_extended_metrics_heatmap.png/pdf** — colour-coded metric heatmap
- **{platform}_transformer_all_metrics_bar.png/pdf** — grouped bar chart for transformers
- Copied ROC/PR curve figures from the main output

## Notes
- Specificity = TN / (TN + FP)  (ability to correctly identify Real text)
- MCC = Matthews Correlation Coefficient (balanced even for imbalanced classes)
- ROC-AUC / PR-AUC computed from model probability scores
- Computational times are wall-clock seconds recorded during training
"""

with open(os.path.join(RR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme)

print("\n✅  All reviewer response materials saved to:")
print(f"    {RR}")
print("    Subfolders: 1_computational_cost, 2_training_validation_graphs, 3_extended_metrics")
