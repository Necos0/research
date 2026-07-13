import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from helpers.utils import get_correlation_data
from helpers.keep import extract_numbers
import pandas as pd
import scipy.stats as stats
from collections import defaultdict
from classes.Metrics import Metrics

def load_json(file_path: str):
    """Load JSON from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def is_source_metric(args):
    if args.metric_key in ["FRE", "FKGL", "ARI", "DALE-CHALL", "Dale-Chall"]:
        return True
    return False

def load_predictions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _score_lens_batch(lens_model, source_texts, prediction_texts, reference_lists, batch_size=64):
    """Batch LENS scoring with compatibility fallback for package versions."""
    try:
        return lens_model.score(
            source_texts,
            prediction_texts,
            reference_lists,
            batch_size=batch_size,
            devices=[0],
        )
    except TypeError:
        return lens_model.score(
            source_texts,
            prediction_texts,
            reference_lists,
            batch_size=batch_size,
        )


def enrich_predictions_with_lens(predictions, batch_size=64):
    """Backfill LENS in prediction_metrics when missing in older output files."""
    cache = {}
    keys_to_score = []

    for item in predictions:
        prediction_metrics = item.get("prediction_metrics", {})
        if prediction_metrics.get("LENS") is not None:
            continue

        source_text = item.get("source_text")
        prediction_text = item.get("prediction")
        reference_text = item.get("reference_simplification")

        if not source_text or not prediction_text:
            continue

        cache_key = (source_text, prediction_text, reference_text)
        if cache_key not in cache:
            cache[cache_key] = None
            keys_to_score.append(cache_key)

    if keys_to_score:
        lens_model = Metrics.load_lens()
        if lens_model is None:
            print("LENS backfill skipped: LENS model unavailable.")
        else:
            for start in range(0, len(keys_to_score), batch_size):
                batch_keys = keys_to_score[start:start + batch_size]
                batch_sources = [key[0] for key in batch_keys]
                batch_predictions = [key[1] for key in batch_keys]
                batch_references = [[key[2]] if key[2] else [] for key in batch_keys]

                try:
                    scores = _score_lens_batch(
                        lens_model,
                        batch_sources,
                        batch_predictions,
                        batch_references,
                        batch_size=min(batch_size, len(batch_keys)),
                    )
                except Exception:
                    # Fallback: avoid dropping the whole run if a batch fails.
                    scores = []
                    for source_text, prediction_text, reference_list in zip(batch_sources, batch_predictions, batch_references):
                        metric_obj = Metrics(
                            input_text=prediction_text,
                            reference_text=reference_list[0] if reference_list and reference_list[0] else None,
                            source_text=source_text,
                        )
                        scores.append(metric_obj.compute_lens())

                for key, score in zip(batch_keys, scores):
                    cache[key] = float(score) if score is not None else None

    updated = 0
    for item in predictions:
        prediction_metrics = item.get("prediction_metrics", {})
        if prediction_metrics.get("LENS") is not None:
            continue

        source_text = item.get("source_text")
        prediction_text = item.get("prediction")
        reference_text = item.get("reference_simplification")
        cache_key = (source_text, prediction_text, reference_text)

        if cache_key in cache:
            prediction_metrics["LENS"] = cache[cache_key]
            item["prediction_metrics"] = prediction_metrics
            if cache[cache_key] is not None:
                updated += 1

    print(f"LENS backfill complete: {updated} predictions updated (batch_size={batch_size}).")
    return predictions

def enrich_predictions_with_bertscore(predictions, batch_size=64):
    """Backfill BERTScore（予測 vs 原文）と BERTScore_ref（予測 vs 正解）を計算する。

    推論中は SKIP_BERTSCORE=1 で roberta-large のロードを避けている（生成中の LLM と
    GPU を取り合って OOM になるため）。LLM が解放されたこの評価プロセスでまとめて計算する。
    Metrics.compute_bertscore() は1件ずつ compute() を呼ぶうえ SKIP_BERTSCORE を見て
    0.0 を返すため、ここでは evaluate のモデルを直接バッチで叩く。
    """
    # (予測文, 比較対象文) -> F1 のキャッシュ。同一ペアの再計算を避ける。
    todo = {}  # cache_key -> None
    plan = []  # (item, metrics_key, cache_key)

    for item in predictions:
        pm = item.get("prediction_metrics", {})
        prediction_text = item.get("prediction")
        if not prediction_text:
            continue
        for metrics_key, other_field in (("BERTScore", "source_text"),
                                         ("BERTScore_ref", "reference_simplification")):
            # 推論時にスキップされた場合は 0.0（未計算）で入っている
            if pm.get(metrics_key):
                continue
            other_text = item.get(other_field)
            if not other_text:
                continue
            cache_key = (prediction_text, other_text)
            todo.setdefault(cache_key, None)
            plan.append((item, metrics_key, cache_key))

    if not plan:
        print("BERTScore backfill: nothing to compute (already present).")
        return predictions

    keys = list(todo)
    try:
        bertscore_model = Metrics.load_bertscore()
        for start in range(0, len(keys), batch_size):
            batch = keys[start:start + batch_size]
            results = bertscore_model.compute(
                predictions=[k[0] for k in batch],
                references=[k[1] for k in batch],
                lang="en",
                batch_size=min(batch_size, len(batch)),
            )
            for key, f1 in zip(batch, results["f1"]):
                todo[key] = float(f1)
    except Exception as exc:
        # 評価全体を落とさない（BERTScore 以外の指標は算出済み）
        print(f"BERTScore backfill skipped: {exc}")
        return predictions

    updated = 0
    for item, metrics_key, cache_key in plan:
        score = todo.get(cache_key)
        if score is None:
            continue
        item.setdefault("prediction_metrics", {})[metrics_key] = score
        updated += 1

    print(f"BERTScore backfill complete: {updated} values updated "
          f"({len(keys)} unique pairs, batch_size={batch_size}).")
    return predictions

def average_predictions_across_runs(json_files):
    """Loads multiple JSON files and averages prediction metrics per sample."""
    all_runs = [enrich_predictions_with_bertscore(enrich_predictions_with_lens(load_json(file)))
                for file in json_files]

    assert all(len(run) == len(all_runs[0]) for run in all_runs), "All files must have the same number of samples"

    averaged_predictions = []
    for i in range(len(all_runs[0])):
        # メトリクス以外のフィールド（source_text・global_id 等）は先頭 run の値を引き継ぐ。
        # KEEP の「数値あり事例」フィルタが source_text を参照するため、ここで落とすと
        # フィルタが常に空になり損失が数値なし事例（自明に1.0）で薄まる。
        averaged_item = {k: v for k, v in all_runs[0][i].items() if k != "prediction_metrics"}
        averaged_item["prediction_metrics"] = defaultdict(list)
        for run in all_runs:
            for key, val in run[i]["prediction_metrics"].items():
                averaged_item["prediction_metrics"][key].append(val)
        
        # Compute means, filtering out None and NaN values
        final_metrics = {}
        for key, vals in averaged_item["prediction_metrics"].items():
            clean_vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            final_metrics[key] = np.mean(clean_vals) if clean_vals else None
        averaged_item["prediction_metrics"] = final_metrics
        
        averaged_predictions.append(averaged_item)

    return averaged_predictions

def extract_metric_values(predictions, metric_key, use_source=False):

    source_vals = []
    reference_vals = []
    prediction_vals = []

    for item in predictions:
        if use_source:
            # print(f"Source metrics: {item['source_metrics']}")
            source_val = item["source_metrics"].get(metric_key)
        reference_val = item["reference_metrics"].get(metric_key)
        prediction_val = item["prediction_metrics"].get(metric_key)

        if reference_val is not None and prediction_val is not None:
            if use_source:
                source_vals.append(source_val)
            reference_vals.append(reference_val)
            prediction_vals.append(prediction_val)

    return source_vals, reference_vals, prediction_vals

def compute_mean_losses(reference_vals, prediction_vals):
    mse = mean_squared_error(reference_vals, prediction_vals)
    mae = mean_absolute_error(reference_vals, prediction_vals)
    return mse, mae

def compute_per_sample_losses(reference_vals, prediction_vals):
    mse = []
    mae = []
    real_errors = []
    for ref, pred in zip(reference_vals, prediction_vals):
        mse.append(mean_squared_error([ref], [pred]))
        mae.append(mean_absolute_error([ref], [pred]))
        real_errors.append(pred - ref)
    return mse, mae, real_errors

def compute_mean_metrics(predictions):
    """Compute mean and 95% confidence intervals for all comparison metrics."""

    def clean_values(values):
        cleaned = []
        for value in values:
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isnan(value):
                continue
            cleaned.append(value)
        return cleaned

    def bootstrap_ci(data, n_bootstraps=1000, ci=0.95):
        boot_means = [
            np.mean(np.random.choice(data, size=len(data), replace=True))
            for _ in range(n_bootstraps)
        ]
        lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
        upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
        return np.mean(data), lower, upper

    BLEU_to_source, BLEU_to_ref = [], []
    BERTScore_to_source, BERTScore_to_ref = [], []
    COMET, SARI, LENS = [], [], []

    for item in predictions:
        BLEU_to_source.append(item["prediction_metrics"].get("BLEU"))
        BLEU_to_ref.append(item["prediction_metrics"].get("BLEU_ref"))
        BERTScore_to_source.append(item["prediction_metrics"].get("BERTScore"))
        BERTScore_to_ref.append(item["prediction_metrics"].get("BERTScore_ref"))
        COMET.append(item["prediction_metrics"].get("COMET"))
        SARI.append(item["prediction_metrics"].get("SARI"))
        LENS.append(item["prediction_metrics"].get("LENS"))

    metrics = {
        "BLEU_to_source": BLEU_to_source,
        "BLEU_to_ref": BLEU_to_ref,
        "BERTScore_to_source": BERTScore_to_source,
        "BERTScore_to_ref": BERTScore_to_ref,
        "COMET": COMET,
        "SARI": SARI,
        "LENS": LENS,
    }

    results = {}
    for name, values in metrics.items():
        clean_metric_values = clean_values(values)
        if not clean_metric_values:
            continue
        mean, lower, upper = bootstrap_ci(clean_metric_values)
        delta_lower = mean - lower
        delta_upper = upper - mean
        results[name] = {
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_range_minus": delta_lower,
            "ci_range_plus": delta_upper,
            "formatted": f"{name}: {mean:.2f} (+{delta_upper:.2f} -{delta_lower:.2f})"
        }

    return results

def plot_metric_scatter(source_vals, reference_vals, prediction_vals, metric_key, output_dir, use_source=False):
    # Cap and sort as before…
    source_vals = cap_outliers(source_vals) if use_source else []
    reference_vals = cap_outliers(reference_vals)
    prediction_vals = cap_outliers(prediction_vals)

    idxs       = np.argsort(reference_vals)
    x_sorted   = np.arange(len(reference_vals))
    ref_sorted = np.array(reference_vals)[idxs]
    pred_sorted= np.array(prediction_vals)[idxs]
    if use_source:
        src_sorted = np.array(source_vals)[idxs]

    fig, ax = plt.subplots(figsize=(4, 3))

    # plot the three series (and their trend‐lines)
    if use_source:
        sc1 = ax.scatter(x_sorted, src_sorted,  color="orchid", label="Source", alpha=0.5, s=20)
        ln1 = ax.plot(   x_sorted, np.poly1d(np.polyfit(x_sorted, src_sorted, 1))(x_sorted),
                        color="darkorchid", linewidth=1.5)
    sc2 = ax.scatter(x_sorted, ref_sorted,  color="gold", label="Reference",  alpha=0.5, s=20)
    sc3 = ax.scatter(x_sorted, pred_sorted, color="mediumseagreen", label="Prediction", alpha=0.5, s=20)

    ref_trend  = np.poly1d(np.polyfit(x_sorted, ref_sorted,  1))
    pred_trend = np.poly1d(np.polyfit(x_sorted, pred_sorted, 1))
    ln2 = ax.plot(x_sorted, ref_trend(x_sorted),  color="darkorange", linewidth=1.5)
    ln3 = ax.plot(x_sorted, pred_trend(x_sorted), color="seagreen",   linewidth=1.5)

    ax.set_ylabel(metric_key,  fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # 1) save the scatter + lines WITHOUT a legend
    scatter_path = os.path.join(output_dir, f"{metric_key}_scatter_plot.png")
    fig.savefig(scatter_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # 2) now build a separate legend figure
    # grab one of the axes to collect handles/labels
    # (we have to recreate the artists; easiest is to collect them before closing)
    handles = []
    labels  = []
    if use_source:
        handles.append(sc1); labels.append("Source")
        handles.append(mlines.Line2D([], [], color="darkorchid", linewidth=1.5))
        labels.append("Source trend")
    handles.append(sc2); labels.append("Reference")
    handles.append(mlines.Line2D([], [], color="darkorange", linewidth=1.5))
    labels.append("Reference trend")
    handles.append(sc3); labels.append("Prediction")
    handles.append(mlines.Line2D([], [], color="seagreen", linewidth=1.5))
    labels.append("Prediction trend")

    fig_leg = plt.figure(figsize=(4, 1))
    fig_leg.legend(handles, labels, ncol=3 if use_source else 2, frameon=False, fontsize=16, loc="center")
    fig_leg.tight_layout()
    legend_path = os.path.join(output_dir, f"{metric_key}_scatter_legend.png")
    fig_leg.savefig(legend_path, bbox_inches='tight', dpi=300)
    plt.close(fig_leg)

    print(f"Scatter plot saved to {scatter_path}")
    print(f"Legend saved to {legend_path}")


# def polyfit_plot(ax, x, y, color):
def polyfit_plot(ax, x, y, color, *, deg=1, lowess_frac=None):
    # x_np = np.array(x, dtype=np.float32)
    # y_np = np.array(y, dtype=np.float32)
    # mask = ~np.isnan(x_np) & ~np.isnan(y_np)
    # if np.sum(mask) >= 2:
    #     coeffs = np.polyfit(x_np[mask], y_np[mask], 1)
    #     trend = np.poly1d(coeffs)
    #     sorted_x = np.sort(x_np[mask])
    #     ax.plot(sorted_x, trend(sorted_x), color=color, linestyle="--", linewidth=1.5)
    x_np = np.array(x, dtype=float)
    y_np = np.array(y, dtype=float)
    mask = np.isfinite(x_np) & np.isfinite(y_np)   # NaN と inf を除外
    x_m, y_m = x_np[mask], y_np[mask]
    if len(x_m) < 2:
        return  # nothing to fit
    try:
        coeffs = np.polyfit(x_m, y_m, deg)
        poly = np.poly1d(coeffs)
        xs = np.linspace(x_m.min(), x_m.max(), 200)
        ax.plot(xs, poly(xs), color=color, linestyle="--", linewidth=1.5)
    except (np.linalg.LinAlgError, ValueError):
        # 退化データ（点が少ない/分散が無い/収束せず）ではトレンド線を省略
        return

def plot_ctrl_attr_vs_metrics(predictions, metric_key, output_dir):

    control_attr_vals = []
    BLEU_to_source, BLEU_to_ref = [], []
    BERTScore_to_source, BERTScore_to_ref = [], []
    COMET, SARI, LENS = [], [], []

    for item in predictions:
        control_attr_vals.append(item["prediction_metrics"].get(metric_key))
        BLEU_to_source.append(item["prediction_metrics"].get("BLEU"))
        BLEU_to_ref.append(item["prediction_metrics"].get("BLEU_ref"))
        BERTScore_to_source.append(item["prediction_metrics"].get("BERTScore"))
        BERTScore_to_ref.append(item["prediction_metrics"].get("BERTScore_ref"))
        COMET.append(item["prediction_metrics"].get("COMET"))
        SARI.append(item["prediction_metrics"].get("SARI"))
        LENS.append(item["prediction_metrics"].get("LENS"))

    fig, axs = plt.subplots(2, 4, figsize=(13, 6))
    metric_groups = [
        (BLEU_to_source, "BLEU to Source", "skyblue"),
        (BLEU_to_ref, "BLEU to Reference", "skyblue"),
        (COMET, "COMET", "skyblue"),
        (BERTScore_to_source, "BERTScore to Source", "skyblue"),
        (BERTScore_to_ref, "BERTScore to Reference", "skyblue"),
        (SARI, "SARI", "skyblue"),
        (LENS, "LENS", "skyblue"),
    ]

    for ax, (metric_vals, title, color) in zip(axs.flat, metric_groups):
        valid_pairs = [
            (x, y) for x, y in zip(control_attr_vals, metric_vals)
            if x is not None and y is not None
        ]
        if len(valid_pairs) < 2:
            ax.set_title("insufficient data")
            ax.set_xlabel(metric_key)
            ax.set_ylabel(title)
            continue

        x_vals = [pair[0] for pair in valid_pairs]
        y_vals = [pair[1] for pair in valid_pairs]

        ax.scatter(x_vals, y_vals, color=color, alpha=0.7, label=title)
        polyfit_plot(ax, x_vals, y_vals, color="black", deg=3)

        # Compute correlation
        correlation_data = get_correlation_data(x_vals, y_vals)
        corr_coeff = correlation_data["correlation_coefficient"]
        p_value = correlation_data["p_value"]
        significance_level = correlation_data["significance_level"]
        corr_strength = correlation_data["correlation_strength"]

        ax.set_title(significance_level)
        ax.set_xlabel(metric_key)
        ax.set_ylabel(title)
        ax.text(0.05, 0.85, f"p={p_value:.2f}\nr={corr_coeff:.2f} ({corr_strength})", transform=ax.transAxes, fontsize=10)

    for ax in axs.flat[len(metric_groups):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric_key}_vs_metrics.png", bbox_inches='tight', dpi=300)
    print(f"Control attribute vs metrics plot saved as {metric_key}_ctrl_attr_vs_metrics.png")

def cap_outliers(y_vals, lower_pct=3, upper_pct=97):
    if len(y_vals) == 0:
        return y_vals
    y_np = np.array(y_vals, dtype=np.float32)
    lower = np.percentile(y_np, lower_pct)
    upper = np.percentile(y_np, upper_pct)
    return np.clip(y_np, lower, upper)

def plot_errors_vs_metrics(predictions, metric_key_mapped, metric_key, output_dir):

    def extract_errors_and_metric(predictions, metric_name, loss_key):
        metric_vals, abs_errors, sq_errors = [], [], []
        for item in predictions:
            loss_data = item.get(f"{loss_key}_losses")
            metric_val = item["prediction_metrics"].get(metric_name)
            if loss_data and metric_val is not None:
                metric_vals.append(metric_val)
                abs_errors.append(loss_data["absolute_error"])
                sq_errors.append(loss_data["squared_error"])
        return metric_vals, abs_errors, sq_errors

    metrics = ["BLEU", "BLEU_ref", "COMET", "BERTScore", "BERTScore_ref", "SARI", "LENS"]
    labels = ["BLEU to Source", "BLEU to Reference", "COMET", "BERTScore to Source", "BERTScore to Reference", "SARI", "LENS"]

    fig_abs, axs_abs = plt.subplots(2, 4, figsize=(13, 6))
    fig_sq, axs_sq = plt.subplots(2, 4, figsize=(13, 6))

    for (ax_abs, ax_sq, metric_name, label) in zip(axs_abs.flat, axs_sq.flat, metrics, labels):
        x_vals, abs_errors, sq_errors = extract_errors_and_metric(predictions, metric_name, metric_key)

        if len(x_vals) < 2:
            ax_abs.set_title("insufficient data")
            ax_abs.set_xlabel(label)
            ax_abs.set_ylabel("Absolute Error")
            ax_sq.set_title("insufficient data")
            ax_sq.set_xlabel(label)
            ax_sq.set_ylabel("Squared Error")
            continue

        abs_errors_capped = cap_outliers(abs_errors)
        sq_errors_capped = cap_outliers(sq_errors)

        # compute correlations
        correlation_data_abs = get_correlation_data(x_vals, abs_errors_capped)
        correlation_data_sq = get_correlation_data(x_vals, sq_errors_capped)
        corr_coeff_abs = correlation_data_abs["correlation_coefficient"]
        p_value_abs = correlation_data_abs["p_value"]
        significance_level_abs = correlation_data_abs["significance_level"]
        corr_strength_abs = correlation_data_abs["correlation_strength"]
        corr_coeff_sq = correlation_data_sq["correlation_coefficient"]
        p_value_sq = correlation_data_sq["p_value"]
        significance_level_sq = correlation_data_sq["significance_level"]
        corr_strength_sq = correlation_data_sq["correlation_strength"]

        ax_abs.scatter(x_vals, abs_errors_capped, alpha=0.7, color="skyblue")
        polyfit_plot(ax_abs, x_vals, abs_errors_capped, color="black")
        ax_abs.set_xlabel(label)
        ax_abs.set_ylabel(f"Absolute Error")
        ax_abs.set_title(f"{significance_level_abs}")
        ax_abs.text(0.05, 0.85, f"p={p_value_abs:.2f}\nr={corr_coeff_abs:.2f} ({corr_strength_abs})", transform=ax_abs.transAxes, fontsize=10)

        ax_sq.scatter(x_vals, sq_errors_capped, alpha=0.7, color="skyblue")
        polyfit_plot(ax_sq, x_vals, sq_errors_capped, color="black")
        ax_sq.set_title(f"{significance_level_sq}")
        ax_sq.set_xlabel(label)
        ax_sq.set_ylabel(f"Squared Error")
        ax_sq.text(0.05, 0.85, f"p={p_value_sq:.2f}\nr={corr_coeff_sq:.2f} ({corr_strength_sq})", transform=ax_sq.transAxes, fontsize=10)

    for ax in axs_abs.flat[len(metrics):]:
        ax.set_visible(False)
    for ax in axs_sq.flat[len(metrics):]:
        ax.set_visible(False)

    fig_abs.tight_layout()
    fig_sq.tight_layout()

    fig_abs.savefig(f"{output_dir}/{metric_key_mapped}_abs_error_vs_metrics.png", bbox_inches='tight', dpi=300)
    fig_sq.savefig(f"{output_dir}/{metric_key_mapped}_sq_error_vs_metrics.png", bbox_inches='tight', dpi=300)
    print(f"Error-vs-metrics plots saved.")

def plot_error_std_vs_reference(reference_vals, real_errors, metric_key, output_dir, num_bins=10):
    import pandas as pd

    # Create DataFrame
    df = pd.DataFrame({
        "reference": reference_vals,
        "error": real_errors
    })
    # KEEP のように参照値の分散が小さい（多くが 1.0）と qcut が有効なビン境界を作れず
    # 例外になる。ビン分割できない場合はこのプロットを省略して評価全体を止めない。
    try:
        df["bin"] = pd.qcut(df["reference"], q=num_bins, duplicates='drop')
    except (ValueError, IndexError) as e:
        print(f"[WARN] Skipping error-std-by-reference-bin plot (degenerate reference values): {e}")
        return
    std_by_bin = df.groupby("bin")["error"].std()
    bin_labels = [f"{interval.left:.1f}–{interval.right:.1f}" for interval in std_by_bin.index]

    plt.figure(figsize=(5, 4))
    plt.bar(bin_labels, std_by_bin.values, color="skyblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("error std", fontsize=14)
    plt.xlabel(f"{metric_key}", fontsize=14)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle="--", alpha=0.6)
    plt.savefig(f"{output_dir}/{metric_key}_error_std_by_ref_bin.png", bbox_inches='tight', dpi=300)
    print(f"STD of error by reference bin saved as {metric_key}_error_std_by_ref_bin.png")

def plot_error_std_binned(reference_vals, real_errors, metric_key, output_dir, num_bins=25):
    mean_error = np.mean(real_errors)

    plt.figure(figsize=(4, 3))
    sns.histplot(
        real_errors, 
        bins=num_bins, 
        kde=True, 
        line_kws={"color":"royalblue", "linewidth": 1},
        color="skyblue", 
        edgecolor="black"
        )
    plt.axvline(mean_error, color='black', linestyle='--', label=f'Mean Error: {mean_error:.2f}')
    plt.xlabel(f"signed error ({metric_key}: prediction − target)", fontsize=14)
    plt.ylabel("count", fontsize=14)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=12)
    # horizontal grid
    plt.grid(axis='y', linestyle="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric_key}_error_std_binned.png", bbox_inches='tight', dpi=300)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", type=str, required=True, help="Path to JSON file with predictions")
    parser.add_argument("--input_files", type=str, nargs='+', required=True, help="Path to JSON file(s) with predictions")
    parser.add_argument("--metric_key", type=str, required=True, help="Metric to visualize")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save plots")
    parser.add_argument("--metric_mapping", type=str, required=True)
    # TODO add
    parser.add_argument("--model_name", type=str, required=True, help="Name of the evaluated model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--user_prompt_id", type=str, required=True, help="User prompt ID")
    parser.add_argument("--summary_file", type=str, default="output/sft_results/all_results.json", help="Path to save overall summary results")

    return parser.parse_args()

def main():

    print("--- Running evaluation...")

    args = parse_args()
    print("Input files:\t", args.input_files)
    print("Metric:\t", args.metric_key)

    use_source = is_source_metric(args) # source vals used
    metric_mapping = load_json(args.metric_mapping)
    metric_key_mapped = metric_mapping[args.metric_key]

    # predictions = load_predictions(args.input_file) 
    predictions = average_predictions_across_runs(args.input_files)
    with open(f"{args.output_dir}/output_averaged.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    # KEEP は原文に数値が無い事例だと保持率が自明に 1.0（reference/prediction とも）
    # になり、損失や平均が薄まる。損失・制御統計は「原文に数値がある事例」だけで計算する。
    loss_predictions = predictions
    numeric_sources_only = False
    if args.metric_key == "KEEP":
        numeric_subset = [
            item for item in predictions if extract_numbers(item.get("source_text", ""))
        ]
        if numeric_subset:
            loss_predictions = numeric_subset
            numeric_sources_only = True
            print(
                f"KEEP: losses computed on {len(numeric_subset)}/{len(predictions)} "
                "samples with numbers in the source"
            )
        else:
            print("[WARN] KEEP: no samples with numbers in the source; using all samples for losses")

    source_vals, reference_vals, prediction_vals = extract_metric_values(loss_predictions, metric_key_mapped, use_source=use_source)

    mse, mae = compute_mean_losses(reference_vals, prediction_vals)
    per_sample_mse, per_sample_mae, per_sample_real_loss = compute_per_sample_losses(reference_vals, prediction_vals)
    std_error = np.std(per_sample_real_loss)
    var_error = np.var(per_sample_real_loss)


    mean_ctrl = {
        "reference": float(np.mean(reference_vals)),
        "prediction": float(np.mean(prediction_vals))
    }
    if use_source:
        mean_ctrl["source"] = float(np.mean(source_vals))

    median_ctrl = {
        "reference": float(np.median(reference_vals)),
        "prediction": float(np.median(prediction_vals))
    }
    if use_source:
        median_ctrl["source"] = float(np.median(source_vals))




    print(f"--- {metric_key_mapped} losses:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Standard Deviation of Errors: {std_error}")
    print(f"Variance of Errors: {var_error}")
    print()

    loss_output = {
        "metric": args.metric_key,
        "MSE": mse,
        "MAE": mae,
        "std_error": std_error,
        "var_error": var_error,
        "num_samples": len(reference_vals),
        "numeric_sources_only": numeric_sources_only,
    }

    with open(f"{args.output_dir}/stats.json", "w") as f:
        json.dump(loss_output, f, indent=4)
    print(f"Saved loss values to stats.json")

    # extract_metric_values と同じ順序・条件で損失計算対象の事例にだけ付与する
    # （値が欠けた事例を挟んでもインデックスがずれないようにする）
    sample_index = 0
    for item in loss_predictions:
        if item["reference_metrics"].get(metric_key_mapped) is None or \
           item["prediction_metrics"].get(metric_key_mapped) is None:
            continue
        item[f"{args.metric_key}_losses"] = {
            "reference": reference_vals[sample_index],
            "prediction": prediction_vals[sample_index],
            "squared_error": per_sample_mse[sample_index],
            "absolute_error": per_sample_mae[sample_index],
            "real_loss": per_sample_real_loss[sample_index]
        }
        sample_index += 1

    with open(f"{args.output_dir}/output_averaged.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)
    print(f"Updated averaged predictions file with per-sample losses: {args.output_dir}/output_averaged.json")

    plot_metric_scatter(source_vals, reference_vals, prediction_vals, metric_key_mapped, args.output_dir, use_source=use_source)
    plot_ctrl_attr_vs_metrics(predictions, metric_key_mapped, args.output_dir)
    plot_errors_vs_metrics(predictions, metric_key_mapped, args.metric_key, args.output_dir)
    plot_error_std_vs_reference(reference_vals, per_sample_real_loss, metric_key_mapped, args.output_dir)
    plot_error_std_binned(reference_vals, per_sample_real_loss, metric_key_mapped, args.output_dir)
    print(f"Plots saved to {args.output_dir}")


    print(f"\nAll plots saved to {args.output_dir}")

    mean_metrics = compute_mean_metrics(predictions)
    print(f"\n--- Mean metrics:")
    for name, values in mean_metrics.items():
        print(f"{name}: {values['formatted']}")
    print()
    try:
        with open(args.summary_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

    # nest by model -> dataset -> metric
    model = args.model_name
    dataset = args.dataset
    metric = args.metric_key

    if model not in all_results:
        all_results[model] = {}
    if dataset not in all_results[model]:
        all_results[model][dataset] = {}

    all_results[model][dataset][metric] = {
        "losses": {
            "MSE": mse,
            "MAE": mae,
            "std_error": std_error,
            "var_error": var_error,
            "num_samples": len(reference_vals),
            "numeric_sources_only": numeric_sources_only,
        },
        "mean_metrics": mean_metrics,
        "mean_ctrl":    mean_ctrl,
        "median_ctrl":  median_ctrl,
    }

    # Write back to the summary file
    os.makedirs(os.path.dirname(args.summary_file) or ".", exist_ok=True)
    with open(args.summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved evaluation summary to {args.summary_file}")


    print("\n--- Evaluation complete.")



if __name__ == "__main__":
    main()