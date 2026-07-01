#!/usr/bin/env python3
"""
Visualize control attribute prediction errors from stats.json files.
Shows how well models hit target control attribute values.
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

def load_stats_from_all_results(all_results_path, dataset, control_attr, models, label_suffix=""):
    """Load stats data from all_results.json file."""
    stats_data = {}
    
    with open(all_results_path, 'r') as f:
        all_results = json.load(f)
    
    for model in models:
        # Extract model name (handle both full paths and short names)
        model_key = model.split('/')[-1] if '/' in model else model
        
        # Try to find the model in all_results
        if model_key in all_results:
            model_data = all_results[model_key]
        elif model in all_results:
            model_data = all_results[model]
        else:
            print(f"Warning: Model '{model}' not found in all_results.json")
            continue
        
        # Navigate to dataset -> control_attr -> losses
        if dataset in model_data and control_attr in model_data[dataset]:
            if 'losses' in model_data[dataset][control_attr]:
                # Add label suffix to distinguish baseline from finetuned
                model_label = f"{model}{label_suffix}"
                stats_data[model_label] = model_data[dataset][control_attr]['losses']
            else:
                print(f"Warning: No losses data for {model} - {dataset} - {control_attr}")
        else:
            print(f"Warning: {dataset}/{control_attr} not found for {model}")
    
    return stats_data

def load_mean_ctrl_from_all_results(all_results_path, dataset, control_attr, models):
    """Load mean_ctrl data (source, reference, prediction) from all_results.json."""
    ctrl_data = {}
    
    with open(all_results_path, 'r') as f:
        all_results = json.load(f)
    
    for model in models:
        # Extract model name
        model_key = model.split('/')[-1] if '/' in model else model
        
        # Try to find the model in all_results
        if model_key in all_results:
            model_data = all_results[model_key]
        elif model in all_results:
            model_data = all_results[model]
        else:
            continue
        
        # Navigate to dataset -> control_attr -> mean_ctrl
        if dataset in model_data and control_attr in model_data[dataset]:
            if 'mean_ctrl' in model_data[dataset][control_attr]:
                ctrl_data[model] = model_data[dataset][control_attr]['mean_ctrl']
    
    return ctrl_data

def load_sample_level_predictions(baseline_dir, finetuned_dir, dataset, control_attr, model):
    """Load sample-level predictions from output_averaged.json for baseline and finetuned models."""
    
    # Baseline path: output/baseline_inference/{MODEL}-{DATASET}-{CTRL_ATTR}/output_averaged.json
    baseline_path = Path(baseline_dir) / f"{model}-{dataset}-{control_attr}" / "output_averaged.json"
    
    # Finetuned path: output/sft_inference/{CTRL_ATTR}-{DATASET}-token_explanation-{MODEL}-*/output_averaged.json
    # Need to find the date-stamped directory
    finetuned_pattern = f"{control_attr}-{dataset}-token_explanation-{model}-*"
    finetuned_dirs = list(Path(finetuned_dir).glob(finetuned_pattern))
    
    data = {
        'source': [],
        'reference': [],
        'baseline_pred': [],
        'finetuned_pred': []
    }
    
    # Load baseline predictions
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        
        for item in baseline_data:
            src_val = item.get('source_metrics', {}).get(control_attr)
            ref_val = item.get('reference_metrics', {}).get(control_attr)
            pred_val = item.get('prediction_metrics', {}).get(control_attr)
            
            if src_val is not None and ref_val is not None and pred_val is not None:
                data['source'].append(src_val)
                data['reference'].append(ref_val)
                data['baseline_pred'].append(pred_val)
    else:
        print(f"Warning: Baseline predictions not found at {baseline_path}")
    
    # Load finetuned predictions
    if finetuned_dirs:
        finetuned_path = finetuned_dirs[0] / "output_averaged.json"
        if finetuned_path.exists():
            with open(finetuned_path, 'r') as f:
                finetuned_data = json.load(f)
            
            # Should have same samples as baseline, add finetuned predictions
            for i, item in enumerate(finetuned_data):
                pred_val = item.get('prediction_metrics', {}).get(control_attr)
                if i < len(data['baseline_pred']) and pred_val is not None:
                    data['finetuned_pred'].append(pred_val)
        else:
            print(f"Warning: Finetuned predictions not found at {finetuned_path}")
    else:
        print(f"Warning: No finetuned directory found matching {finetuned_pattern}")
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data

def load_predictions_for_scatter(base_dir, dataset, control_attr, models, metric_key):
    """Load prediction files to create scatter plots of predicted vs actual values."""
    scatter_data = {}
    
    for model in models:
        # Try both the full model path and the model name
        model_dir_name = model.replace('/', '_')
        output_path = Path(base_dir) / f"{model}-{dataset}-{control_attr}" / "output_averaged.json"
        
        # Also try with model name only (without org prefix)
        if not output_path.exists():
            model_short = model.split('/')[-1] if '/' in model else model
            output_path = Path(base_dir) / f"{model_short}-{dataset}-{control_attr}" / "output_averaged.json"
        
        if output_path.exists():
            with open(output_path, 'r') as f:
                predictions = json.load(f)
                
            # Extract predicted and reference values
            pred_vals = []
            ref_vals = []
            for item in predictions:
                if 'prediction_metrics' in item and 'reference_metrics' in item:
                    pred_metric = item['prediction_metrics'].get(metric_key)
                    ref_metric = item['reference_metrics'].get(metric_key)
                    if pred_metric is not None and ref_metric is not None:
                        pred_vals.append(pred_metric)
                        ref_vals.append(ref_metric)
            
            scatter_data[model] = {
                'predictions': np.array(pred_vals),
                'references': np.array(ref_vals)
            }
        else:
            print(f"Warning: Predictions file not found for {model}")
    
    return scatter_data

def create_error_comparison_bar(stats_data, output_path, dataset, control_attr, compare_mode=False):
    """Create bar chart comparing MAE across models."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if compare_mode:
        title = f'MAE: Baseline vs Finetuned Models\n{dataset} - {control_attr}'
    else:
        title = f'Mean Absolute Error (MAE) - Control Attribute Prediction\n{dataset} - {control_attr}'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    models = list(stats_data.keys())
    mae_values = [stats_data[model].get('MAE', 0) for model in models]
    
    if compare_mode:
        # Group by model name (baseline vs finetuned)
        model_groups = {}
        for model, mae in zip(models, mae_values):
            # Extract base model name (without baseline/finetuned suffix)
            if ' (baseline)' in model:
                base_name = model.replace(' (baseline)', '')
                if base_name not in model_groups:
                    model_groups[base_name] = {}
                model_groups[base_name]['baseline'] = mae
            elif ' (finetuned)' in model:
                base_name = model.replace(' (finetuned)', '')
                if base_name not in model_groups:
                    model_groups[base_name] = {}
                model_groups[base_name]['finetuned'] = mae
            else:
                # Handle models without suffix
                if model not in model_groups:
                    model_groups[model] = {}
                model_groups[model]['baseline'] = mae
        
        # Create grouped bars
        model_names = list(model_groups.keys())
        baseline_mae = [model_groups[m].get('baseline', 0) for m in model_names]
        finetuned_mae = [model_groups[m].get('finetuned', 0) for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_mae, width, label='Baseline', 
                      color='#ff9999', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, finetuned_mae, width, label='Finetuned', 
                      color='#66b3ff', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in model_names], rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only show label if there's data
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add improvement percentage
        for i, model_name in enumerate(model_names):
            base = model_groups[model_name].get('baseline', 0)
            fine = model_groups[model_name].get('finetuned', 0)
            if base > 0 and fine > 0:
                improvement = ((base - fine) / base) * 100
                color = 'green' if improvement > 0 else 'red'
                symbol = '↓' if improvement > 0 else '↑'
                ax.text(x[i], max(base, fine) * 1.05,
                       f'{symbol}{abs(improvement):.1f}%',
                       ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
    else:
        # Original single-model comparison
        sorted_data = sorted(zip(models, mae_values), key=lambda x: x[1])
        sorted_models, sorted_mae = zip(*sorted_data)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
        x_pos = np.arange(len(sorted_models))
        bars = ax.bar(x_pos, sorted_mae, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.split('/')[-1] for m in sorted_models], rotation=45, ha='right', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bar, val) in enumerate(zip(bars, sorted_mae)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                   f'#{i+1}',
                   ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved MAE comparison bar chart to {output_path}")
    plt.close()

def create_sorted_scatter_canvas(baseline_ctrl, finetuned_ctrl, output_path, dataset, control_attrs, models):
    """Create a canvas of sorted scatter plots comparing baseline vs finetuned models.
    
    Shows reference line, source line, baseline scatter points, and finetuned scatter points.
    Each subplot is one dataset/control_attr combination.
    """
    n_plots = len(control_attrs)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle(f'Control Attribute Prediction: Baseline vs Finetuned\n{dataset}', 
                 fontsize=18, fontweight='bold')
    
    for idx, control_attr in enumerate(control_attrs):
        ax = axes[idx]
        
        # Collect all reference values from models (should be same across models)
        all_refs = []
        all_sources = []
        
        # Get reference and source from first available model
        for model in models:
            if model in baseline_ctrl and control_attr in baseline_ctrl[model]:
                ref_val = baseline_ctrl[model][control_attr].get('reference')
                src_val = baseline_ctrl[model][control_attr].get('source')
                if ref_val is not None:
                    all_refs.append(ref_val)
                if src_val is not None:
                    all_sources.append(src_val)
                break
        
        if not all_refs:
            ax.text(0.5, 0.5, f'No data for {control_attr}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(control_attr, fontsize=13, fontweight='bold')
            continue
        
        # Use single reference and source value (mean across all samples)
        ref_val = np.mean(all_refs)
        src_val = np.mean(all_sources) if all_sources else None
        
        # Collect predictions from all models
        baseline_preds = []
        finetuned_preds = []
        model_names = []
        
        for model in models:
            base_pred = None
            fine_pred = None
            
            # Get baseline prediction
            if model in baseline_ctrl and control_attr in baseline_ctrl[model]:
                base_pred = baseline_ctrl[model][control_attr].get('prediction')
            
            # Get finetuned prediction
            if model in finetuned_ctrl and control_attr in finetuned_ctrl[model]:
                fine_pred = finetuned_ctrl[model][control_attr].get('prediction')
            
            if base_pred is not None or fine_pred is not None:
                model_names.append(model.split('/')[-1])
                baseline_preds.append(base_pred if base_pred is not None else np.nan)
                finetuned_preds.append(fine_pred if fine_pred is not None else np.nan)
        
        if not model_names:
            ax.text(0.5, 0.5, f'No predictions for {control_attr}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(control_attr, fontsize=13, fontweight='bold')
            continue
        
        # Sort by baseline predictions
        x_positions = np.arange(len(model_names))
        
        # Plot horizontal lines for source and reference
        if src_val is not None:
            ax.axhline(y=src_val, color='orchid', linestyle='--', linewidth=2, 
                      label='Source', alpha=0.8, zorder=1)
        ax.axhline(y=ref_val, color='gold', linestyle='--', linewidth=2, 
                  label='Reference', alpha=0.8, zorder=1)
        
        # Plot scatter points for baseline and finetuned
        ax.scatter(x_positions, baseline_preds, color='#ff6b6b', s=100, alpha=0.7, 
                  label='Baseline', marker='o', edgecolors='darkred', linewidth=1.5, zorder=3)
        ax.scatter(x_positions, finetuned_preds, color='#4ecdc4', s=100, alpha=0.7, 
                  label='Finetuned', marker='s', edgecolors='darkblue', linewidth=1.5, zorder=3)
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel(control_attr, fontsize=11, fontweight='bold')
        ax.set_title(control_attr, fontsize=13, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        
        if idx == 0:  # Only show legend on first subplot
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved sorted scatter canvas to {output_path}")
    plt.close()

def create_per_sample_scatter_canvas(baseline_dir, finetuned_dir, output_path, dataset, control_attrs, models):
    """Create a canvas of per-sample scatter plots sorted by reference value.
    
    Shows every individual text sample with source, reference, baseline, and finetuned predictions.
    Grid: rows = control attributes, columns = models
    """
    n_rows = len(control_attrs)
    n_cols = len(models)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Per-Sample Control Attribute Predictions\n{dataset}', 
                 fontsize=18, fontweight='bold')
    
    for row_idx, control_attr in enumerate(control_attrs):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            
            # Load data for this model and control attribute
            data = load_sample_level_predictions(baseline_dir, finetuned_dir, dataset, control_attr, model)
            
            if len(data['reference']) > 0 and len(data['baseline_pred']) > 0:
                # Sort by reference values
                sort_indices = np.argsort(data['reference'])
                x_sorted = np.arange(len(data['reference']))
                
                src_sorted = data['source'][sort_indices]
                ref_sorted = data['reference'][sort_indices]
                base_sorted = data['baseline_pred'][sort_indices] if len(data['baseline_pred']) > 0 else []
                fine_sorted = data['finetuned_pred'][sort_indices] if len(data['finetuned_pred']) > 0 else []
                
                # Plot scatter points and curve for source
                ax.scatter(x_sorted, src_sorted, color='orchid', s=15, alpha=0.5, 
                          label='Source', marker='D', zorder=3)
                src_trend = np.poly1d(np.polyfit(x_sorted, src_sorted, 4))
                ax.plot(x_sorted, src_trend(x_sorted), color='orchid', 
                       linewidth=2.5, alpha=0.8, linestyle='-', zorder=2, label='Source fit')
                
                # Plot reference line
                ax.plot(x_sorted, ref_sorted, color='gold', linestyle='-', linewidth=2, 
                       label='Reference', alpha=0.7, zorder=2)
                
                # Plot scatter points for baseline and finetuned
                if len(base_sorted) > 0:
                    ax.scatter(x_sorted, base_sorted, color='#ff6b6b', s=15, alpha=0.6, 
                              label='Baseline', marker='o', zorder=3)
                    base_trend = np.poly1d(np.polyfit(x_sorted, base_sorted, 4))
                    ax.plot(x_sorted, base_trend(x_sorted), color='#ff6b6b', 
                           linewidth=2.5, alpha=0.8, linestyle='-', zorder=2, label='Baseline fit')
                
                if len(fine_sorted) > 0:
                    ax.scatter(x_sorted, fine_sorted, color='#4ecdc4', s=15, alpha=0.6, 
                              label='Finetuned', marker='s', zorder=3)
                    fine_trend = np.poly1d(np.polyfit(x_sorted, fine_sorted, 4))
                    ax.plot(x_sorted, fine_trend(x_sorted), color='#4ecdc4', 
                           linewidth=2.5, alpha=0.8, linestyle='-', zorder=2, label='Finetuned fit')
                
                # Labels and formatting
                if row_idx == n_rows - 1:  # Bottom row
                    ax.set_xlabel('Sample Index (sorted by reference)', fontsize=10, fontweight='bold')
                if col_idx == 0:  # Left column
                    ax.set_ylabel(control_attr, fontsize=10, fontweight='bold')
                
                # Title: model name on top row, control attr on left
                if row_idx == 0:
                    ax.set_title(f'{model.split("/")[-1]}', fontsize=11, fontweight='bold')
                
                ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
                
                # Only show legend on first subplot
                if row_idx == 0 and col_idx == 0:
                    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
            else:
                # No data available for this combination
                ax.text(0.5, 0.5, f'No predictions\navailable', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                if row_idx == 0:
                    ax.set_title(f'{model.split("/")[-1]}', fontsize=11, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(control_attr, fontsize=10, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-sample scatter canvas to {output_path}")
    plt.close()

def create_prediction_scatter(scatter_data, output_path, dataset, control_attr, metric_key):
    """Create scatter plots showing predicted vs reference control values."""
    n_models = len(scatter_data)
    if n_models == 0:
        print("No data available for scatter plot")
        return
    
    # Determine grid layout
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle(f'Predicted vs Actual Control Values\n{dataset} - {control_attr} ({metric_key})', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    for idx, (model, data) in enumerate(scatter_data.items()):
        ax = axes[idx]
        
        predictions = data['predictions']
        references = data['references']
        
        if len(predictions) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model.split('/')[-1], fontsize=11, fontweight='bold')
            continue
        
        # Scatter plot
        ax.scatter(references, predictions, alpha=0.5, s=30, color=colors[idx], edgecolors='black', linewidth=0.5)
        
        # Add perfect prediction line (y=x)
        min_val = min(references.min(), predictions.min())
        max_val = max(references.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(references, predictions, 1)
        p = np.poly1d(z)
        ax.plot(references, p(references), 'b-', linewidth=1.5, alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Calculate R²
        correlation_matrix = np.corrcoef(references, predictions)
        r_squared = correlation_matrix[0, 1]**2
        
        ax.set_xlabel('Actual (Reference)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_title(f'{model.split("/")[-1]}\nR²={r_squared:.3f}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved prediction scatter plot to {output_path}")
    plt.close()

def create_error_distribution_box(scatter_data, output_path, dataset, control_attr):
    """Create box plots showing distribution of prediction errors across models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    error_data = []
    labels = []
    
    for model, data in scatter_data.items():
        predictions = data['predictions']
        references = data['references']
        
        if len(predictions) > 0:
            errors = predictions - references
            error_data.append(errors)
            labels.append(model.split('/')[-1])
    
    if len(error_data) == 0:
        print("No data available for box plot")
        plt.close()
        return
    
    positions = range(1, len(error_data) + 1)
    bp = ax.boxplot(error_data, positions=positions, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(error_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Prediction (Error=0)')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Error (Predicted - Actual)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of Control Attribute Prediction Errors\n{dataset} - {control_attr}', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved error distribution box plot to {output_path}")
    plt.close()

def create_error_heatmap(stats_data, output_path, dataset, control_attr):
    """Create simple table visualization showing MAE across models."""
    models = list(stats_data.keys())
    
    # Get MAE values
    mae_values = np.array([[stats_data[model].get('MAE', 0)] for model in models])
    
    # Sort by MAE
    sorted_indices = np.argsort(mae_values.flatten())
    sorted_models = [models[i] for i in sorted_indices]
    sorted_mae = mae_values[sorted_indices]
    
    # Normalize for color mapping
    mae_normalized = (sorted_mae - sorted_mae.min()) / (sorted_mae.max() - sorted_mae.min() + 1e-10)
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(models) * 0.5)))
    fig.suptitle(f'MAE Ranking\n{dataset} - {control_attr}', fontsize=16, fontweight='bold')
    
    # Create heatmap
    sns.heatmap(sorted_mae, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                xticklabels=['MAE'], yticklabels=[m.split('/')[-1] for m in sorted_models],
                ax=ax, cbar_kws={'label': 'Mean Absolute Error'}, vmin=sorted_mae.min(), vmax=sorted_mae.max())
    ax.set_ylabel('Model (sorted by performance)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved MAE ranking heatmap to {output_path}")
    plt.close()

def create_combined_dashboard(stats_data, scatter_data, output_path, dataset, control_attr, metric_key):
    """Create a comprehensive dashboard with multiple visualizations."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Control Attribute Error Analysis Dashboard\n{dataset} - {control_attr}', 
                 fontsize=18, fontweight='bold')
    
    models = list(stats_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # 1. Bar chart for MAE (top left - larger)
    ax1 = fig.add_subplot(gs[0, :2])
    mae_vals = [stats_data[m].get('MAE', 0) for m in models]
    
    # Sort by MAE
    sorted_data = sorted(zip(models, mae_vals), key=lambda x: x[1])
    sorted_models, sorted_mae = zip(*sorted_data)
    
    x = np.arange(len(sorted_models))
    bar_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
    bars = ax1.bar(x, sorted_mae, color=bar_colors, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Mean Absolute Error', fontweight='bold', fontsize=11)
    ax1.set_title('MAE Comparison (Lower is Better)', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.split('/')[-1] for m in sorted_models], rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_mae)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Empty space (top middle - removed)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # 3. Model ranking (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    # Rank by MAE (lower is better)
    mae_ranking = sorted([(m, stats_data[m].get('MAE', float('inf'))) for m in models], 
                         key=lambda x: x[1])
    rank_models = [m.split('/')[-1] for m, _ in mae_ranking]
    rank_values = [v for _, v in mae_ranking]
    y_pos = np.arange(len(rank_models))
    ax3.barh(y_pos, rank_values, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models))))
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(rank_models, fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel('MAE', fontweight='bold')
    ax3.set_title('Model Ranking (by MAE)', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4-6. Scatter plots for first 3 models (middle row)
    scatter_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for idx, (model, data) in enumerate(list(scatter_data.items())[:3]):
        ax = scatter_axes[idx]
        predictions = data['predictions']
        references = data['references']
        
        if len(predictions) > 0:
            ax.scatter(references, predictions, alpha=0.5, s=20, color=colors[idx], edgecolors='black', linewidth=0.3)
            min_val = min(references.min(), predictions.min())
            max_val = max(references.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, alpha=0.7)
            
            r_squared = np.corrcoef(references, predictions)[0, 1]**2
            ax.set_xlabel('Actual', fontsize=9, fontweight='bold')
            ax.set_ylabel('Predicted', fontsize=9, fontweight='bold')
            ax.set_title(f'{model.split("/")[-1][:20]}\nR²={r_squared:.3f}', fontsize=10, fontweight='bold')
            ax.grid(alpha=0.3)
    
    # 7. Error distribution box plot (bottom, spans all columns)
    ax7 = fig.add_subplot(gs[2, :])
    error_data = []
    labels = []
    for model, data in scatter_data.items():
        predictions = data['predictions']
        references = data['references']
        if len(predictions) > 0:
            errors = predictions - references
            error_data.append(errors)
            labels.append(model.split('/')[-1])
    
    if len(error_data) > 0:
        positions = range(1, len(error_data) + 1)
        bp = ax7.boxplot(error_data, positions=positions, labels=labels, patch_artist=True,
                         showmeans=True, meanline=True)
        box_colors = plt.cm.Set3(np.linspace(0, 1, len(error_data)))
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
        
        ax7.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax7.set_ylabel('Error (Predicted - Actual)', fontweight='bold')
        ax7.set_title('Error Distribution Across Models', fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined dashboard to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize control attribute prediction errors')
    parser.add_argument('--baseline_results', type=str, default='output/nonsft_results_baseline/all_results.json',
                       help='Path to baseline all_results.json file')
    parser.add_argument('--finetuned_results', type=str, default=None,
                       help='Path to finetuned all_results.json file (optional, for comparison)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., Med-EASi)')
    parser.add_argument('--control_attrs', type=str, nargs='+', required=True,
                       help='Control attributes (e.g., FKGL ARI CHAR_COMPRESSION)')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Model names (e.g., Llama-3.2-1B-Instruct or meta-llama/Llama-3.2-1B-Instruct)')
    parser.add_argument('--output_dir', type=str, default='visualizations/control_errors',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if we're comparing baseline vs finetuned
    compare_mode = args.finetuned_results is not None
    
    if compare_mode:
        # Load stats for each control attribute
        for control_attr in args.control_attrs:
            print(f"\n=== Processing {control_attr} ===")
            print(f"Loading baseline stats from {args.baseline_results}...")
            baseline_stats = load_stats_from_all_results(args.baseline_results, args.dataset, 
                                                          control_attr, args.models, " (baseline)")
            
            print(f"Loading finetuned stats from {args.finetuned_results}...")
            finetuned_stats = load_stats_from_all_results(args.finetuned_results, args.dataset, 
                                                           control_attr, args.models, " (finetuned)")
            
            # Merge both datasets
            stats_data = {**baseline_stats, **finetuned_stats}
            
            if not stats_data:
                print(f"Warning: No stats found for {control_attr}")
                continue
            
            # Generate bar chart for this control attribute
            base_filename = f"{args.dataset}_{control_attr}_comparison"
            create_error_comparison_bar(stats_data, output_dir / f"{base_filename}_MAE.png",
                                         args.dataset, control_attr, compare_mode)
        
        # Load mean_ctrl data for scatter canvas (all control attributes at once)
        print(f"\n=== Generating Scatter Canvas ===")
        print(f"Loading baseline mean_ctrl from {args.baseline_results}...")
        baseline_ctrl = {}
        for model in args.models:
            for control_attr in args.control_attrs:
                ctrl_data = load_mean_ctrl_from_all_results(args.baseline_results, args.dataset, 
                                                            control_attr, [model])
                if model in ctrl_data:
                    if model not in baseline_ctrl:
                        baseline_ctrl[model] = {}
                    baseline_ctrl[model][control_attr] = ctrl_data[model]
        
        print(f"Loading finetuned mean_ctrl from {args.finetuned_results}...")
        finetuned_ctrl = {}
        for model in args.models:
            for control_attr in args.control_attrs:
                ctrl_data = load_mean_ctrl_from_all_results(args.finetuned_results, args.dataset, 
                                                            control_attr, [model])
                if model in ctrl_data:
                    if model not in finetuned_ctrl:
                        finetuned_ctrl[model] = {}
                    finetuned_ctrl[model][control_attr] = ctrl_data[model]
        
        # Generate scatter canvas
        canvas_filename = f"{args.dataset}_scatter_canvas"
        create_sorted_scatter_canvas(baseline_ctrl, finetuned_ctrl, 
                                     output_dir / f"{canvas_filename}.png",
                                     args.dataset, args.control_attrs, args.models)
        
        # Generate per-sample scatter canvas
        print(f"\n=== Generating Per-Sample Scatter Canvas ===")
        per_sample_filename = f"{args.dataset}_per_sample_scatter"
        create_per_sample_scatter_canvas("output/baseline_inference", "output/sft_inference",
                                         output_dir / f"{per_sample_filename}.png",
                                         args.dataset, args.control_attrs, args.models)
        
    else:
        # Load only baseline data for each control attribute
        for control_attr in args.control_attrs:
            print(f"\n=== Processing {control_attr} ===")
            print(f"Loading stats from {args.baseline_results}...")
            stats_data = load_stats_from_all_results(args.baseline_results, args.dataset, 
                                                      control_attr, args.models)
            
            if not stats_data:
                print(f"Warning: No stats found for {control_attr}")
                continue
            
            # Generate bar chart
            base_filename = f"{args.dataset}_{control_attr}"
            create_error_comparison_bar(stats_data, output_dir / f"{base_filename}_MAE.png",
                                         args.dataset, control_attr, False)
    
    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
