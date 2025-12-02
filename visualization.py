"""
Visualization and result saving utilities for MA-ICL
"""

import os
import json
import csv
import re
import numpy as np
from typing import List, Dict, Any, Optional
import logging

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except:
    _HAS_MPL = False

try:
    import networkx as nx
    _HAS_NX = True
except:
    _HAS_NX = False

try:
    from sklearn.metrics import confusion_matrix
    _HAS_SKLEARN = True
except:
    _HAS_SKLEARN = False

logger = logging.getLogger(__name__)

# =========================
# IO UTILITIES
# =========================

def _safe_write_json(path: str, obj: Any):
    """Safely write JSON to file"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write JSON to {path}: {e}")

def _safe_append_jsonl(path: str, obj: Any):
    """Safely append JSONL line to file"""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")
    except Exception as e:
        logger.warning(f"Could not append JSONL to {path}: {e}")

def _safe_write_text(path: str, text: str):
    """Safely write text to file"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        logger.warning(f"Could not write text to {path}: {e}")

# =========================
# VISUALIZATION FUNCTIONS
# =========================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], 
                          title: str = "Confusion Matrix", output_path: Optional[str] = None):
    """Plot and save confusion matrix"""
    if not _HAS_MPL:
        logger.warning("Matplotlib not available, skipping confusion matrix plot")
        return
    
    if not _HAS_SKLEARN:
        logger.warning("sklearn not available, skipping confusion matrix plot")
        return
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved confusion matrix to {output_path}")
        
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to plot confusion matrix: {e}")


def plot_performance_comparison(ml_baseline: Dict[str, Any], 
                                pre_maicl: Dict[str, Any], 
                                post_maicl: Dict[str, Any],
                                task_type: str = "classification",
                                output_path: Optional[str] = None,
                                llm_only_maicl: Optional[Dict[str, Any]] = None,
                                vanilla_llm_metrics: Optional[Dict[str, Any]] = None):
    """Plot performance comparison between ML baseline, pre-training MA-ICL, post-training MA-ICL, LLM Mechanism, and Vanilla LLM"""
    if not _HAS_MPL:
        logger.warning("Matplotlib not available, skipping performance comparison plot")
        return
    
    try:
        if task_type == "classification":
            metrics = ['accuracy', 'f1']
            metric_labels = ['Accuracy', 'F1 Score']
            ylabel = 'Score'
            ylim = [0, 1.05]
        else:
            metrics = ['r2', 'mae']
            metric_labels = ['R² Score', 'MAE']
            ylabel = 'Score'
            ylim = None
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(max(6, 2 * len(metrics) + 4), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        models = ['ML Baseline', 'MA-ICL\n(Pre-train)', 'MA-ICL\n(Post-train)']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        if llm_only_maicl:
            models.append('LLM Mechanism')
            colors.append('#9b59b6') # Purple color for LLM Mechanism
            
        if vanilla_llm_metrics:
            models.append('Vanilla LLM')
            colors.append('#f1c40f') # Yellow color for Vanilla LLM
        
        for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            
            # Get values, handling missing data
            ml_val = ml_baseline.get(metric, 0)
            pre_val = pre_maicl.get(metric, 0)
            post_val = post_maicl.get(metric, 0)
            
            values = [ml_val, pre_val, post_val]
            
            if llm_only_maicl:
                llm_val = llm_only_maicl.get(metric, 0)
                values.append(llm_val)
                
            if vanilla_llm_metrics:
                vanilla_val = vanilla_llm_metrics.get(metric, 0)
                values.append(vanilla_val)
            
            bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                
                # Position text correctly for negative values
                if height < 0:
                    text_va = 'top'
                    text_y = height - (abs(height) * 0.05 if abs(height) > 0 else 0.01) 
                else:
                    text_va = 'bottom'
                    text_y = height + (height * 0.01 if height > 0 else 0.01)

                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va=text_va, fontsize=10, fontweight='bold')
            
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(metric_label, fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            if ylim:
                ax.set_ylim(ylim)
            
            # Rotate x labels slightly for better readability
            ax.tick_params(axis='x', labelsize=10)
        
        fig.suptitle('Performance Comparison: ML Baseline vs MA-ICL', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved performance comparison to {output_path}")
        
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to plot performance comparison: {e}")


def plot_scatter_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str, output_path: Optional[str] = None,
                            r2: Optional[float] = None, mae: Optional[float] = None,
                            mse: Optional[float] = None):
    """Plot scatter plot of predicted vs actual values for regression"""
    if not _HAS_MPL:
        logger.warning("Matplotlib not available, skipping scatter plot")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=60, color='#3498db', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add metrics to title
        metrics_str = ""
        if r2 is not None:
            metrics_str += f"R²={r2:.4f}"
        if mae is not None:
            metrics_str += f", MAE={mae:.4f}" if metrics_str else f"MAE={mae:.4f}"
        if mse is not None:
            metrics_str += f", MSE={mse:.4f}" if metrics_str else f"MSE={mse:.4f}"
        
        if metrics_str:
            title = f"{title}\n{metrics_str}"
        
        ax.set_xlabel('True Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved scatter plot to {output_path}")
        
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Failed to plot scatter: {e}")


def create_result_visualizations(y_test: np.ndarray, 
                                 ml_baseline_metrics: Dict[str, Any],
                                 pre_metrics: Dict[str, Any],
                                 post_metrics: Dict[str, Any],
                                 task_type: str = "classification",
                                 class_names: Optional[List[str]] = None,
                                 output_dir: Optional[str] = None,
                                 llm_only_metrics: Optional[Dict[str, Any]] = None,
                                 vanilla_llm_metrics: Optional[Dict[str, Any]] = None):
    """Create all result visualizations (confusion matrices + performance comparison + scatter plots)
    
    Args:
        y_test: True labels
        ml_baseline_metrics: ML baseline metrics with predictions
        pre_metrics: Pre-training MA-ICL metrics with predictions
        post_metrics: Post-training MA-ICL metrics with predictions
        task_type: "classification" or "regression"
        class_names: List of class names for classification
        output_dir: Output directory for visualizations
        llm_only_metrics: Optional metrics from LLM-only evaluation (excluding ML)
        vanilla_llm_metrics: Optional metrics from Vanilla LLM baseline
    """
    if output_dir is None:
        from maicl_lib_v2 import OUTPUT_DIR
        output_dir = OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Performance comparison plot
    perf_plot_path = os.path.join(output_dir, "performance_comparison.png")
    plot_performance_comparison(ml_baseline_metrics, pre_metrics, post_metrics, 
                                task_type=task_type, output_path=perf_plot_path,
                                llm_only_maicl=llm_only_metrics,
                                vanilla_llm_metrics=vanilla_llm_metrics)
    
    # Scatter plots for regression
    if task_type == "regression":
        # ML baseline scatter plot
        if "predictions" in ml_baseline_metrics:
            y_pred_ml = np.array(ml_baseline_metrics["predictions"])
            ml_r2 = ml_baseline_metrics.get('r2', None)
            ml_mae = ml_baseline_metrics.get('mae', None)
            ml_mse = ml_baseline_metrics.get('mse', None)
            ml_scatter_path = os.path.join(output_dir, "scatter_ml_baseline.png")
            plot_scatter_predictions(y_test, y_pred_ml, "ML Baseline Predictions",
                                    output_path=ml_scatter_path, r2=ml_r2, mae=ml_mae, mse=ml_mse)
        
        # LLM-only scatter plot (if available)
        if llm_only_metrics is not None and "predictions" in llm_only_metrics:
            y_pred_llm_only = np.array(llm_only_metrics["predictions"])
            llm_r2 = llm_only_metrics.get('r2', None)
            llm_mae = llm_only_metrics.get('mae', None)
            llm_mse = llm_only_metrics.get('mse', None)
            llm_scatter_path = os.path.join(output_dir, "scatter_llm_only.png")
            plot_scatter_predictions(y_test, y_pred_llm_only, "LLM-only Mechanism Predictions",
                                    output_path=llm_scatter_path, r2=llm_r2, mae=llm_mae, mse=llm_mse)
        
        # Vanilla LLM scatter plot (if available)
        if vanilla_llm_metrics is not None and "predictions" in vanilla_llm_metrics:
            y_pred_vanilla = np.array(vanilla_llm_metrics["predictions"])
            vanilla_r2 = vanilla_llm_metrics.get('r2', None)
            vanilla_mae = vanilla_llm_metrics.get('mae', None)
            vanilla_mse = vanilla_llm_metrics.get('mse', None)
            vanilla_scatter_path = os.path.join(output_dir, "scatter_vanilla_llm.png")
            plot_scatter_predictions(y_test, y_pred_vanilla, "Vanilla LLM Predictions",
                                    output_path=vanilla_scatter_path, r2=vanilla_r2, mae=vanilla_mae, mse=vanilla_mse)
        
        # MA-ICL pre-training scatter plot
        if "predictions" in pre_metrics:
            y_pred_pre = np.array(pre_metrics["predictions"])
            pre_r2 = pre_metrics.get('r2', None)
            pre_mae = pre_metrics.get('mae', None)
            pre_mse = pre_metrics.get('mse', None)
            pre_scatter_path = os.path.join(output_dir, "scatter_maicl_pre.png")
            plot_scatter_predictions(y_test, y_pred_pre, "MA-ICL Pre-training Predictions",
                                    output_path=pre_scatter_path, r2=pre_r2, mae=pre_mae, mse=pre_mse)
        
        # MA-ICL post-training scatter plot
        if "predictions" in post_metrics:
            y_pred_post = np.array(post_metrics["predictions"])
            post_r2 = post_metrics.get('r2', None)
            post_mae = post_metrics.get('mae', None)
            post_mse = post_metrics.get('mse', None)
            post_scatter_path = os.path.join(output_dir, "scatter_maicl_post.png")
            plot_scatter_predictions(y_test, y_pred_post, "MA-ICL Post-training Predictions",
                                    output_path=post_scatter_path, r2=post_r2, mae=post_mae, mse=post_mse)
    
    # Confusion matrices for classification
    if task_type == "classification" and class_names is not None:
        try:
            # ML Baseline confusion matrix
            if "predictions" in ml_baseline_metrics:
                y_pred_ml = np.array(ml_baseline_metrics["predictions"])
                cm_ml_path = os.path.join(output_dir, "confusion_matrix_ml_baseline.png")
                plot_confusion_matrix(y_test, y_pred_ml, class_names, 
                                    title="ML Baseline - Confusion Matrix",
                                    output_path=cm_ml_path)
            
            # Pre-training MA-ICL confusion matrix
            if "predictions" in pre_metrics:
                y_pred_pre = np.array(pre_metrics["predictions"])
                cm_pre_path = os.path.join(output_dir, "confusion_matrix_maicl_pre.png")
                plot_confusion_matrix(y_test, y_pred_pre, class_names,
                                    title="MA-ICL (Pre-training) - Confusion Matrix",
                                    output_path=cm_pre_path)
            
            # Post-training MA-ICL confusion matrix
            if "predictions" in post_metrics:
                y_pred_post = np.array(post_metrics["predictions"])
                cm_post_path = os.path.join(output_dir, "confusion_matrix_maicl_post.png")
                plot_confusion_matrix(y_test, y_pred_post, class_names,
                                    title="MA-ICL (Post-training) - Confusion Matrix",
                                    output_path=cm_post_path)
            
            # LLM-only post-training confusion matrix (if provided)
            if llm_only_metrics is not None and "predictions" in llm_only_metrics:
                y_pred_llm_only = np.array(llm_only_metrics["predictions"])
                cm_llm_only_path = os.path.join(output_dir, "confusion_matrix_maicl_llm_only_post.png")
                plot_confusion_matrix(y_test, y_pred_llm_only, class_names,
                                    title="MA-ICL LLM-only (Post-training, no ML) - Confusion Matrix",
                                    output_path=cm_llm_only_path)
                logger.info(f"  ✓ Saved LLM-only confusion matrix to {cm_llm_only_path}")
        except Exception as e:
            logger.warning(f"Failed to create confusion matrices: {e}")
    
    logger.info(f"✓ Created visualizations in {output_dir}")


def parse_causal_graph(mechanism_text: str):
    """
    Parses the mechanism text to extract nodes and edges for the causal graph.
    Returns a list of nodes and a list of edges (tuples).
    """
    nodes = []
    edges = []
    
    # Regular expressions for parsing
    node_pattern = re.compile(r"^\s*-\s*([^:]+):")
    edge_pattern = re.compile(r"^\s*-\s*([^->]+)\s*->\s*([^:]+):")
    
    lines = mechanism_text.split('\n')
    section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if "## NODES" in line:
            section = "NODES"
            continue
        elif "## EDGES" in line:
            section = "EDGES"
            continue
        elif "## EXECUTION FLOW" in line or "EXECUTABLE FORMULA" in line:
            section = None
            continue
            
        if section == "NODES":
            match = node_pattern.match(line)
            if match:
                node_name = match.group(1).strip()
                nodes.append(node_name)
        
        elif section == "EDGES":
            match = edge_pattern.match(line)
            if match:
                source = match.group(1).strip()
                target = match.group(2).strip()
                edges.append((source, target))
                
    return nodes, edges


def visualize_mechanism_graph(mechanism_text: str, title: str, output_path: str):
    """
    Visualizes the causal graph from the mechanism text and saves it to a file.
    """
    if not _HAS_MPL or not _HAS_NX:
        if not _HAS_NX:
            logger.warning("NetworkX not available, skipping graph visualization")
        return

    try:
        nodes, edges = parse_causal_graph(mechanism_text)
        
        if not nodes and not edges:
            return

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue', alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        plt.title(title, fontsize=15)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.warning(f"Failed to visualize mechanism graph: {e}")


# =========================
# RESULT SAVING FUNCTIONS
# =========================

def export_training_artifacts(training_history: Dict[str, Any],
                              mechanisms: List[str],
                              mechanism_types: List[str],
                              mechanism_performance_snapshot: Dict[int, float],
                              task_type: str,
                              use_ml_mechanism: bool,
                              ml_mechanism: Optional[Any],
                              feature_cols: List[str],
                              output_dir: Optional[str] = None,
                              max_feature_importance_top: int = 20):
    """Export training history (JSON/CSV) and training plots
    
    Args:
        training_history: Training history dictionary
        mechanisms: List of mechanism strings
        mechanism_types: List of mechanism type strings
        mechanism_performance_snapshot: Dictionary mapping mechanism index to performance score
        task_type: "classification" or "regression"
        use_ml_mechanism: Whether ML mechanism is used
        ml_mechanism: ML mechanism object (optional)
        feature_cols: List of feature column names
        output_dir: Output directory (uses OUTPUT_DIR from maicl_lib_v2 if None)
        max_feature_importance_top: Maximum number of features to show in importance ranking
    """
    if output_dir is None:
        from maicl_lib_v2 import OUTPUT_DIR
        output_dir = OUTPUT_DIR
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        return
    
    # History JSON
    history_json = os.path.join(output_dir, "training_history.json")
    _safe_write_json(history_json, training_history)
    
    # Export mechanism interpretations
    export_mechanism_interpretations(
        mechanisms, mechanism_types, mechanism_performance_snapshot,
        training_history, task_type, use_ml_mechanism, ml_mechanism,
        feature_cols, output_dir, max_feature_importance_top
    )
    
    # History CSV (compact)
    csv_path = os.path.join(output_dir, "training_metrics.csv")
    try:
        fieldnames = ["iteration", "val_loss", "accepted_updates", "rejected_updates",
                      "val_accuracy", "val_f1", "val_r2", "val_mae", "llm_calls", "llm_batches"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(len(training_history["iterations"])):
                row = {
                    "iteration": training_history["iterations"][idx],
                    "val_loss": training_history["val_loss"][idx],
                    "accepted_updates": training_history["accepted_updates"][idx],
                    "rejected_updates": training_history["rejected_updates"][idx],
                    "val_accuracy": training_history["val_accuracy"][idx],
                    "val_f1": training_history["val_f1"][idx],
                    "val_r2": training_history["val_r2"][idx],
                    "val_mae": training_history["val_mae"][idx],
                    "llm_calls": training_history["llm_calls"][idx],
                    "llm_batches": training_history["llm_batches"][idx],
                }
                writer.writerow(row)
    except Exception:
        pass
    
    # Plots
    if _HAS_MPL:
        try:
            iters = training_history["iterations"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            # Loss
            axes[0].plot(iters, training_history["val_loss"], marker="o", label="val_loss")
            axes[0].set_title("Validation Loss")
            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            # Metrics
            if task_type == "classification":
                axes[1].plot(iters, training_history["val_accuracy"], marker="o", label="accuracy")
                axes[1].plot(iters, training_history["val_f1"], marker="s", label="f1")
                axes[1].set_ylim(0, 1)
                axes[1].set_ylabel("Score")
            else:
                ax1 = axes[1]
                ax2 = ax1.twinx()
                ax1.plot(iters, training_history["val_mae"], color="tab:orange", marker="o", label="MAE")
                ax2.plot(iters, training_history["val_r2"], color="tab:blue", marker="s", label="R2")
                ax1.set_ylabel("MAE", color="tab:orange")
                ax2.set_ylabel("R2", color="tab:blue")
            axes[1].set_title("Validation Metrics")
            axes[1].set_xlabel("Iteration")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="best")
            plt.tight_layout()
            png_path = os.path.join(output_dir, "training_curves.png")
            pdf_path = os.path.join(output_dir, "training_curves.pdf")
            plt.savefig(png_path, dpi=150)
            try:
                plt.savefig(pdf_path)
            except Exception:
                pass
            plt.close(fig)
        except Exception:
            pass


def export_mechanism_interpretations(mechanisms: List[str],
                                     mechanism_types: List[str],
                                     mechanism_performance_snapshot: Dict[int, float],
                                     training_history: Dict[str, Any],
                                     task_type: str,
                                     use_ml_mechanism: bool,
                                     ml_mechanism: Optional[Any],
                                     feature_cols: List[str],
                                     output_dir: str,
                                     max_feature_importance_top: int = 20):
    """Export human-readable mechanism interpretations"""
    try:
        interp_path = os.path.join(output_dir, "mechanism_interpretations.txt")
        with open(interp_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("LEARNED MECHANISMS - INTERPRETABLE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, (mech, mtype) in enumerate(zip(mechanisms, mechanism_types)):
                f.write(f"Mechanism {idx + 1} ({mtype.upper()})\n")
                f.write("-" * 80 + "\n")
                f.write(f"{mech}\n\n")
                
                # Add performance if available
                if idx in mechanism_performance_snapshot:
                    perf = mechanism_performance_snapshot[idx]
                    f.write(f"Performance Score: {perf:.4f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Add feature importance analysis
            if use_ml_mechanism and ml_mechanism is not None:
                if hasattr(ml_mechanism, '_model_info') and ml_mechanism._model_info:
                    f.write("ML Mechanism Feature Importance:\n")
                    info = ml_mechanism._model_info
                    if 'feature_importance' in info and info['feature_importance']:
                        imp = info['feature_importance']
                        if isinstance(imp, dict):
                            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:max_feature_importance_top]
                            for feat, score in sorted_imp:
                                f.write(f"  - {feat}: {score:.4f}\n")
                        elif isinstance(imp, (list, np.ndarray)):
                            imp_arr = np.array(imp)
                            if feature_cols and len(imp_arr) == len(feature_cols):
                                sorted_idx = np.argsort(imp_arr)[::-1][:max_feature_importance_top]
                                for idx in sorted_idx:
                                    f.write(f"  - {feature_cols[idx]}: {imp_arr[idx]:.4f}\n")
                    f.write("\n")
            
            f.write("Mechanism Evolution:\n")
            if 'accepted_updates' in training_history:
                total_accepted = sum(training_history['accepted_updates'])
                total_rejected = sum(training_history['rejected_updates'])
                f.write(f"  - Accepted updates: {total_accepted}\n")
                f.write(f"  - Rejected updates: {total_rejected}\n")
                if total_accepted + total_rejected > 0:
                    accept_rate = total_accepted / (total_accepted + total_rejected)
                    f.write(f"  - Acceptance rate: {accept_rate:.1%}\n")
        
        logger.info(f"  ✓ Saved mechanism interpretations to {interp_path}")
    except Exception as e:
        logger.warning(f"Failed to export mechanism interpretations: {e}")


def persist_iteration_artifacts(iteration: int, 
                                mech_snapshot: Dict[str, Any], 
                                metrics: Dict[str, Any],
                                output_dir: Optional[str] = None):
    """Save per-iteration mechanism and metric snapshots to output directory"""
    if output_dir is None:
        from maicl_lib_v2 import OUTPUT_DIR
        output_dir = OUTPUT_DIR
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        return
    
    # mechanisms text
    mech_txt_path = os.path.join(output_dir, f"mechanisms_iter_{iteration}.txt")
    try:
        content = []
        for i, (m, t) in enumerate(zip(mech_snapshot["mechanisms"], mech_snapshot["mechanism_types"])):
            content.append(f"[{t.upper()}] {m}")
            
            # Visualize the graph if it's an LLM mechanism (which typically contains the graph)
            if t.lower() == "llm":
                graph_path = os.path.join(output_dir, f"mechanism_graph_iter_{iteration}_mech_{i+1}.png")
                visualize_mechanism_graph(m, f"Mechanism {i+1} (Iter {iteration})", graph_path)
                
        _safe_write_text(mech_txt_path, "\n\n".join(content))
    except Exception as e:
        logger.warning(f"Failed to save mechanisms text or visualization: {e}")
    
    # JSONL append
    jsonl_path = os.path.join(output_dir, "mechanism_evolution.jsonl")
    _safe_append_jsonl(jsonl_path, {**mech_snapshot, "metrics": metrics})
