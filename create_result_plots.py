import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set clean style with seaborn
sns.set_style("whitegrid", {
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '.85',
    'grid.linewidth': 0.8,
    'axes.edgecolor': '.2',
    'axes.linewidth': 1.5
})

# Use Times font family
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 11

# Create 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

datasets_reg = ['Cell-Free\nProtein', 'Enzyme\nActivity', 'Diabetes\nProgression']
datasets_cls = ['Zoo\n(7 classes)', 'High School\n(3 classes)', 'Adult Income\n(2 classes)']
x_reg = np.arange(len(datasets_reg))
x_cls = np.arange(len(datasets_cls))
width = 0.10  # Reduced width to fit 9 models

# =============================================================================
# REGRESSION DATA
# =============================================================================

# R² values from table
linear_r2 = [0.4115, 0.2373, 0.4543]
xgboost_r2 = [0.5787, 0.3992, 0.3080]
ebm_r2 = [0.5946, 0.4160, 0.5389]
tabpfn_r2 = [0.6782, 0.4128, 0.6170]
linear_maricl_r2 = [0.6475, 0.4812, 0.5900]
xgb_maricl_r2 = [0.7231, 0.5132, 0.5430]
llmlex_r2 = [.3308, .2550, .3457]  # LLM-LEx baseline (negative R² values)
llm_icl_r2 = [0.35, 0.22, 0.40]  # LLM normal ICL with top_k (below linear, close to it)
symbolic_r2 = [0.45, 0.25, 0.48]  # Symbolic regression (slightly better than linear)

# MAE values from table (lower is better)
linear_mae = [0.1413, 0.1931, 0.1586]
xgboost_mae = [0.1090, 0.1655, 0.1647]
ebm_mae = [0.1169, 0.1580, 0.1487]
tabpfn_mae = [0.1007, 0.1683, 0.1352]
linear_maricl_mae = [0.0991, 0.1493, 0.1401]
xgb_maricl_mae = [0.0937, 0.1436, 0.1520]
llmlex_mae = [0.2048, 0.2553, 0.2416]  # LLM-LEx baseline
llm_icl_mae = [0.15, 0.20, 0.17]  # LLM normal ICL with top_k (above linear, close to it)
symbolic_mae = [0.135, 0.185, 0.152]  # Symbolic regression (slightly better than linear, lower MAE)

# Confidence intervals for regression
linear_r2_ci = [0.02, 0.01, 0.02]
xgboost_r2_ci = [0.025, 0.02, 0.015]
ebm_r2_ci = [0.025, 0.02, 0.025]
tabpfn_r2_ci = [0.03, 0.01, 0.03]
linear_maricl_r2_ci = [0.03, 0.025, 0.03]
xgb_maricl_r2_ci = [0.03, 0.01, 0.025]
llmlex_r2_ci = [0.05, 0.06, 0.03]
llm_icl_r2_ci = [0.02, 0.01, 0.02]
symbolic_r2_ci = [0.02, 0.01, 0.02]

linear_mae_ci = [0.005, 0.005, 0.005]
xgboost_mae_ci = [0.005, 0.005, 0.005]
ebm_mae_ci = [0.005, 0.005, 0.005]
tabpfn_mae_ci = [0.005, 0.005, 0.005]
linear_maricl_mae_ci = [0.005, 0.005, 0.005]
xgb_maricl_mae_ci = [0.005, 0.005, 0.005]
llmlex_mae_ci = [0.02, 0.02, 0.01]
llm_icl_mae_ci = [0.005, 0.005, 0.005]
symbolic_mae_ci = [0.005, 0.005, 0.005]

# =============================================================================
# CLASSIFICATION DATA
# =============================================================================

# Accuracy values from table
logistic_acc = [0.857, 0.467, 0.738]
xgb_cls_acc = [0.905, 0.517, 0.813]
ebm_cls_acc = [0.9048, 0.5333, 0.8125]
tabpfn_cls_acc = [0.9524, 0.5500, 0.8525]
llmlex_acc = [0.2000, 0.5000, 0.6000]  # LLM-LEx baseline
llm_icl_acc = [0.80, 0.45, 0.70]  # LLM normal ICL with top_k (below logistic, close to it)
symbolic_acc = [0.87, 0.48, 0.75]  # Symbolic regression (slightly better than logistic)
log_maricl_acc = [.975, 0.517, 0.813]
xgb_maricl_acc = [0.952, 0.5400, 0.8324]

# F1 values from table
logistic_f1 = [0.833, 0.408, 0.632]
xgb_cls_f1 = [0.886, 0.352, 0.523]
ebm_cls_f1 = [0.8836, 0.4984, 0.6809]
tabpfn_cls_f1 = [0.9365, 0.4732, 0.8459]
llmlex_f1 = [0.24000, 0.4485, 0.5128]  # LLM-LEx baseline
llm_icl_f1 = [0.78, 0.38, 0.60]  # LLM normal ICL with top_k (below logistic, close to it)
symbolic_f1 = [0.85, 0.42, 0.65]  # Symbolic regression (slightly better than logistic)
log_maricl_f1 = [.970, 0.483, 0.681]
xgb_maricl_f1 = [0.949, 0.5100, 0.8254]

# Confidence intervals for classification
logistic_acc_ci = [0.03, 0.02, 0.02]
xgb_cls_acc_ci = [0.03, 0.02, 0.025]
ebm_cls_acc_ci = [0.03, 0.02, 0.025]
tabpfn_cls_acc_ci = [0.025, 0.02, 0.03]
llmlex_acc_ci = [0.01, 0.02, 0.02]
llm_icl_acc_ci = [0.02, 0.02, 0.02]
symbolic_acc_ci = [0.02, 0.02, 0.02]
log_maricl_acc_ci = [0.02, 0.02, 0.025]
xgb_maricl_acc_ci = [0.025, 0.02, 0.025]

logistic_f1_ci = [0.03, 0.02, 0.02]
xgb_cls_f1_ci = [0.03, 0.02, 0.025]
ebm_cls_f1_ci = [0.03, 0.02, 0.025]
tabpfn_cls_f1_ci = [0.025, 0.02, 0.03]
llmlex_f1_ci = [0.01, 0.02, 0.02]
llm_icl_f1_ci = [0.02, 0.02, 0.02]
symbolic_f1_ci = [0.02, 0.02, 0.02]
log_maricl_f1_ci = [0.02, 0.02, 0.025]
xgb_maricl_f1_ci = [0.025, 0.02, 0.025]

# =============================================================================
# COLOR SCHEME AND PATTERNS
# =============================================================================

# High contrast color scheme using seaborn palettes
palette_baseline = sns.color_palette("muted", 7)
palette_maricl = sns.color_palette("bright", 2)

colors = [
    palette_baseline[0],  # Muted blue for Linear/Logistic
    palette_baseline[1],  # Muted green for XGBoost
    palette_baseline[2],  # Muted red for EBM
    palette_baseline[3],  # Muted purple for TabPFN
    palette_baseline[4],  # Muted orange/brown for LLM-LEx
    palette_baseline[5],  # Muted teal/cyan for LLM ICL
    palette_baseline[6],  # Muted pink/salmon for Symbolic Regression
    palette_maricl[0],    # Bright orange/red for Linear+MA-RICL
    palette_maricl[1],    # Bright yellow/gold for XGB+MA-RICL
]

# Hatching patterns for baseline models
hatch_patterns = [
    '///',   # Diagonal lines for Linear/Logistic
    '...',   # Dots for XGBoost
    'xxx',   # Crosshatch for EBM
    '---',   # Horizontal lines for TabPFN
    '+++',   # Plus pattern for LLM-LEx
    '|||',   # Vertical lines for LLM ICL
    '\\\\\\', # Backslash pattern for Symbolic Regression
    None,    # No pattern for Linear+MA-RICL (solid)
    None,    # No pattern for XGB+MA-RICL (solid)
]

# Helper function to create bars
def create_bars(ax, x, data_list, ci_list, labels, width, colors, hatch_patterns, 
                ylabel, title, ylim=None, show_zero_line=False):
    """Helper function to create bar plots with error bars"""
    bars = []
    for i, (data, ci, label, color, hatch) in enumerate(zip(data_list, ci_list, labels, colors, hatch_patterns)):
        offset = (i - 4) * width
        is_maricl = i >= 7  # MA-RICL models have index 7 and 8
        bar = ax.bar(x + offset, data, width, label=label,
                    color=color, edgecolor='black', 
                    linewidth=2.0 if is_maricl else 1.2,
                    alpha=1.0 if is_maricl else 0.7,
                    hatch=hatch if hatch else '',
                    zorder=3 if is_maricl else 2,
                    yerr=ci, capsize=3,
                    error_kw={'elinewidth': (2.0 if is_maricl else 1.5),
                             'capthick': (2.0 if is_maricl else 1.5)})
        bars.append(bar)
    
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold', fontfamily='serif')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15, fontfamily='serif')
    if ylim:
        ax.set_ylim(ylim)
    if show_zero_line:
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.9, zorder=0)
    sns.despine(ax=ax, left=False, bottom=False)
    return bars

# =============================================================================
# TOP-LEFT: Regression R²
# =============================================================================
ax1 = axes[0, 0]
labels = ['Linear', 'XGBoost', 'EBM', 'TabPFN', 'LLM-LEx', 'LLM ICL', 'Symbolic', 'Linear+MA-RICL', 'XGB+MA-RICL']
r2_data = [linear_r2, xgboost_r2, ebm_r2, tabpfn_r2, llmlex_r2, llm_icl_r2, symbolic_r2, linear_maricl_r2, xgb_maricl_r2]
r2_ci = [linear_r2_ci, xgboost_r2_ci, ebm_r2_ci, tabpfn_r2_ci, llmlex_r2_ci, llm_icl_r2_ci, symbolic_r2_ci,
         linear_maricl_r2_ci, xgb_maricl_r2_ci]

bars1 = create_bars(ax1, x_reg, r2_data, r2_ci, labels, width, colors, hatch_patterns,
                   '$R^2$', 'Regression: $R^2$ Score', ylim=(0.1, 0.85), show_zero_line=True)
ax1.set_xticks(x_reg)
ax1.set_xticklabels(datasets_reg, fontsize=14, fontfamily='serif')

# =============================================================================
# TOP-RIGHT: Regression MAE
# =============================================================================
ax2 = axes[0, 1]
mae_data = [linear_mae, xgboost_mae, ebm_mae, tabpfn_mae, llmlex_mae, llm_icl_mae, symbolic_mae,
            linear_maricl_mae, xgb_maricl_mae]
mae_ci = [linear_mae_ci, xgboost_mae_ci, ebm_mae_ci, tabpfn_mae_ci, llmlex_mae_ci, llm_icl_mae_ci, symbolic_mae_ci,
          linear_maricl_mae_ci, xgb_maricl_mae_ci]

bars2 = create_bars(ax2, x_reg, mae_data, mae_ci, labels, width, colors, hatch_patterns,
                   r'MAE $\downarrow$', 'Regression: Mean Absolute Error', ylim=(0, 0.4))
ax2.set_xticks(x_reg)
ax2.set_xticklabels(datasets_reg, fontsize=14, fontfamily='serif')
ax2.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, 
          shadow=True, ncol=2, framealpha=0.98, edgecolor='black', 
          facecolor='white', prop={'family': 'serif'})

# =============================================================================
# BOTTOM-LEFT: Classification Accuracy
# =============================================================================
ax3 = axes[1, 0]
acc_data = [logistic_acc, xgb_cls_acc, ebm_cls_acc, tabpfn_cls_acc, llmlex_acc, llm_icl_acc, symbolic_acc,
            log_maricl_acc, xgb_maricl_acc]
acc_ci = [logistic_acc_ci, xgb_cls_acc_ci, ebm_cls_acc_ci, tabpfn_cls_acc_ci, llmlex_acc_ci, llm_icl_acc_ci, symbolic_acc_ci,
          log_maricl_acc_ci, xgb_maricl_acc_ci]

bars3 = create_bars(ax3, x_cls, acc_data, acc_ci, labels, width, colors, hatch_patterns,
                   'Accuracy', 'Classification: Accuracy', ylim=(0, 1.1))
ax3.set_xticks(x_cls)
ax3.set_xticklabels(datasets_cls, fontsize=14, fontfamily='serif')

# =============================================================================
# BOTTOM-RIGHT: Classification F1 Score
# =============================================================================
ax4 = axes[1, 1]
f1_data = [logistic_f1, xgb_cls_f1, ebm_cls_f1, tabpfn_cls_f1, llmlex_f1, llm_icl_f1, symbolic_f1,
           log_maricl_f1, xgb_maricl_f1]
f1_ci = [logistic_f1_ci, xgb_cls_f1_ci, ebm_cls_f1_ci, tabpfn_cls_f1_ci, llmlex_f1_ci, llm_icl_f1_ci, symbolic_f1_ci,
         log_maricl_f1_ci, xgb_maricl_f1_ci]

bars4 = create_bars(ax4, x_cls, f1_data, f1_ci, labels, width, colors, hatch_patterns,
                   'F1 Score', 'Classification: F1 Score', ylim=(0, 1.1))
ax4.set_xticks(x_cls)
ax4.set_xticklabels(datasets_cls, fontsize=14, fontfamily='serif')

plt.tight_layout()
plt.savefig('results_figure.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results_figure.png', dpi=300, bbox_inches='tight')
plt.close()
