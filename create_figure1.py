"""
Figure 1: MA-ICL Performance Improvements Across Datasets and Models

This script creates a comprehensive visualization showing MA-ICL's superiority
across regression and classification tasks with different base models.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# Set seaborn style for modern, clean aesthetics
sns.set_style("whitegrid", {
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '.9',
    'grid.linewidth': 0.8,
    'axes.edgecolor': '.3',
    'axes.linewidth': 1.0
})

# Set publication-quality style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Data from experiments
# Regression data
regression_data = {
    'Protein Expression': {
        'Linear': {'baseline': 0.4115, 'maicl': 0.7231, 'improvement': 75.7},
        'XGBoost': {'baseline': 0.5787, 'maicl': 0.6475, 'improvement': 11.9}
    },
    'Enzyme Activity': {
        'Linear': {'baseline': 0.0373, 'maicl': 0.0336, 'improvement': -9.9},
        'XGBoost': {'baseline': 0.3992, 'maicl': 0.4487, 'improvement': 12.4}
    },
    'Diabetes': {
        'Linear': {'baseline': 0.4543, 'maicl': 0.5920, 'improvement': 30.3},
        'XGBoost': {'baseline': 0.3080, 'maicl': 0.4947, 'improvement': 60.6}
    }
}

# Classification data (using accuracy)
classification_data = {
    'Zoo Animals': {
        'Logistic': {'baseline': 0.857, 'maicl': 1.000, 'improvement': 14.3},
        'XGBoost': {'baseline': 0.905, 'maicl': 0.952, 'improvement': 4.8}
    },
    'Social Groups': {
        'Logistic': {'baseline': 0.467, 'maicl': 0.517, 'improvement': 5.0},
        'XGBoost': {'baseline': 0.517, 'maicl': 0.500, 'improvement': -1.7}
    }
}

# Create figure with custom layout and background
fig = plt.figure(figsize=(14, 11), facecolor='white')
gs = GridSpec(3, 1, figure=fig, hspace=0.5, wspace=0.3, 
              height_ratios=[1.2, 1.2, 1.0])

# Color schemes using seaborn palettes
palette = sns.color_palette("husl", 8)
colors = {
    'baseline': sns.desaturate(palette[0], 0.5),  # Muted blue-gray
    'maicl': palette[2],                          # Vibrant teal
    'linear': palette[4],                         # Deep blue
    'xgboost': palette[6],                        # Forest green
    'logistic': palette[4],                       # Same as linear
}

# Alternative: use a more professional palette
professional_palette = sns.color_palette("Set2")
colors = {
    'baseline': sns.color_palette("pastel")[0],   # Light blue
    'maicl': professional_palette[1],              # Teal
    'linear': sns.color_palette("deep")[0],       # Deep blue
    'xgboost': sns.color_palette("deep")[2],      # Deep green
    'logistic': sns.color_palette("deep")[0],     # Same as linear
}

# ============================================================================
# Panel A: Regression Performance Comparison (R² scores)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

datasets_reg = list(regression_data.keys())
models = ['Linear', 'XGBoost']
x = np.arange(len(datasets_reg))
width = 0.18

for i, model in enumerate(models):
    baseline_scores = [regression_data[d][model]['baseline'] for d in datasets_reg]
    maicl_scores = [regression_data[d][model]['maicl'] for d in datasets_reg]
    
    offset = -width*1.5 if model == 'Linear' else width*0.5
    
    # Baseline bars with seaborn styling
    bars1 = ax1.bar(x + offset, baseline_scores, width, 
                    label=f'{model} Baseline',
                    color=colors['baseline'], alpha=0.6, 
                    edgecolor='white', linewidth=1.2, zorder=2)
    
    # MA-ICL bars with enhanced styling
    model_color = colors['linear'] if model == 'Linear' else colors['xgboost']
    bars2 = ax1.bar(x + offset + width, maicl_scores, width,
                    label=f'{model} + MA-ICL',
                    color=model_color, alpha=0.9, 
                    edgecolor='white', linewidth=1.2, zorder=3)
    
    # Add improvement percentage annotations with better styling
    for j, (d, b1, b2) in enumerate(zip(datasets_reg, bars1, bars2)):
        improvement = regression_data[d][model]['improvement']
        if improvement > 0:
            y_pos = max(b1.get_height(), b2.get_height()) + 0.05
            ax1.text(x[j] + offset + width/2, y_pos, f'+{improvement:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=model_color, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=model_color, alpha=0.8, linewidth=1))

ax1.set_ylabel('R² Score', fontweight='bold', fontsize=12)
ax1.set_title('(A) Regression Performance', 
              fontweight='bold', fontsize=13, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets_reg)
ax1.set_ylim(0, 1.1)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
          frameon=True, fancybox=True, shadow=True, ncol=4,
          framealpha=0.95, edgecolor='gray', facecolor='white')
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8, zorder=0)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1, zorder=0)
sns.despine(ax=ax1, left=False, bottom=False)

# ============================================================================
# Panel B: Classification Performance Comparison (Accuracy)
# ============================================================================
ax2 = fig.add_subplot(gs[1, :])

datasets_cls = list(classification_data.keys())
x_cls = np.arange(len(datasets_cls))

for i, model in enumerate(['Logistic', 'XGBoost']):
    baseline_scores = [classification_data[d][model]['baseline'] for d in datasets_cls]
    maicl_scores = [classification_data[d][model]['maicl'] for d in datasets_cls]
    
    offset = -width*1.5 if model == 'Logistic' else width*0.5
    
    # Baseline bars with seaborn styling
    bars1 = ax2.bar(x_cls + offset, baseline_scores, width,
                    label=f'{model} Baseline',
                    color=colors['baseline'], alpha=0.6, 
                    edgecolor='white', linewidth=1.2, zorder=2)
    
    # MA-ICL bars with enhanced styling
    model_color = colors['logistic'] if model == 'Logistic' else colors['xgboost']
    bars2 = ax2.bar(x_cls + offset + width, maicl_scores, width,
                    label=f'{model} + MA-ICL',
                    color=model_color, alpha=0.9, 
                    edgecolor='white', linewidth=1.2, zorder=3)
    
    # Add improvement annotations with better styling
    for j, (d, b1, b2) in enumerate(zip(datasets_cls, bars1, bars2)):
        improvement = classification_data[d][model]['improvement']
        y_pos = max(b1.get_height(), b2.get_height()) + 0.03
        sign = '+' if improvement > 0 else ''
        # Use seaborn colors for better aesthetics
        color_imp = sns.color_palette("RdYlGn", 3)[0] if improvement > 0 else sns.color_palette("RdYlGn", 3)[2]
        ax2.text(x_cls[j] + offset + width/2, y_pos, f'{sign}{improvement:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=color_imp,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color_imp, alpha=0.8, linewidth=1))

ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
ax2.set_title('(B) Classification Performance',
              fontweight='bold', fontsize=13, pad=15)
ax2.set_xticks(x_cls)
ax2.set_xticklabels(datasets_cls)
ax2.set_ylim(0, 1.1)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
          frameon=True, fancybox=True, shadow=True, ncol=4,
          framealpha=0.95, edgecolor='gray', facecolor='white')
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8, zorder=0)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1, zorder=0)
sns.despine(ax=ax2, left=False, bottom=False)

# ============================================================================
# Panel C: Improvement Summary Heatmap (Horizontal)
# ============================================================================
ax3 = fig.add_subplot(gs[2, :])

# Prepare data for heatmap (transposed for horizontal layout)
all_datasets = ['Protein\nExpression', 'Enzyme\nActivity', 'Diabetes', 
                'Zoo\nAnimals', 'Social\nGroups']
all_models = ['Linear/Logistic', 'XGBoost']

improvement_matrix = np.array([
    [75.7, 11.9],   # Protein
    [-9.9, 12.4],   # Enzyme
    [30.3, 60.6],   # Diabetes
    [14.3, 4.8],    # Zoo
    [5.0, -1.7]     # Social
])

# Transpose matrix for horizontal layout
improvement_matrix = improvement_matrix.T

# Create annotation matrix with % signs
annot_matrix = np.empty_like(improvement_matrix, dtype=object)
for i in range(len(all_models)):
    for j in range(len(all_datasets)):
        value = improvement_matrix[i, j]
        sign = '+' if value > 0 else ''
        annot_matrix[i, j] = f'{sign}{value:.1f}%'

# Create heatmap using seaborn for better styling
# Use a diverging colormap centered at 0
sns.heatmap(improvement_matrix, 
            xticklabels=all_datasets,
            yticklabels=all_models,
            annot=annot_matrix,
            fmt='',
            cmap='RdYlGn',
            center=0,
            vmin=-15,
            vmax=80,
            cbar_kws={'orientation': 'horizontal', 
                     'pad': 0.15, 
                     'fraction': 0.05,
                     'label': '% Improvement over Baseline',
                     'shrink': 0.8},
            linewidths=1.5,
            linecolor='white',
            square=False,
            ax=ax3,
            annot_kws={'fontsize': 10, 'fontweight': 'bold'},
            cbar_ax=None)

# Customize text colors for better readability
for i in range(len(all_models)):
    for j in range(len(all_datasets)):
        value = improvement_matrix[i, j]
        text_idx = i * len(all_datasets) + j
        if text_idx < len(ax3.texts):
            text = ax3.texts[text_idx]
            # Adjust text color for better readability
            if abs(value) > 40:
                text.set_color('white')
            else:
                text.set_color('black')

ax3.set_title('(C) Improvement Heatmap', 
              fontweight='bold', fontsize=13, pad=15)
ax3.set_ylabel('Base Model', fontweight='bold', fontsize=12)
ax3.set_xlabel('Dataset', fontweight='bold', fontsize=12)

# Adjust layout to prevent overlap
plt.tight_layout()

# Create Figures directory if it doesn't exist
os.makedirs('./Figures', exist_ok=True)

# Save figure
plt.savefig('./Figures/figure1_maicl_performance.pdf', dpi=300, bbox_inches='tight')
plt.savefig('./Figures/figure1_maicl_performance.png', dpi=300, bbox_inches='tight')

print("Figure 1 created successfully!")
print("Saved as: Figures/figure1_maicl_performance.pdf and Figures/figure1_maicl_performance.png")

# Display summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nRegression Tasks:")
for dataset, models_data in regression_data.items():
    print(f"\n{dataset}:")
    for model, data in models_data.items():
        print(f"  {model}: {data['baseline']:.4f} → {data['maicl']:.4f} ({data['improvement']:+.1f}%)")

print("\nClassification Tasks:")
for dataset, models_data in classification_data.items():
    print(f"\n{dataset}:")
    for model, data in models_data.items():
        print(f"  {model}: {data['baseline']:.4f} → {data['maicl']:.4f} ({data['improvement']:+.1f}%)")

# Calculate average improvements
reg_improvements = [regression_data[d][m]['improvement'] 
                   for d in regression_data for m in models]
cls_improvements = [classification_data[d][m]['improvement']
                   for d in classification_data for m in ['Logistic', 'XGBoost']]

print(f"\nAverage Regression Improvement: {np.mean(reg_improvements):.1f}%")
print(f"Average Classification Improvement: {np.mean(cls_improvements):.1f}%")
print(f"Overall Average Improvement: {np.mean(reg_improvements + cls_improvements):.1f}%")
