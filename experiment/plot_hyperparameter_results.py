#!/usr/bin/env python3
"""
Create performance comparison plots for hyperparameter search results.
Generates plots showing MAE and R² performance comparisons across models and configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def create_performance_plots(csv_path):
    """
    Create comprehensive performance comparison plots.
    
    Args:
        csv_path: Path to the CSV file with results
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Get output directory (same as CSV file location)
    output_dir = Path(csv_path).parent
    
    # Create a configuration label for easier identification
    df['config_label'] = (
        df['ml_mech'].astype(str) + '_' +
        df['top_k'].astype(str) + '_' +
        'nmech' + df['num_mechanisms_unknown'].astype(str) + '_' +
        df['acceptance_set'].astype(str) + '_' +
        df['no_scaling'].map({True: 'noscale', False: 'scale'})
    )
    
    # 1. R² and MAE comparison by ML mechanism
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² by ML mechanism
    sns.boxplot(data=df, x='ml_mech', y='post_r2', ax=axes[0], hue='ml_mech', palette='Set2', legend=False)
    axes[0].set_title('R² Performance by ML Mechanism', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('ML Mechanism', fontsize=11)
    axes[0].set_ylabel('R² Score', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Add mean markers
    for i, mech in enumerate(df['ml_mech'].unique()):
        mean_val = df[df['ml_mech'] == mech]['post_r2'].mean()
        axes[0].plot(i, mean_val, 'ro', markersize=8, label='Mean' if i == 0 else '')
    axes[0].legend()
    
    # MAE by ML mechanism
    sns.boxplot(data=df, x='ml_mech', y='post_mae', ax=axes[1], hue='ml_mech', palette='Set2', legend=False)
    axes[1].set_title('MAE Performance by ML Mechanism', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('ML Mechanism', fontsize=11)
    axes[1].set_ylabel('MAE', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Add mean markers
    for i, mech in enumerate(df['ml_mech'].unique()):
        mean_val = df[df['ml_mech'] == mech]['post_mae'].mean()
        axes[1].plot(i, mean_val, 'ro', markersize=8, label='Mean' if i == 0 else '')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_ml_mechanism.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_by_ml_mechanism.png'}")
    plt.close()
    
    # 2. R² and MAE comparison by Top K
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² by Top K
    sns.boxplot(data=df, x='top_k', y='post_r2', ax=axes[0], hue='top_k', palette='viridis', legend=False)
    axes[0].set_title('R² Performance by Top K', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Top K', fontsize=11)
    axes[0].set_ylabel('R² Score', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE by Top K
    sns.boxplot(data=df, x='top_k', y='post_mae', ax=axes[1], hue='top_k', palette='viridis', legend=False)
    axes[1].set_title('MAE Performance by Top K', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Top K', fontsize=11)
    axes[1].set_ylabel('MAE', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_topk.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_by_topk.png'}")
    plt.close()
    
    # 3. R² vs MAE scatter plot colored by ML mechanism
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for mech in df['ml_mech'].unique():
        subset = df[df['ml_mech'] == mech]
        ax.scatter(subset['post_mae'], subset['post_r2'], 
                  label=mech, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('MAE (lower is better)', fontsize=11, fontweight='bold')
    ax.set_ylabel('R² Score (higher is better)', fontsize=11, fontweight='bold')
    ax.set_title('R² vs MAE Performance Comparison', fontsize=12, fontweight='bold')
    ax.legend(title='ML Mechanism', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_vs_mae_scatter.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'r2_vs_mae_scatter.png'}")
    plt.close()
    
    # 4. Performance by acceptance set (validation vs test)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² by acceptance set
    sns.boxplot(data=df, x='acceptance_set', y='post_r2', ax=axes[0], hue='acceptance_set', palette='coolwarm', legend=False)
    axes[0].set_title('R² Performance by Acceptance Set', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Acceptance Set', fontsize=11)
    axes[0].set_ylabel('R² Score', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE by acceptance set
    sns.boxplot(data=df, x='acceptance_set', y='post_mae', ax=axes[1], hue='acceptance_set', palette='coolwarm', legend=False)
    axes[1].set_title('MAE Performance by Acceptance Set', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Acceptance Set', fontsize=11)
    axes[1].set_ylabel('MAE', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_acceptance_set.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_by_acceptance_set.png'}")
    plt.close()
    
    # 5. Performance by number of mechanisms and scaling
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # R² by num_mechanisms_unknown
    sns.boxplot(data=df, x='num_mechanisms_unknown', y='post_r2', ax=axes[0, 0], hue='num_mechanisms_unknown', palette='pastel', legend=False)
    axes[0, 0].set_title('R² by Number of Mechanisms', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Mechanisms', fontsize=10)
    axes[0, 0].set_ylabel('R² Score', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE by num_mechanisms_unknown
    sns.boxplot(data=df, x='num_mechanisms_unknown', y='post_mae', ax=axes[0, 1], hue='num_mechanisms_unknown', palette='pastel', legend=False)
    axes[0, 1].set_title('MAE by Number of Mechanisms', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Mechanisms', fontsize=10)
    axes[0, 1].set_ylabel('MAE', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # R² by no_scaling
    sns.boxplot(data=df, x='no_scaling', y='post_r2', ax=axes[1, 0], hue='no_scaling', palette='Set3', legend=False)
    axes[1, 0].set_title('R² by Scaling (False=scaled, True=no scaling)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('No Scaling', fontsize=10)
    axes[1, 0].set_ylabel('R² Score', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE by no_scaling
    sns.boxplot(data=df, x='no_scaling', y='post_mae', ax=axes[1, 1], hue='no_scaling', palette='Set3', legend=False)
    axes[1, 1].set_title('MAE by Scaling (False=scaled, True=no scaling)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('No Scaling', fontsize=10)
    axes[1, 1].set_ylabel('MAE', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_mechanisms_and_scaling.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_by_mechanisms_and_scaling.png'}")
    plt.close()
    
    # 6. Heatmap of average R² by ML mechanism and Top K
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² heatmap
    pivot_r2 = df.pivot_table(values='post_r2', index='ml_mech', columns='top_k', aggfunc='mean')
    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Mean R²'})
    axes[0].set_title('Average R² by ML Mechanism and Top K', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Top K', fontsize=11)
    axes[0].set_ylabel('ML Mechanism', fontsize=11)
    
    # MAE heatmap
    pivot_mae = df.pivot_table(values='post_mae', index='ml_mech', columns='top_k', aggfunc='mean')
    sns.heatmap(pivot_mae, annot=True, fmt='.3f', cmap='YlGnBu_r', ax=axes[1], cbar_kws={'label': 'Mean MAE'})
    axes[1].set_title('Average MAE by ML Mechanism and Top K', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Top K', fontsize=11)
    axes[1].set_ylabel('ML Mechanism', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_heatmap.png'}")
    plt.close()
    
    # 7. Top 20 configurations bar chart
    top_configs = df.nlargest(20, 'post_r2')
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # R² for top 20
    axes[0].barh(range(len(top_configs)), top_configs['post_r2'], 
                 color=['green' if x == 'linear' else 'blue' for x in top_configs['ml_mech']])
    axes[0].set_yticks(range(len(top_configs)))
    axes[0].set_yticklabels([f"{row['ml_mech']}_k{row['top_k']}_nmech{row['num_mechanisms_unknown']}_{row['acceptance_set']}" 
                            for _, row in top_configs.iterrows()], fontsize=8)
    axes[0].set_xlabel('R² Score', fontsize=11, fontweight='bold')
    axes[0].set_title('Top 20 Configurations by R² Score', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    
    # Add value labels
    for i, val in enumerate(top_configs['post_r2']):
        axes[0].text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=8)
    
    # MAE for top 20
    axes[1].barh(range(len(top_configs)), top_configs['post_mae'], 
                 color=['green' if x == 'linear' else 'blue' for x in top_configs['ml_mech']])
    axes[1].set_yticks(range(len(top_configs)))
    axes[1].set_yticklabels([f"{row['ml_mech']}_k{row['top_k']}_nmech{row['num_mechanisms_unknown']}_{row['acceptance_set']}" 
                            for _, row in top_configs.iterrows()], fontsize=8)
    axes[1].set_xlabel('MAE', fontsize=11, fontweight='bold')
    axes[1].set_title('MAE for Top 20 Configurations (by R²)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    
    # Add value labels
    for i, val in enumerate(top_configs['post_mae']):
        axes[1].text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top20_configurations.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'top20_configurations.png'}")
    plt.close()
    
    # 8. Combined performance comparison (all factors)
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create a grouped comparison
    comparison_data = []
    for _, row in df.iterrows():
        comparison_data.append({
            'ML_Mech': row['ml_mech'],
            'Top_K': f"K={row['top_k']}",
            'N_Mech': f"Mech={row['num_mechanisms_unknown']}",
            'Scaling': 'No Scale' if row['no_scaling'] else 'Scaled',
            'Acceptance': row['acceptance_set'],
            'R2': row['post_r2'],
            'MAE': row['post_mae']
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart for R²
    pivot_combined = comp_df.pivot_table(
        values='R2', 
        index=['ML_Mech', 'Top_K', 'N_Mech', 'Scaling'], 
        columns='Acceptance', 
        aggfunc='mean'
    )
    
    x_pos = np.arange(len(pivot_combined))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    if 'validation' in pivot_combined.columns and 'test' in pivot_combined.columns:
        ax.bar(x_pos - width/2, pivot_combined['validation'], width, 
               label='Validation', alpha=0.8, color='skyblue')
        ax.bar(x_pos + width/2, pivot_combined['test'], width, 
               label='Test', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Configuration', fontsize=11, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax.set_title('R² Performance Comparison Across All Configurations', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{idx[0]}_{idx[1]}_{idx[2]}_{idx[3]}" for idx in pivot_combined.index], 
                       rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_configurations_comparison.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'all_configurations_comparison.png'}")
    plt.close()
    
    print("\n" + "="*80)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    csv_path = "analysis_protein_expression_plate5_20251207_102528/regression_results_summary.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    create_performance_plots(csv_path)

