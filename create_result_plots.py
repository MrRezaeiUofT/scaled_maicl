import matplotlib.pyplot as plt
import numpy as np

# Set clean style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# =============================================================================
# Left: Regression Results (R² scores)
# =============================================================================
ax1 = axes[0]

datasets_reg = ['Cell-Free\nProtein', 'Enzyme\nActivity', 'Diabetes\nProgression']
x_reg = np.arange(len(datasets_reg))
width = 0.15

# R² values from table
linear = [0.4115, 0.0373, 0.4543]
xgboost = [0.5787, 0.3992, 0.3080]
ebm = [0.5946, 0.4160, 0.5389]
tabpfn = [0.6782, 0.1128, 0.6170]
linear_maricl = [0.7231, 0.2132, 0.5430]
xgb_maricl = [0.6475, 0.4812, 0.5900]

colors = ['#bdbdbd', '#969696', '#737373', '#525252', '#2196F3', '#1565C0']

bars1 = ax1.bar(x_reg - 2.5*width, linear, width, label='Linear', color=colors[0], edgecolor='white')
bars2 = ax1.bar(x_reg - 1.5*width, xgboost, width, label='XGBoost', color=colors[1], edgecolor='white')
bars3 = ax1.bar(x_reg - 0.5*width, ebm, width, label='EBM', color=colors[2], edgecolor='white')
bars4 = ax1.bar(x_reg + 0.5*width, tabpfn, width, label='TabPFN', color=colors[3], edgecolor='white')
bars5 = ax1.bar(x_reg + 1.5*width, linear_maricl, width, label='Linear+MA-RICL', color=colors[4], edgecolor='white')
bars6 = ax1.bar(x_reg + 2.5*width, xgb_maricl, width, label='XGB+MA-RICL', color=colors[5], edgecolor='white')

ax1.set_ylabel('$R^2$', fontsize=13, fontweight='bold')
ax1.set_xticks(x_reg)
ax1.set_xticklabels(datasets_reg, fontsize=11)
ax1.set_ylim(0, 0.85)
ax1.set_title('Regression Performance', fontsize=14, fontweight='bold', pad=10)
ax1.legend(loc='upper right', fontsize=9, frameon=False, ncol=2)

# =============================================================================
# Right: Classification Results (Accuracy)
# =============================================================================
ax2 = axes[1]

datasets_cls = ['Zoo\n(7 classes)', 'High School\n(3 classes)', 'Adult Income\n(2 classes)']
x_cls = np.arange(len(datasets_cls))

# Accuracy values from table
logistic = [0.857, 0.467, 0.738]
xgb_cls = [0.905, 0.517, 0.813]
ebm_cls = [0.9048, 0.5333, 0.8125]
tabpfn_cls = [0.9524, 0.5500, 0.8525]
log_maricl = [1.000, 0.517, 0.813]
xgb_maricl_cls = [0.952, 0.5400, 0.8324]

bars1 = ax2.bar(x_cls - 2.5*width, logistic, width, label='Logistic', color=colors[0], edgecolor='white')
bars2 = ax2.bar(x_cls - 1.5*width, xgb_cls, width, label='XGBoost', color=colors[1], edgecolor='white')
bars3 = ax2.bar(x_cls - 0.5*width, ebm_cls, width, label='EBM', color=colors[2], edgecolor='white')
bars4 = ax2.bar(x_cls + 0.5*width, tabpfn_cls, width, label='TabPFN', color=colors[3], edgecolor='white')
bars5 = ax2.bar(x_cls + 1.5*width, log_maricl, width, label='Logistic+MA-RICL', color=colors[4], edgecolor='white')
bars6 = ax2.bar(x_cls + 2.5*width, xgb_maricl_cls, width, label='XGB+MA-RICL', color=colors[5], edgecolor='white')

ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax2.set_xticks(x_cls)
ax2.set_xticklabels(datasets_cls, fontsize=11)
ax2.set_ylim(0, 1.1)
ax2.set_title('Classification Performance', fontsize=14, fontweight='bold', pad=10)
ax2.legend(loc='upper right', fontsize=9, frameon=False, ncol=2)

plt.tight_layout()
plt.savefig('results_figure.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results_figure.png', dpi=300, bbox_inches='tight')
plt.show()