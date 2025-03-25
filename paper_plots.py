# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:52:29 2024

@author: Ahsan
"""

import numpy as np
import matplotlib.pyplot as plt

lambda_strain = np.load("Regularization_paper_results/variables/OASIS/AD3_1/lamda_strain.npy")
mu_shear = np.load("Regularization_paper_results/variables/OASIS/AD3_1/mu_shear.npy")
gradient_norm = np.load("Regularization_paper_results/variables/OASIS/AD3_1/gradient_norm.npy")
folding_plenty = np.load("Regularization_paper_results/variables/OASIS/AD3_1/folding_penalty.npy")
shear_E = np.load("Regularization_paper_results/variables/OASIS/AD3_1/shear_E.npy")
strain_E = np.load("Regularization_paper_results/variables/OASIS/AD3_1/strain_E.npy")
total_E = np.load("Regularization_paper_results/variables/OASIS/AD3_1/total_E.npy")
trace_E = np.load("Regularization_paper_results/variables/OASIS/AD3_1/trace_E.npy")



plt.scatter(gradient_norm,lambda_strain)
plt.scatter(gradient_norm,mu_shear)
plt.scatter(strain_E,lambda_strain)
plt.scatter(shear_E, mu_shear)
plt.scatter(total_E,lambda_strain)
plt.scatter(total_E,mu_shear)
plt.scatter(total_E,folding_plenty)
plt.scatter(shear_E,trace_E)
plt.scatter(strain_E,folding_plenty)
plt.scatter(shear_E,folding_plenty)


#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
lambda_strain = np.load("Regularization_paper_results/variables/OASIS/AD3_1/lamda_strain.npy")
mu_shear = np.load("Regularization_paper_results/variables/OASIS/AD3_1/mu_shear.npy")
gradient_norm = np.load("Regularization_paper_results/variables/OASIS/AD3_1/gradient_norm.npy")
folding_penalty = np.load("Regularization_paper_results/variables/OASIS/AD3_1/folding_penalty.npy")
shear_E = np.load("Regularization_paper_results/variables/OASIS/AD3_1/shear_E.npy")
strain_E = np.load("Regularization_paper_results/variables/OASIS/AD3_1/strain_E.npy")
total_E = np.load("Regularization_paper_results/variables/OASIS/AD3_1/total_E.npy")
trace_E = np.load("Regularization_paper_results/variables/OASIS/AD3_1/trace_E.npy")

# Set seaborn style
sns.set(style="whitegrid", palette="muted", font_scale=1.5)

# Plot each scatter plot separately
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.flatten()

# Scatter plots
axes[0].scatter(gradient_norm, lambda_strain, s=50, alpha=0.7)
axes[0].set_title("$\\lambda_{strain}$ vs $|| \\nabla u||$")
axes[0].set_xlabel(" $|| \\nabla u||$")
axes[0].set_ylabel("$\\lambda_{strain}$")

axes[1].scatter(gradient_norm, mu_shear, s=50, alpha=0.7)
axes[1].set_title("$\\mu_{shear}$ vs  $|| \\nabla u||$")
axes[1].set_xlabel(" $|| \\nabla u||$")
axes[1].set_ylabel("$\\mu_{shear}$")

axes[2].scatter(strain_E, lambda_strain, s=50, alpha=0.7)
axes[2].set_title("$\\lambda_{strain}$ vs $E_{strain}$")
axes[2].set_xlabel(" $E_{strain}$")
axes[2].set_ylabel("$\\lambda_{strain}$")

axes[3].scatter(shear_E, mu_shear, s=50, alpha=0.7)
axes[3].set_title("$\\mu_{shear}$ vs  $E_{shear}$")
axes[3].set_xlabel(" $E_{shear}$")
axes[3].set_ylabel("$\\mu_{shear}$")

plt.savefig("plt_set1.png", dpi=150)

#%%

# Set seaborn style
sns.set(style="whitegrid", palette="muted", font_scale=1.5)

# Plot each scatter plot separately
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.flatten()

axes[0].scatter(total_E, lambda_strain, s=50, alpha=0.7)
axes[0].set_title("$\\lambda_{strain}$ vs $E_{total}$")
axes[0].set_xlabel(" $E_{total}$")
axes[0].set_ylabel("$\\lambda_{strain}$")

axes[1].scatter(total_E, mu_shear, s=50, alpha=0.7)
axes[1].set_title("$\\mu_{shear}$ vs $E_{total}$")
axes[1].set_xlabel("$E_{total}$")
axes[1].set_ylabel("$\\mu_{shear}$")

axes[2].scatter(total_E, folding_penalty, s=50, alpha=0.7)
axes[2].set_title("Folding Penalty vs $E_{total}$")
axes[2].set_xlabel("$E_{total}$")
axes[2].set_ylabel("Folding Penalty")

axes[3].scatter(strain_E, trace_E, s=50, alpha=0.7)
axes[3].set_title("Trace Energy $E_{trace}$ vs Shear Energy $E_{shear}$")
axes[3].set_xlabel(" $E_{shear}$")
axes[3].set_ylabel("$E_{trace}$")

# Adjust layout and show plot
plt.tight_layout()
plt.savefig("plt_set2.png", dpi=150)
plt.show()

#%%
# Set seaborn style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# 3D Scatter plot for lambda_strain, mu_shear, and total_E
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot data
ax.scatter(lambda_strain, mu_shear, total_E, c='b', marker='o', alpha=0.7)

# Labels
ax.set_xlabel("$\\lambda_{strain}$", fontsize=12)
ax.set_ylabel("$\\mu_{shear}$", fontsize=12)
ax.set_zlabel("Total Energy $E_{total}$", fontsize=12)

# Title
ax.set_title("3D Plot of $\\lambda_{strain}$, $\\mu_{shear}$, and $E_{total}$", fontsize=14)

# Show plot
plt.tight_layout()
plt.show()
