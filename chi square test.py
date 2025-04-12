import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import stats
from scipy.stats import chi2_contingency


def normalized_crosstab(df, var1, var2, alpha=1):
    ct = pd.crosstab(df[var1], df[var2])

    ct_smoothed = (ct + alpha) / (ct.sum(axis=1).values[:, np.newaxis] + alpha*len(ct.columns))

    return ct,  ct_smoothed

def plot_categorical_comparison(df, var1, var2, plot_type='heatmap'):

    ct, ct_smoothed = normalized_crosstab(df, var1, var2)

    if plot_type == 'heatmap':
        plt.figure(figsize=(max(6, len(ct.columns)*1.5), max(6, len(ct.index)*1)))
        sns.heatmap(ct_smoothed, annot=True, fmt=".2f", cmap='coolwarm',
                   cbar_kws={'label': 'Proportion (smoothed)'})
        plt.title(f'{var1} vs {var2} (Row Proportions with Smoothing)')
    else:
        plt.figure(figsize=(10, 6))
        ct_normalized = ct.div(ct.sum(axis=1), axis=0)
        ct_normalized.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title(f'{var1} vs {var2} (Stacked Proportions)')
        plt.ylabel('Proportion')
        plt.legend(title=var2, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel(var2)
    plt.ylabel(var1)

    chi2, pval, dof, expected = chi2_contingency(ct)
    n = ct.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape)-1)))

    print(f" Chi-Square: χ²({dof}) = {chi2:.1f}, p = {pval:.4f}")
    print(f" Effect size: Cramér's V = {cramers_v:.2f}")

    if len(ct.columns) == 2 and len(ct.index) >= 2:
        print("\n Risk Ratios (reference=first category):")
        base_rate = ct_smoothed.iloc[0, 1]
        for i in range(1, len(ct.index)):
            rr = ct_smoothed.iloc[i, 1] / base_rate
            print(f"{ct.index[i]} vs {ct.index[0]}: RR = {rr:.2f}")
            
df = pd.read_csv("AllSamples.csv")

plot_categorical_comparison(df, 'Blood Group', 'Covid', plot_type='heatmap')
