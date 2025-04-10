import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from config import FIGURES_DIR

def correlation_heatmap(df, cols):
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True)
    plt.title("Correlation Heatmap")
    plt.savefig(f'{FIGURES_DIR}/correlation_heatmap.png')
