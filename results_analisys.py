import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from src.data_loader import load_and_clean_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

df = pd.read_parquet("results/optuna_trials/all_trials_results.parquet")
X, y, cat_features, num_features = load_and_clean_data()


best_index = df['mean_log_loss'].idxmin()
best_predictions = df.loc[best_index, 'predictions']


predictions = pd.DataFrame(columns=['fid', 'predictions'])
predictions["fid"] = y.index
predictions["predictions"] = best_predictions
predictions.to_csv("manuscript/gis_figure/best_predictions.csv", index=False)


# 1. Prepare the data: Get the best (min) log loss per model/resampling pair
plot_df = df.groupby(['model', 'resampling'])['mean_log_loss'].min().reset_index()

# 2. Define the order of models (sorted by performance in the 'None' baseline)
# This ensures all subplots have the same model order for easy comparison.
baseline_order = (
    plot_df[plot_df['resampling'] == 'None']
    .sort_values('mean_log_loss')['model']
    .tolist()
)
if not baseline_order:
    baseline_order = (
        plot_df.groupby('model')['mean_log_loss']
        .min()
        .sort_values()
        .index
        .tolist()
    )

# 3. Setup the 4-panel vertical figure
resampling_methods = [method for method in ['None', 'SMOTE', 'ADASYN', 'RandomUnderSampler'] if method in set(plot_df['resampling'])]
fig, axes = plt.subplots(nrows=len(resampling_methods), ncols=1, figsize=(12, 4 * len(resampling_methods)), sharex=True, sharey=True)
axes = np.atleast_1d(axes)

# Define a color palette for consistency across bars
palette = sns.color_palette("viridis", n_colors=len(baseline_order))

for i, method in enumerate(resampling_methods):
    ax = axes[i]
    subset = plot_df[plot_df['resampling'] == method]
    
    sns.barplot(
        data=subset,
        x='model',
        y='mean_log_loss',
        order=baseline_order,
        palette="Blues_d", # Use a single color shade to keep focus on the values
        ax=ax
    )
    
    # Labeling and styling
    ax.set_title(f'Resampling Method: {method}', fontsize=14, fontweight='bold', loc='left')
    ax.set_ylabel('Best Log Loss')
    ax.set_xlabel('') # Clear individual x-labels to avoid clutter
    
    # Add the value on top of the bars for precision (optional for papers)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=9)

# Final formatting
plt.xlabel('Model Type', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.ylim(0, 1)
# Save with high DPI for publication
# plt.savefig('resampling_comparison_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

if 'k' not in df.columns:
    df['k'] = np.nan

# Save the DataFrame to a parquet file
# Note: This requires the 'pyarrow' or 'fastparquet' library to be installed.
df.to_parquet('results/oof_predictions/all_trials_results.parquet', index=False)
