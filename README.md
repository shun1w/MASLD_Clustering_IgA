# MASLD_Clustering_IgA

## Usage of `modelapplication.py`

The `Newmodelapply` class provides preprocessing, two-step clustering, and survival analysis utilities.

### Basic workflow
```python
from newmodelapplication import Newmodelapply

# 1) Prepare data and parameter lists
data = ...  # pandas DataFrame containing all parameters
parameters = [...]  # columns used for modeling
categorical_params = [...]  # subset of parameters treated as categorical

# 2) Initialize
model_app = Newmodelapply(data, parameters, categorical_params)

# 3) Preprocess (outlier removal + missing handling + standardization)
preprocessed = model_app.data_preprocessing(
    missing_value='imputation',  # or 'drop'
    outliner_sd=5
)

# 4) First clustering (fixed 10 clusters, visualization only)
model_app.first_clustering(random_state=42)

# 5) Second clustering (configurable clusters + feature plots)
result_df = model_app.second_clustering(
    n_clusters=5,
    n_param_show=10,
    random_state=42
)

# 6) Survival analysis (Nelson-Aalen + pairwise log-rank with Holm)
model_app.figure_logrank_nelson(
    time='time_column',
    event='event_column',
    event_name='Outcome Name',
    group='cluster',
    ylim=0.5,
    xlim=60
)
```

### Notes
- Outliers are removed per numerical column using the `outliner_sd` threshold.
- Missing values: `imputation` applies KNN to numerical columns and most-frequent to categorical columns; `drop` removes rows with any missing values.
- Standardization is applied to numerical parameters and stored in `self.scaler`.
- The second clustering writes the `cluster` label back to the original `data` and `preprocessed_data`.
- `figure_logrank_nelson` displays cumulative hazard curves and a Holm-corrected log-rank p-value table.