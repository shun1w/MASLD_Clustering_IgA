# import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import mixture
from sklearn.impute import KNNImputer,SimpleImputer
import matplotlib.pyplot as plt
from lifelines import NelsonAalenFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from IPython.display import display
import logging
import warnings
warnings.simplefilter('ignore', FutureWarning)

# data preprocessing class
class Newmodelapply:
    def __init__(self, data, parameters, categorical_params, log_level=logging.INFO):
        self.data = data
        self.preprocessed_data = None
        self.parameters = parameters
        self.scaler = None
        self.last_model = None
        self.categorical_params = categorical_params
        
        # Configure logger
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.logger.setLevel(log_level)
        
        # Add handler only if it does not already exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"Newmodelapply initialized with {len(data)} samples and {len(parameters)} parameters")
    
    def data_preprocessing(self, missing_value='imputation',outliner_sd=5):
        """
        Run preprocessing on the dataset.

        Usage:
            data_preprocessing(missing_value='imputation', outliner_sd=5)

        Args:
            missing_value: choose 'imputation' (default) or 'drop' for handling missing values.
            outliner_sd: standard deviation threshold for outlier removal (default is 5).

        Returns:
            Preprocessed DataFrame stored in self.preprocessed_data.
        """
        self.logger.info(f"Starting data preprocessing with missing_value='{missing_value}' and outlier_sd={outliner_sd}")
        
        try:
            # Handle outliers
            # Collect numerical parameters
            numerical_params = [col for col in self.parameters if col not in self.categorical_params]
            self.logger.info(f"Identified {len(numerical_params)} numerical parameters: {numerical_params}")
            
            # Remove outliers (>= outliner_sd) for each numerical parameter
            df = self.data.copy()
            df = df[self.parameters]
            # Remove outliers only for numerical_params
            outliners = []
            for i in numerical_params:
                kagen=df[i].mean()-outliner_sd*df[i].std()
                jougen = df[i].mean()+outliner_sd*df[i].std()
                outliner = df[(df[i]<kagen)|(df[i]>jougen)]
                if len(outliner)==0:
                    continue
                else:
                    outliners+=list(outliner.index)
            
            self.logger.info(f'Outliers detected: {len(outliners)} cases will be removed')
            print(f'outliners: {len(outliners)} cases are removed')
            # Drop outliers from analysis data
            df = df.drop(outliners)
            # Drop outliers from original data
            self.data = self.data.drop(outliners)
            # Store cleaned data in preprocessed_data for downstream use
            self.preprocessed_data = df

            # Handle missing values -> imputation or drop
            self.logger.info(f"Handling missing values using method: {missing_value}")
            
            # Imputation branch
            if missing_value=='imputation':
                # Detect columns with >50% missing values and log a warning
                missing_ratio = df.isnull().mean()
                high_missing_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
                if high_missing_cols:
                    self.logger.warning(f"Warning: {len(high_missing_cols)} columns have more than 50% missing values: {high_missing_cols}")
                # Apply KNN imputation to numerical_params
                self.logger.info("Applying KNN imputation to numerical parameters")
                knn_imputer = KNNImputer(n_neighbors=5)
                df[numerical_params] = knn_imputer.fit_transform(df[numerical_params])
                # Apply most_frequent imputation to categorical_params
                self.logger.info("Applying most_frequent imputation to categorical parameters")
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                df[self.categorical_params] = categorical_imputer.fit_transform(df[self.categorical_params])

            # Drop branch
            elif missing_value=='drop':
                self.logger.info("Dropping rows with missing values")
                before_drop = len(df)
                df = df.dropna()
                dropped_count = before_drop - len(df)
                self.logger.info(f"{dropped_count} cases are removed because of missing values")

            # Standardization
            # Scale numerical_params
            self.logger.info("Applying standardization to numerical parameters")
            scaler = StandardScaler()
            df[numerical_params] = scaler.fit_transform(df[numerical_params])
            self.preprocessed_data = df
            self.scaler = scaler
            
            self.logger.info(f"Data preprocessing completed. Final dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def first_clustering(self, random_state=42):
        """
        Run the initial clustering.
        Number of clusters is fixed at n=10.
        """
        self.logger.info("Starting first clustering with 10 components")
        
        try:
            # Select clustering model
            model = mixture.GaussianMixture(n_components=10, random_state=random_state)
            self.logger.info("Fitting Gaussian Mixture model...")
            model.fit(self.preprocessed_data)

            # Obtain clustering results
            cluster_labels = model.predict(self.preprocessed_data)
            self.logger.info(f"First clustering completed. Cluster distribution: {np.bincount(cluster_labels)}")

            # Visualize clustering results
            self.logger.info("Generating cluster weights visualization")
            fig,axes = plt.subplots(1,1,figsize=(10,3))
            xticks=np.arange(0,10,1)
            axes.bar(xticks, model.weights_, width = 0.7, tick_label=xticks)
            plt.title("First Clustering - Cluster Weights")
            plt.show()
            
            self.logger.info("First clustering visualization completed")
            
        except Exception as e:
            self.logger.error(f"Error in first clustering: {str(e)}")
            raise
    
    def second_clustering(self, n_clusters,n_param_show=10, random_state=42):
        """
        Run the second clustering.
        n_clusters: number of clusters.
        n_param_show: number of features to visualize.
        random_state: random seed.
        """
        self.logger.info(f"Starting second clustering with {n_clusters} components")
        
        try:
            # Select clustering model
            model = mixture.GaussianMixture(n_components=n_clusters, random_state=random_state)
            self.logger.info("Fitting Gaussian Mixture model for second clustering...")
            model.fit(self.preprocessed_data)

            self.last_model = model

            # Obtain clustering results
            cluster_labels = model.predict(self.preprocessed_data)
            self.preprocessed_data['cluster'] = cluster_labels
            self.logger.info(f"Second clustering completed. Cluster distribution: {np.bincount(cluster_labels)}")

            # Visualize features
            self.logger.info(f"Generating feature visualizations for {n_clusters} clusters")
            for name,grouped_df in self.preprocessed_data.groupby('cluster'):
                grouped_df = grouped_df.drop('cluster',axis=1)
                results = abs(grouped_df.median()).sort_values(ascending = False).head(n_param_show)
                plt.figure()
                plt.barh(results.index, results)
                plt.title(f'Cluster{name} - Top {n_param_show} Features')
                plt.xlabel('Median Absolute Value')
                plt.show()
                self.logger.info(f"Visualization completed for Cluster {name}")

            # Add cluster labels back to the original data
            self.data['cluster'] = cluster_labels
            self.logger.info("Second clustering completed successfully")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error in second clustering: {str(e)}")
            raise
    
    def figure_logrank_nelson(self, time, event, event_name, group = 'cluster',ylim = 0.5,xlim = 60):
        self.logger.info(f"Starting survival analysis for {event_name}")
        
        try:
            fig, axes = plt.subplots(1,1,figsize=(10,10))
            axes.set_ylim(0,ylim)
            axes.set_xlim(0,xlim)
            fitters = []
            
            self.logger.info(f"Fitting Nelson-Aalen estimators for {len(self.data[group].unique())} groups")
            for name, grouped_df in self.data.groupby(group):
                fitter = NelsonAalenFitter()
                fitter.fit(grouped_df[time], event_observed = grouped_df[event],label=f'Cluster{name}')
                axes = fitter.plot_cumulative_hazard()
                fitters.append(fitter)
                self.logger.info(f"Fitted Nelson-Aalen for Cluster {name}: {len(grouped_df)} samples")
            
            plt.title(event_name)
            add_at_risk_counts(*fitters,ax = axes)
            fig.tight_layout()
            plt.show()
            self.logger.info("Survival curve visualization completed")

            # Logrank test with Holm
            self.logger.info("Performing Log-rank tests between all cluster pairs")
            log_rank = {}
            unique_groups = self.data[group].unique()
            self.logger.info(f"Testing {len(list(combinations(unique_groups, 2)))} pairwise comparisons")
            
            for pairs in combinations(unique_groups,2):
                ix = self.data[group] == pairs[0]
                ex = self.data[group] == pairs[1]
                duration_A =self.data.loc[ix, time]
                duration_B = self.data.loc[ex, time]
                event_A= self.data.loc[ix, event]
                event_B =self.data.loc[ex, event]
                results = logrank_test(durations_A=duration_A, durations_B=duration_B, event_observed_A=event_A, event_observed_B=event_B)
                log_rank[f'{pairs}'] = [round(results.p_value,6)]
                self.logger.info(f"Log-rank test for {pairs}: p-value = {results.p_value:.6f}")
            
            log_rank=pd.DataFrame(log_rank).T.rename(columns={0:'P-value(Log-rank with Holm)'})
            # Use statsmodels multipletest
            log_rank['P-value(Log-rank with Holm)'] = multipletests(log_rank['P-value(Log-rank with Holm)'].to_list(),method='holm')[1]
            self.logger.info("Multiple testing correction (Holm) applied")
            display(log_rank)
            self.logger.info("Survival analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in survival analysis: {str(e)}")
            raise