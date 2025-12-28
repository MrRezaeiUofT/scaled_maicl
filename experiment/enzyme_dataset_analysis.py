"""
Enzyme Dataset Analysis Script
Load, save, and analyze enzyme activity datasets from:
https://github.com/samgoldman97/enzyme-datasets

Author: Analysis Script
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EnzymeDatasetLoader:
    """Class to load and manage enzyme datasets"""
    
    def __init__(self, data_dir="enzyme-datasets/data"):
        """
        Initialize the dataset loader
        
        Args:
            data_dir: Path to the enzyme-datasets/data directory
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def list_available_datasets(self):
        """List all available datasets in the directory"""
        if not self.data_dir.exists():
            print(f"Directory {self.data_dir} not found!")
            print("Please clone the repository first:")
            print("git clone https://github.com/samgoldman97/enzyme-datasets.git")
            return []
        
        dataset_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        print(f"Found {len(dataset_dirs)} datasets:")
        for d in dataset_dirs:
            print(f"  - {d.name}")
        return dataset_dirs
    
    def load_dataset(self, dataset_name, file_type='tsv'):
        """
        Load a specific dataset
        
        Args:
            dataset_name: Name of the dataset folder or processed dataset file (without extension)
            file_type: Type of file to load ('tsv', 'csv', 'pkl')
            
        Returns:
            DataFrame with the dataset
        """
        dataset_path = self.data_dir / dataset_name
        data_files = []
        
        # First, try the direct path (for raw datasets in subdirectories)
        if dataset_path.exists():
            # Try to find data files
            if dataset_path.is_file():
                # If it's a file, use it directly
                data_files = [dataset_path]
                file_type = dataset_path.suffix[1:] if dataset_path.suffix else 'csv'
            else:
                # If it's a directory, search for files
                data_files = list(dataset_path.glob(f"*.{file_type}"))
                
                if not data_files:
                    # Try alternative extensions
                    for ext in ['csv', 'tsv', 'txt', 'pkl']:
                        data_files = list(dataset_path.glob(f"*.{ext}"))
                        if data_files:
                            file_type = ext
                            break
        else:
            # Try looking in processed directory for CSV files
            processed_dir = self.data_dir / "processed"
            if processed_dir.exists():
                # Look for CSV files matching the dataset name
                data_files = list(processed_dir.glob(f"{dataset_name}.csv"))
                if not data_files:
                    print(f"Dataset {dataset_name} not found!")
                    print(f"Checked: {self.data_dir / dataset_name}")
                    print(f"Checked: {processed_dir / f'{dataset_name}.csv'}")
                    return None
                # Found file in processed directory, file_type is already 'csv'
            else:
                print(f"Dataset {dataset_name} not found!")
                return None
        
        if not data_files:
            print(f"No data files found in {dataset_path}")
            return None
        
        print(f"\nLoading {dataset_name}...")
        print(f"Found {len(data_files)} file(s)")
        
        # Load the first data file found
        data_file = data_files[0]
        print(f"Loading: {data_file.name}")
        
        try:
            if file_type in ['tsv', 'txt']:
                df = pd.read_csv(data_file, sep='\t')
            elif file_type == 'csv':
                df = pd.read_csv(data_file)
            elif file_type == 'pkl':
                df = pd.read_pickle(data_file)
            else:
                print(f"Unsupported file type: {file_type}")
                return None
            
            self.datasets[dataset_name] = df
            print(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def save_dataset(self, df, output_path, format='csv'):
        """
        Save dataset to file
        
        Args:
            df: DataFrame to save
            output_path: Path to save the file
            format: Format to save ('csv', 'tsv', 'pkl', 'json', 'excel')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'tsv':
                df.to_csv(output_path, sep='\t', index=False)
            elif format == 'pkl':
                df.to_pickle(output_path)
            elif format == 'json':
                df.to_json(output_path, orient='records', indent=2)
            elif format == 'excel':
                df.to_excel(output_path, index=False)
            else:
                print(f"Unsupported format: {format}")
                return False
            
            print(f"Dataset saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving dataset: {e}")
            return False


class EnzymeDatasetAnalyzer:
    """Class to analyze enzyme datasets"""
    
    def __init__(self, df):
        """
        Initialize analyzer with a dataset
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("=" * 80)
        print("DATASET BASIC INFORMATION")
        print("=" * 80)
        
        print(f"\nDataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        print("\nColumn Names and Types:")
        print(self.df.dtypes)
        
        print(f"\nNumeric Columns ({len(self.numeric_cols)}):")
        print(self.numeric_cols)
        
        print(f"\nCategorical Columns ({len(self.categorical_cols)}):")
        print(self.categorical_cols)
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        missing_pct = 100 * missing / len(self.df)
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        if len(missing_df[missing_df['Missing Count'] > 0]) == 0:
            print("No missing values found!")
        
        print("\nFirst few rows:")
        print(self.df.head())
        
    def statistical_summary(self):
        """Show statistical summary of numeric columns"""
        print("\n" + "=" * 80)
        print("STATISTICAL SUMMARY")
        print("=" * 80)
        
        if len(self.numeric_cols) > 0:
            print("\nNumeric Columns Statistics:")
            print(self.df[self.numeric_cols].describe())
        else:
            print("No numeric columns found!")
        
        if len(self.categorical_cols) > 0:
            print("\n\nCategorical Columns Value Counts:")
            for col in self.categorical_cols[:5]:  # Show first 5 categorical columns
                print(f"\n{col}:")
                print(self.df[col].value_counts().head(10))
    
    def correlation_analysis(self, save_path=None):
        """
        Analyze correlations between numeric features
        
        Args:
            save_path: Path to save the correlation heatmap
        """
        if len(self.numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation analysis")
            return None
        
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find highly correlated pairs
        print("\nHighly Correlated Feature Pairs (|r| > 0.7):")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr:
            high_corr_df = pd.DataFrame(high_corr)
            print(high_corr_df)
        else:
            print("No highly correlated pairs found")
        
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', square=True, ax=ax, cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nHeatmap saved to: {save_path}")
        
        plt.show()
        
        return corr_matrix
    
    def distribution_analysis(self, save_path=None):
        """
        Plot distributions of numeric features
        
        Args:
            save_path: Path to save the distribution plots
        """
        if len(self.numeric_cols) == 0:
            print("No numeric columns to plot")
            return
        
        print("\n" + "=" * 80)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        n_cols = min(3, len(self.numeric_cols))
        n_rows = (len(self.numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, col in enumerate(self.numeric_cols):
            if idx < len(axes):
                self.df[col].hist(bins=30, ax=axes[idx], edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                
                # Add mean and median lines
                mean_val = self.df[col].mean()
                median_val = self.df[col].median()
                axes[idx].axvline(mean_val, color='red', linestyle='--', 
                                 label=f'Mean: {mean_val:.2f}')
                axes[idx].axvline(median_val, color='green', linestyle='--',
                                 label=f'Median: {median_val:.2f}')
                axes[idx].legend()
        
        # Hide extra subplots
        for idx in range(len(self.numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nDistribution plots saved to: {save_path}")
        
        plt.show()
    
    def regression_analysis(self, target_col, feature_cols=None, test_size=0.2):
        """
        Perform simple regression analysis
        
        Args:
            target_col: Name of the target column for prediction
            feature_cols: List of feature columns (if None, use all numeric except target)
            test_size: Fraction of data to use for testing
        """
        if target_col not in self.df.columns:
            print(f"Target column '{target_col}' not found!")
            return None
        
        print("\n" + "=" * 80)
        print(f"REGRESSION ANALYSIS - Predicting {target_col}")
        print("=" * 80)
        
        # Prepare features and target
        if feature_cols is None:
            feature_cols = [col for col in self.numeric_cols if col != target_col]
        
        if len(feature_cols) == 0:
            print("No feature columns available for regression!")
            return None
        
        # Remove rows with missing values
        df_clean = self.df[[target_col] + feature_cols].dropna()
        print(f"\nUsing {len(df_clean)} samples (removed {len(self.df) - len(df_clean)} with missing values)")
        print(f"Features: {feature_cols}")
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        print("\nTraining Random Forest Regressor...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = rf_model.predict(X_train_scaled)
        y_test_pred = rf_model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print("\n" + "-" * 50)
        print("MODEL PERFORMANCE")
        print("-" * 50)
        print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
        print("-" * 50)
        print(f"{'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
        print(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")
        print(f"{'MAE':<20} {train_mae:<15.4f} {test_mae:<15.4f}")
        print("-" * 50)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n" + "-" * 50)
        print("FEATURE IMPORTANCE")
        print("-" * 50)
        print(feature_importance)
        
        # Plot predictions vs actual
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training set
        axes[0].scatter(y_train, y_train_pred, alpha=0.5, edgecolors='k')
        axes[0].plot([y_train.min(), y_train.max()], 
                     [y_train.min(), y_train.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel(f'Actual {target_col}', fontweight='bold')
        axes[0].set_ylabel(f'Predicted {target_col}', fontweight='bold')
        axes[0].set_title(f'Training Set (R² = {train_r2:.4f})', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Test set
        axes[1].scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k', color='orange')
        axes[1].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[1].set_xlabel(f'Actual {target_col}', fontweight='bold')
        axes[1].set_ylabel(f'Predicted {target_col}', fontweight='bold')
        axes[1].set_title(f'Test Set (R² = {test_r2:.4f})', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance['Importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
        plt.xlabel('Importance', fontweight='bold')
        plt.title('Feature Importance for Prediction', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return {
            'model': rf_model,
            'scaler': scaler,
            'metrics': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae
            },
            'feature_importance': feature_importance
        }


def main():
    """Main function to demonstrate usage"""
    print("=" * 80)
    print("ENZYME DATASET ANALYSIS TOOL")
    print("=" * 80)
    
    # Initialize loader
    loader = EnzymeDatasetLoader()
    
    # List available datasets
    print("\nSearching for datasets...")
    datasets = loader.list_available_datasets()
    
    if not datasets:
        print("\nCreating example dataset for demonstration...")
        # Create a synthetic enzyme dataset for demonstration
        np.random.seed(42)
        n_samples = 500
        
        example_df = pd.DataFrame({
            'enzyme_id': [f'ENZ_{i:04d}' for i in range(n_samples)],
            'substrate_concentration': np.random.uniform(0.1, 10, n_samples),
            'temperature': np.random.uniform(20, 40, n_samples),
            'pH': np.random.uniform(5, 9, n_samples),
            'enzyme_concentration': np.random.uniform(0.01, 1, n_samples),
            'reaction_time': np.random.uniform(10, 120, n_samples),
        })
        
        # Create synthetic activity based on features
        example_df['activity'] = (
            example_df['substrate_concentration'] * 0.5 +
            (example_df['temperature'] - 30) * 0.3 +
            (7.5 - abs(example_df['pH'] - 7.5)) * 0.4 +
            example_df['enzyme_concentration'] * 2 +
            np.random.normal(0, 0.5, n_samples)
        )
        
        example_df['enzyme_class'] = np.random.choice(
            ['Hydrolase', 'Oxidoreductase', 'Transferase'], 
            n_samples
        )
        
        # Save example dataset
        loader.save_dataset(example_df, '/home/claude/example_enzyme_data.csv', format='csv')
        print("\nExample dataset created and saved!")
        df = example_df
        
    else:
        # Try to load the first available dataset
        dataset_name = datasets[0].name
        df = loader.load_dataset(dataset_name)
        
        if df is None:
            print("Could not load dataset. Using example data instead.")
            return
    
    # Initialize analyzer
    analyzer = EnzymeDatasetAnalyzer(df)
    
    # Perform analyses
    print("\n")
    analyzer.basic_info()
    
    print("\n")
    analyzer.statistical_summary()
    
    print("\n")
    analyzer.distribution_analysis()
    
    if len(analyzer.numeric_cols) >= 2:
        print("\n")
        analyzer.correlation_analysis()
    
    # Perform regression if we have a suitable target column
    if 'activity' in df.columns:
        analyzer.regression_analysis('activity')
    elif len(analyzer.numeric_cols) > 1:
        # Use the last numeric column as target
        target = analyzer.numeric_cols[-1]
        analyzer.regression_analysis(target)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()