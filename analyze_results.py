# analyze_results.py
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelAnalyzer:
    def __init__(self, model_path='models/churn_model.joblib'):
        self.model = joblib.load(model_path)
        self.data_loader = DataLoader('data/raw/customer_churn.csv')
        self.feature_engineer = FeatureEngineer()

    def prepare_data(self):
        """Prepare data for analysis"""
        df = self.data_loader.load_data()
        df = self.data_loader.preprocess_data(df)
        df = self.feature_engineer.create_features(df)
        
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        return X, y

    def plot_feature_importance(self, X):
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=importances.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        plt.savefig('analysis/feature_importance.png')
        plt.close()
        
        return importances

    def plot_roc_curve(self, X, y):
        """Plot ROC curve"""
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('analysis/roc_curve.png')
        plt.close()

    def analyze_predictions(self, X, y):
        """Analyze prediction probabilities"""
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = self.model.predict(X)
        
        # Create prediction analysis dataframe
        analysis_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred,
            'Probability': y_pred_proba
        })
        
        # Analyze prediction distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=analysis_df, x='Probability', hue='Actual', bins=50)
        plt.title('Distribution of Prediction Probabilities by Actual Class')
        plt.savefig('analysis/prediction_distribution.png')
        plt.close()
        
        return analysis_df

    def run_analysis(self):
        """Run all analyses"""
        logger.info("Starting model analysis...")
        
        # Create analysis directory if it doesn't exist
        import os
        if not os.path.exists('analysis'):
            os.makedirs('analysis')
        
        # Prepare data
        X, y = self.prepare_data()
        
        # Run analyses
        importances = self.plot_feature_importance(X)
        self.plot_roc_curve(X, y)
        analysis_df = self.analyze_predictions(X, y)
        
        # Print summary statistics
        logger.info("\nTop 5 Most Important Features:")
        logger.info(importances.head().to_string())
        
        logger.info("\nPrediction Distribution Summary:")
        logger.info(analysis_df.groupby('Actual')['Probability'].describe().to_string())
        
        logger.info("\nAnalysis completed! Check the 'analysis' directory for visualizations.")

if __name__ == "__main__":
    analyzer = ModelAnalyzer()
    analyzer.run_analysis()