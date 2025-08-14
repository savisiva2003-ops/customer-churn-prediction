# main.py
import pandas as pd
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.utils.logger import get_logger
import os

logger = get_logger(__name__)

def main():
    try:
        # Initialize components
        data_loader = DataLoader('data/raw/customer_churn.csv')
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer(model_type='xgboost')
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = data_loader.load_data()
        df = data_loader.preprocess_data(df)
        
        # Feature engineering
        logger.info("Performing feature engineering...")
        df = feature_engineer.create_features(df)
        
        # Prepare features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # Train and evaluate model
        logger.info("Training model...")
        X_test, y_test, y_pred = model_trainer.train_model(X, y)
        
        # Save model
        if not os.path.exists('models'):
            os.makedirs('models')
        model_trainer.save_model('models/churn_model.joblib')
        
        logger.info("Project execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

    # first train model