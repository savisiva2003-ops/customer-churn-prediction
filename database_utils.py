import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objs as go
import pymongo
from pymongo.errors import ConnectionFailure, PyMongoError

# MongoDB Connection Configuration
def get_mongodb_connection():
    """
    Establish a MongoDB connection using PyMongo.
    
    Expected environment variables or Streamlit secrets:
    - MONGODB_URI: Full connection string
    - MONGODB_DATABASE: Database name
    - MONGODB_COLLECTION: Collection name for customer churn data
    """
    try:
        # Attempt to get connection details from Streamlit secrets
        # If not using Streamlit secrets, replace with your own connection method
        mongodb_uri = st.secrets.get("MONGODB_URI", 
            "mongodb://localhost:27017")
        database_name = st.secrets.get("MONGODB_DATABASE", "customer_analytics")
        collection_name = st.secrets.get("MONGODB_COLLECTION", "churn_data")
        
        # Create MongoDB client
        client = pymongo.MongoClient(mongodb_uri)
        
        # Test the connection
        client.admin.command('ping')
        
        # Return database and collection
        db = client[database_name]
        collection = db[collection_name]
        
        return collection
    
    except ConnectionFailure:
        st.error("Failed to connect to MongoDB. Check your connection details.")
        return None
    except Exception as e:
        st.error(f"An error occurred while connecting to MongoDB: {e}")
        return None

def fetch_churn_data():
    """
    Fetch customer churn data from MongoDB.
    Converts MongoDB documents to a pandas DataFrame.
    """
    collection = get_mongodb_connection()
    if not collection:
        return None
    
    try:
        # Fetch all documents from the collection
        cursor = collection.find({})
        
        # Convert to DataFrame
        df = pd.DataFrame(list(cursor))
        
        # Remove MongoDB's internal _id field if present
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        # Ensure all necessary columns are present
        required_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
            'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        
        # Validate DataFrame columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            st.warning(f"Missing columns in data: {missing_columns}")
        
        # Select only required columns
        df = df[[col for col in required_columns if col in df.columns]]
        
        return df
    
    except PyMongoError as e:
        st.error(f"Error fetching data from MongoDB: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error processing MongoDB data: {e}")
        return None

def preprocess_input(data_dict):
    """
    Preprocess input data for churn prediction.
    (This function remains the same as in the previous implementation)
    """
    # [Previous preprocess_input implementation]
    # Convert to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Create AvgChargePerMonth
    df['AvgChargePerMonth'] = df['TotalCharges'] / np.where(df['tenure'] == 0, 1, df['tenure'])
    
    # Binary encoding
    df['PaperlessBilling'] = (df['PaperlessBilling'] == 'Yes').astype(int)
    df['gender'] = (df['gender'] == 'Male').astype(int)
    df['Partner'] = (df['Partner'] == 'Yes').astype(int)
    df['Dependents'] = (df['Dependents'] == 'Yes').astype(int)
    df['PhoneService'] = (df['PhoneService'] == 'Yes').astype(int)
    
    # Existing preprocessing logic continues...
    # (Rest of the preprocessing function remains the same)
    
    return df

# Keep all other existing functions from the previous implementation:
# - load_models()
# - predict_churn()
# - prediction_page()
# - data_insights_page()
# - risk_factors_page()

def main():
    st.sidebar.title('Churn Prediction Dashboard')
    
    # Fetch dataset from MongoDB
    df = fetch_churn_data()
    
    if df is None:
        st.error("Failed to load dataset. Check your MongoDB connection.")
        return
    
if __name__ == '__main__':
    main()