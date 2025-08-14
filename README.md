# Customer Churn Prediction System

## Project Overview
The Customer Churn Prediction System is designed to help businesses predict customer churn and take proactive measures to improve customer retention. The system identifies the likelihood of customers churning based on their details and offers tailored retention strategies. It consists of three main pages:

1. **Customer Form Page**: Allows users to input customer details and displays:
   - Churn percentage (High, Medium, or Low)
   - Identified problems
   - Recommended retention strategies

2. **Customer List Page**: Displays a list of customers with their churn status.

3. **Dashboard Page**: Provides an overview of key metrics, trends, and insights.

4. **Interaction Page**: Displays a list of employees with the status.

## Features
- Predict customer churn with actionable insights.
- Display customer data and churn status in an organized manner.
- Interactive dashboard for real-time analytics.

## Steps to Initialize the Project

1. **Create Project Directory**
   ```bash
   mkdir churn-management-system
   cd churn-management-system
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install Necessary Packages**
   Create a `requirements.txt` file with the following dependencies:
   ```plaintext
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   xgboost
   joblib
   ```
   Then install them:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create Directory Structure**
   Use the project structure above to create folders and empty files:
   ```bash
   mkdir -p data/raw data/processed notebooks src/data src/features src/models src/utils tests
   touch src/data/data_loader.py src/data/data_cleaning.py
   touch src/features/feature_engineering.py
   touch src/models/train.py src/models/predict.py src/models/model_utils.py
   touch src/utils/logger.py src/utils/config.py
   touch README.md requirements.txt setup.py .gitignore
   ```

5. **Set Up Version Control**
   ```bash
   git init
   echo "venv/" > .gitignore
   ```

6. **Write Initial Scripts**
   - **`src/utils/logger.py`:**
     ```python
     import logging

     def get_logger(name):
         logging.basicConfig(level=logging.INFO)
         return logging.getLogger(name)
     ```

   - **`src/utils/config.py`:**
     ```python
     # Configuration file

     DATA_PATH = "data/"
     MODEL_PATH = "models/"
     ```

## Dependencies
- Python 3.8 or higher
- Pandas
- Scikit-learn
- Matplotlib

## Database
- MongoDB 8.0

## Usage
1. Navigate to the Customer Form Page and input customer details.
2. Review the churn prediction and suggested strategies.
3. Use the Customer List Page to manage and monitor customer data.
4. Analyze overall trends and metrics on the Dashboard Page.


