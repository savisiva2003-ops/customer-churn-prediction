# retrain_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
import joblib
import warnings

warnings.filterwarnings("ignore")

def retrain_model():
    # Load & preprocess
    df = pd.read_csv('data/raw/customer_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    # Feature engineering
    df['AvgChargePerMonth'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    df['ChargesPerTenure']   = df['MonthlyCharges'] * df['tenure']
    df['ChargeTenureRatio']  = df['ChargesPerTenure'] / (df['AvgChargePerMonth'] + 1e-6)

    # Binary encode
    df['Churn']            = (df['Churn'] == 'Yes').astype(int)
    df['PaperlessBilling'] = (df['PaperlessBilling'] == 'Yes').astype(int)
    df['gender']           = (df['gender'] == 'Male').astype(int)
    df['Partner']          = (df['Partner'] == 'Yes').astype(int)
    df['Dependents']       = (df['Dependents'] == 'Yes').astype(int)
    df['PhoneService']     = (df['PhoneService'] == 'Yes').astype(int)

    # Service columns mapping
    services = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies']
    for svc in services:
        df[svc] = df[svc].map({'No': 0, 'Yes': 1, 'No internet service': 0})
    df['MultipleLines'] = df['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 0})

    # One-hot encode remaining categoricals
    df = pd.get_dummies(df, columns=['InternetService','Contract','PaymentMethod'])

    # Features & target
    X = df.drop(['customerID','Churn'], axis=1)
    y = df['Churn']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Balance classes with ADASYN
    ada = ADASYN(random_state=42)
    X_train, y_train = ada.fit_resample(X_train, y_train)

    # Scale numeric features
    num_cols = ['tenure','MonthlyCharges','TotalCharges',
                'AvgChargePerMonth','ChargesPerTenure','ChargeTenureRatio']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # Define base learners with tuned parameters
    lgb = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=50,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1
    )

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=3,
        scale_pos_weight=1,
        random_state=42
    )

    # Build and train stacking ensemble
    stack = StackingClassifier(
        estimators=[('lgb', lgb), ('xgb', xgb)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1,
        passthrough=True
    )
    stack.fit(X_train, y_train)

    # Threshold sweep for optimal accuracy
    proba = stack.predict_proba(X_test)[:, 1]
    best_acc, best_thr = 0, 0.5
    for thr in np.linspace(0.3, 0.8, 51):
        pred = (proba >= thr).astype(int)
        acc = accuracy_score(y_test, pred)
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    # Final predictions at optimal threshold
    y_pred = (proba >= best_thr).astype(int)

    # Evaluate
    print(f"\n Optimal threshold: {best_thr:.2f}")
    print(f" Model Accuracy: {best_acc:.4f}")
    print(classification_report(y_test, y_pred))
    print(f" ROC-AUC Score: {roc_auc_score(y_test, proba):.4f}")

    # Generate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

   # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Feature importances (average of both models)
    # Ensure base learners are fitted
    lgb.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    fi_lgb = pd.DataFrame({'feature': X.columns, 'imp_lgb': lgb.feature_importances_})
    fi_xgb = pd.DataFrame({'feature': X.columns, 'imp_xgb': xgb.feature_importances_})
    fi = fi_lgb.merge(fi_xgb, on='feature')
    fi['importance'] = fi[['imp_lgb','imp_xgb']].mean(axis=1)
    fi = fi.sort_values('importance', ascending=False)

    print("\n Top 10 Important Features:")
    print(fi[['feature','importance']].head(10).to_string(index=False))

    # Sample predictions
    print("\n Sample Predictions:")
    for i, p in enumerate(proba[:5]):
        print(f"Example {i+1}: {p:.3f}")

    # Save model and scaler
    joblib.dump(stack,  'models/churn_model_ensemble.joblib')
    joblib.dump(scaler,'models/scaler_ensemble.joblib')


if __name__ == "__main__":
    retrain_model()
