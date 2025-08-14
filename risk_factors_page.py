import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st

def risk_factors_page(df):
    """Detailed analysis of churn risk factors"""
    st.title('Churn Risk Factors Analysis')
    
    # Key risk factors visualization
    risk_factors = ['MonthlyCharges', 'tenure', 'TotalCharges', 'SeniorCitizen']
    
    for factor in risk_factors:
        st.subheader(f'{factor} vs Churn')
        fig = px.box(
            df, 
            x='Churn', 
            y=factor, 
            title=f'{factor} Distribution by Churn Status',
            color='Churn',
            color_discrete_map={'Yes': 'red', 'No': 'green'}
        )
        st.plotly_chart(fig)
    
    # Correlation heatmap
    # Create a copy of the dataframe for correlation
    df_corr = df.copy()
    
    # Convert Churn to numeric
    df_corr['Churn'] = (df_corr['Churn'] == 'Yes').astype(int)
    
    # Clean and convert columns
    columns_to_convert = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                           'SeniorCitizen', 'Partner', 'Dependents']
    
    for col in columns_to_convert:
        # Handle different column types
        if df_corr[col].dtype == 'object':
            # Convert boolean-like strings to numeric
            df_corr[col] = df_corr[col].replace({' ': np.nan, 'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
        
        # Convert to numeric, coercing errors to NaN
        df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
    
    # Select columns for correlation
    corr_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                    'SeniorCitizen', 'Partner', 'Dependents', 'Churn']
    
    # Calculate correlation
    try:
        correlation_matrix = df_corr[corr_columns].corr()
        
        # Create correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix, 
            text_auto=True, 
            aspect='auto', 
            title='Correlation Heatmap of Churn Factors',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr)
    except Exception as e:
        st.error(f"Error creating correlation matrix: {e}")
