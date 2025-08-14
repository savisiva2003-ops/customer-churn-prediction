import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objs as go

# Preprocessing Utility Functions
def preprocess_input(data_dict):
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
    
    # Handle services
    service_map = {'No': 0, 'Yes': 1, 'No internet service': 0}
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        df[col] = df[col].map(service_map)
    
    df['MultipleLines'] = df['MultipleLines'].map({
        'No': 0, 'Yes': 1, 'No phone service': 0
    })
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])
    
    # Ensure all columns from training are present
    expected_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                       'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'AvgChargePerMonth',
                       'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
                       'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
                       'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
                       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[expected_columns]

def load_models():
    model = joblib.load('models/churn_model_v2.joblib')
    scaler = joblib.load('models/scaler_v2.joblib')
    return model, scaler

def predict_churn(input_data):
    # Load models
    model, scaler = load_models()
    
    # Preprocess input
    processed_input = preprocess_input(input_data)
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargePerMonth']
    processed_input[numerical_cols] = scaler.transform(processed_input[numerical_cols])
    
    # Make prediction
    churn_probability = float(model.predict_proba(processed_input)[0][1])
    return churn_probability

def prediction_page():
    st.title('Customer Churn Prediction')
    st.write('Enter customer details to predict churn probability')
    
    with st.form("customer_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Personal Information')
            gender = st.selectbox('Gender', ['Male', 'Female'])
            senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
            partner = st.selectbox('Partner', ['No', 'Yes'])
            dependents = st.selectbox('Dependents', ['No', 'Yes'])
            
            st.subheader('Basic Services')
            tenure = st.number_input('Tenure (months)', min_value=0, max_value=120, value=1)
            phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
            multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
            internet_service = st.selectbox('Internet Service', ['Fiber optic', 'DSL', 'No'])
        
        with col2:
            st.subheader('Additional Services')
            if internet_service != 'No':
                online_security = st.selectbox('Online Security', ['No', 'Yes'])
                online_backup = st.selectbox('Online Backup', ['No', 'Yes'])
                device_protection = st.selectbox('Device Protection', ['No', 'Yes'])
                tech_support = st.selectbox('Tech Support', ['No', 'Yes'])
                streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes'])
                streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes'])
            else:
                online_security = 'No internet service'
                online_backup = 'No internet service'
                device_protection = 'No internet service'
                tech_support = 'No internet service'
                streaming_tv = 'No internet service'
                streaming_movies = 'No internet service'
            
            st.subheader('Contract Information')
            contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
            paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
            payment_method = st.selectbox('Payment Method', 
                                        ['Electronic check', 'Mailed check', 
                                         'Bank transfer (automatic)', 'Credit card (automatic)'])
            monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=90.0)
            total_charges = st.number_input('Total Charges', min_value=0.0, 
                                          value=monthly_charges * tenure if tenure > 0 else 0.0)
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            input_data = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
            }
            
            try:
                # Make prediction
                churn_probability = predict_churn(input_data)
                
                # Display results with improved visuals
                st.subheader('Prediction Results')
                
                # Create a progress bar with color coding
                color = 'red' if churn_probability > 0.6 else 'orange' if churn_probability > 0.3 else 'green'
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                        <h3 style="color: {color}; margin-bottom: 10px;">
                            Churn Probability: {churn_probability*100:.1f}%
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Risk category
                if churn_probability > 0.6:
                    risk_category = 'High Risk'
                    color = 'red'
                elif churn_probability > 0.3:
                    risk_category = 'Medium Risk'
                    color = 'orange'
                else:
                    risk_category = 'Low Risk'
                    color = 'green'
                
                st.markdown(f"<h4 style='color: {color};'>Risk Category: {risk_category}</h4>", 
                          unsafe_allow_html=True)
                
                # Key risk factors
                st.subheader('Risk Factors Analysis')
                risk_factors = []
                
                if contract == 'Month-to-month':
                    risk_factors.append("Month-to-month contract")
                if internet_service == 'Fiber optic':
                    risk_factors.append("Fiber optic service")
                if tenure < 12:
                    risk_factors.append("Low tenure (< 12 months)")
                if monthly_charges > 70:
                    risk_factors.append("High monthly charges")
                if payment_method == 'Electronic check':
                    risk_factors.append("Payment by electronic check")
                if tech_support == 'No':
                    risk_factors.append("No technical support")
                if online_security == 'No':
                    risk_factors.append("No online security")
                
                for factor in risk_factors:
                    st.markdown(f"• {factor}")
                
                # Recommendations
                st.subheader('Recommended Actions')
                if churn_probability > 0.6:
                    st.markdown("""
                    🚨 **Immediate Actions Required:**
                    * Schedule urgent customer outreach
                    * Offer contract upgrade with significant incentives
                    * Provide free trial of security services
                    * Consider personalized discount package
                    * Schedule service quality review
                    """)
                elif churn_probability > 0.3:
                    st.markdown("""
                    ⚠️ **Preventive Measures:**
                    * Initiate customer satisfaction survey
                    * Offer service upgrades at promotional rates
                    * Recommend suitable protection services
                    * Schedule regular check-ins
                    """)
                else:
                    st.markdown("""
                    ✅ **Retention Strategies:**
                    * Maintain regular communication
                    * Consider upselling premium services
                    * Enroll in loyalty rewards program
                    * Monitor service usage patterns
                    """)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def data_insights_page(df):
    """Provide insights and visualizations about customer churn"""
    st.title('Churn Data Insights')
    
    # Overall Churn Rate
    churn_rate = df['Churn'].value_counts(normalize=True)
    st.subheader('Overall Churn Overview')
    fig_churn_rate = px.pie(
        values=churn_rate.values, 
        names=churn_rate.index, 
        title='Customer Churn Distribution',
        color_discrete_map={'Yes': 'red', 'No': 'green'}
    )
    st.plotly_chart(fig_churn_rate)
    
    # Churn by Contract Type
    churn_by_contract = df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
    churn_by_contract_pct = churn_by_contract.div(churn_by_contract.sum(axis=1), axis=0) * 100
    
    fig_contract_churn = px.bar(
        x=churn_by_contract_pct.index, 
        y=churn_by_contract_pct['Yes'], 
        title='Churn Rate by Contract Type',
        labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'}
    )
    st.plotly_chart(fig_contract_churn)
    
    # Tenure vs Churn
    st.subheader('Tenure Impact on Churn')
    fig_tenure_churn = go.Figure()
    fig_tenure_churn.add_trace(go.Box(
        x=df[df['Churn'] == 'Yes']['tenure'], 
        name='Churned Customers', 
        marker_color='red'
    ))
    fig_tenure_churn.add_trace(go.Box(
        x=df[df['Churn'] == 'No']['tenure'], 
        name='Retained Customers', 
        marker_color='green'
    ))
    fig_tenure_churn.update_layout(
        title='Customer Tenure Distribution by Churn Status',
        xaxis_title='Tenure (Months)'
    )
    st.plotly_chart(fig_tenure_churn)
    
    # Services Impact
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    service_churn_impact = {}
    
    for service in services:
        service_churn = df.groupby([service, 'Churn']).size().unstack(fill_value=0)
        service_churn_pct = service_churn.div(service_churn.sum(axis=1), axis=0) * 100
        service_churn_impact[service] = service_churn_pct['Yes']
    
    fig_service_churn = go.Figure(data=[
        go.Bar(
            name=service, 
            x=[f"{service}: {status}" for status in service_churn_impact[service].index], 
            y=service_churn_impact[service].values
        ) for service in services
    ])
    fig_service_churn.update_layout(
        title='Churn Rate by Additional Services',
        yaxis_title='Churn Rate (%)',
        barmode='group'
    )
    st.plotly_chart(fig_service_churn)
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

def main():
    st.sidebar.title('Churn Prediction Dashboard')
    
    # Load dataset
    try:
       df = pd.read_csv('data/raw/customer_churn.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'customer_churn_data.csv' is in the data folder.")
        return
    
    # Page navigation
    page = st.sidebar.radio(
        'Navigate', 
        ['Prediction', 'Data Insights', 'Risk Factors Analysis']
    )
    
    if page == 'Prediction':
        prediction_page()
    elif page == 'Data Insights':
        data_insights_page(df)
    elif page == 'Risk Factors Analysis':
        risk_factors_page(df)

if __name__ == '__main__':
    main()