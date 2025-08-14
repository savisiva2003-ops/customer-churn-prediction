import streamlit as st

# IMPORTANT: set_page_config must be the very first Streamlit command
st.set_page_config(
    page_title="Telebond - Churn Prediction",
    page_icon="📊",
    layout="wide"
)

from streamlit_modal import Modal
import extra_streamlit_components as stx
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objs as go
import pymongo
from datetime import datetime, timedelta 
import hashlib
import time


# Load model and scaler
def load_models():
    try:
        model = joblib.load('models/churn_model_v2.joblib')
        scaler = joblib.load('models/scaler_v2.joblib')
        
        # Simple validation
        if model is None or scaler is None:
            raise ValueError("Model or scaler failed to load properly")
            
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, scaler = load_models()

# MongoDB connection setup
uri = "mongodb+srv://savisiva:savilachine123@cluster0.eov2a.mongodb.net/churn?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(uri)
db = client["churn"]
collection = db["customer_data"]
users_collection = db["users"]
interaction_collection = db["interaction_history"]

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'
if 'is_authenticated' not in st.session_state:
    st.session_state['is_authenticated'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = 'guest'

# Authentication functions
def verify_login(email, password):
    """Verify login credentials and set session"""
    user = users_collection.find_one({"email": email})
    if user and user["password"] == hashlib.sha256(password.encode()).hexdigest():
        # Generate auth token with longer expiry
        auth_token = hashlib.sha256(f"{email}:{datetime.now().isoformat()}:{email}".encode()).hexdigest()
        
        # Update token in database
        users_collection.update_one(
            {"email": email},
            {
                "$set": {
                    "last_login": datetime.utcnow(),
                    "auth_token": auth_token,
                    "token_expiry": datetime.utcnow() + timedelta(days=1)  # Token expires in 1 day
                }
            }
        )
        
        # Set session state
        st.session_state['is_authenticated'] = True
        st.session_state['username'] = email
        st.session_state['role'] = user.get("role", "guest")
        st.session_state['user_email'] = email
        st.session_state['auth_token'] = auth_token
        
        # Set page to dashboard
        st.session_state['page'] = "Dashboard"
        
        # Update query parameters to reflect the page change
        st.query_params['page'] = 'dashboard'
        
        return True
    return False

def handle_logout():
    """Handle logout and clear session securely"""
    try:
        auth_token = st.session_state.get('auth_token')
        if auth_token:
            # Clear token in database
            users_collection.update_one(
                {"auth_token": auth_token},
                {"$set": {"auth_token": None}}
            )
    except Exception as e:
        st.error(f"Error during logout: {str(e)}")

    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Initialize with defaults
    st.session_state['page'] = 'login'
    st.session_state['is_authenticated'] = False
    
    # Clear URL parameters
    st.query_params.clear()

def get_prediction_history(role):
    if role == "fibre":
        return list(collection.find({"InternetService": "Fiber optic"}).sort("timestamp", -1).limit(100))
    elif role == "dsl":
        return list(collection.find({"InternetService": "DSL"}).sort("timestamp", -1).limit(100))
    else:
        return list(collection.find().sort("timestamp", -1).limit(100))

def validate_email(email):
    """Basic email validation"""
    return '@' in email and '.' in email

# Login page
def login_page():
    try:
        # Try to load logo
        from PIL import Image
        import base64
        try:
            logo_path = "logo.png"
            with open(logo_path, "rb") as f:
                data = f.read()
            encoded = base64.b64encode(data).decode()

            st.markdown(
               f"""
               <div style='text-align: center; margin-bottom: 30px;'>
                <img src="data:image/png;base64,{encoded}" width="200"/>
                <p style='color: #6B7280; font-size: 1.2em; margin-top: 10px;'>Customer Churn Prediction Dashboard</p>
                </div>
                """,
            unsafe_allow_html=True
            )
        except:
            st.title("Customer Churn Prediction Dashboard")
    except:
        st.title("Customer Churn Prediction Dashboard")

    with st.form("login_form"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
           login_submitted = st.form_submit_button("Login", use_container_width=True)

        if login_submitted:
            if verify_login(email, password):
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid email or password")

# Data preprocessing functions
def preprocess_input(data_dict):
    """Preprocess input data for model prediction"""
    model_data_dict = {k: v for k, v in data_dict.items() if k != 'customer_name'}
    df = pd.DataFrame([model_data_dict])
    
    df['AvgChargePerMonth'] = df['TotalCharges'] / np.where(df['tenure'] == 0, 1, df['tenure'])
    
    binary_cols = ['PaperlessBilling', 'gender', 'Partner', 'Dependents', 'PhoneService']
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)
    
    service_map = {'No': 0, 'Yes': 1, 'No internet service': 0}
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        df[col] = df[col].map(service_map)
    
    df['MultipleLines'] = df['MultipleLines'].map({
        'No': 0, 'Yes': 1, 'No phone service': 0
    })
    
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])
    
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

def predict_churn(input_data):
    """Make churn prediction"""
    if model is None or scaler is None:
        st.error("Models not loaded properly. Please check the model files.")
        return 0.5  # Default to 0.5 if models aren't loaded
        
    processed_input = preprocess_input(input_data)
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargePerMonth']
    processed_input[numerical_cols] = scaler.transform(processed_input[numerical_cols])
    
    return float(model.predict_proba(processed_input)[0][1])

def save_prediction_to_mongodb(input_data, churn_probability):
    """Save prediction data to MongoDB"""
    input_data['churn_probability'] = churn_probability
    input_data['timestamp'] = datetime.utcnow()
    input_data['risk_category'] = (
        'High Risk' if churn_probability > 0.6 
        else 'Medium Risk' if churn_probability > 0.3 
        else 'Low Risk'
    )

    # Remove _id if present to avoid duplicate key error
    input_data.pop('_id', None)

    try:
        collection.insert_one(input_data)
        st.success("Prediction data saved successfully!")
    except Exception as e:
        st.error(f"Failed to save prediction: {e}")

# Dashboard pages
def prediction_page():
    """Prediction page UI"""
    edit_mode = st.session_state.get("edit_mode", False)
    edit_data = st.session_state.get("edit_customer_data", {})

    st.title('Customer Churn Prediction')
    
    with st.form("customer_data_form"):
        customer_name = st.text_input('Customer Name', '')
        
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
            role = st.session_state.get("role", "guest")
            if role == "fibre":
                st.caption("Internet Service is auto-selected based on your role.")
                internet_service = st.selectbox('Internet Service', ['Fiber optic'], index=0, disabled=True)
            elif role == "dsl":
                st.caption("Internet Service is auto-selected based on your role.")
                internet_service = st.selectbox('Internet Service', ['DSL'], index=0, disabled=True)
            else:
                internet_service = st.selectbox('Internet Service', ['Fiber optic', 'DSL'])
            

        with col2:
            st.subheader('Additional Services')
            online_security = st.selectbox('Online Security', ['No', 'Yes'])
            online_backup = st.selectbox('Online Backup', ['No', 'Yes'])
            device_protection = st.selectbox('Device Protection', ['No', 'Yes'])
            tech_support = st.selectbox('Tech Support', ['No', 'Yes'])
            streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes'])
            streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes'])
            
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
        if not customer_name.strip():
            st.error("Please enter a customer name")
            return

        input_data = {
            'customer_name': customer_name,
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
            'TotalCharges': total_charges
        }
        with st.spinner('Predicting churn... Please wait...'):
            churn_probability = predict_churn(input_data)

        if edit_mode and edit_data:
            collection.update_one(
                {"_id": edit_data["_id"]},
                {"$set": {
                    **input_data,
                    'churn_probability': churn_probability,
                    'timestamp': datetime.utcnow(),
                    'risk_category': (
                        'High Risk' if churn_probability > 0.6 
                        else 'Medium Risk' if churn_probability > 0.3 
                        else 'Low Risk'
                    )
                }}
            )
            st.success("Customer record updated successfully!")
            st.session_state.edit_mode = False
            st.session_state.edit_customer_data = {}
        else:
            try:
                # Save prediction to MongoDB
                save_prediction_to_mongodb(input_data, churn_probability)
                
                st.subheader('Prediction Results')
                color = 'red' if churn_probability > 0.6 else 'orange' if churn_probability > 0.3 else 'green'
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                        <h3 style="color: {color}; margin-bottom: 10px;">
                            Churn Probability: {churn_probability*100:.1f}%
                        </h3>
                        <p>Customer: {customer_name}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                risk_category = (
                    'High Risk' if churn_probability > 0.6
                    else 'Medium Risk' if churn_probability > 0.3
                    else 'Low Risk'
                )
                st.markdown(f"<h4 style='color: {color};'>Risk Category: {risk_category}</h4>", 
                          unsafe_allow_html=True)
                
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
                
                st.subheader('Recommended Actions')
                if churn_probability > 0.6:
                    st.markdown(f"""
                    🚨 **Immediate Actions Required for {customer_name}:**
                    * Schedule urgent customer outreach
                    * Offer contract upgrade with significant incentives
                    * Provide free trial of security services
                    * Consider personalized discount package
                    * Schedule service quality review
                    """)
                elif churn_probability > 0.3:
                    st.markdown(f"""
                    ⚠️ **Preventive Measures for {customer_name}:**
                    * Initiate customer satisfaction survey
                    * Offer service upgrades at promotional rates
                    * Recommend suitable protection services
                    * Schedule regular check-ins
                    """)
                else:
                    st.markdown(f"""
                    ✅ **Retention Strategies for {customer_name}:**
                    * Maintain regular communication
                    * Consider upselling premium services
                    * Enroll in loyalty rewards program
                    * Monitor service usage patterns
                    """)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # ----------------- Bulk Upload Section -----------------
    st.markdown("---")
    st.subheader("📤 Bulk Upload Customers from CSV")

    uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:", df_uploaded.head())

        if st.button("🚀 Upload Customers to Database"):
            try:
                count = 0
                for _, row in df_uploaded.iterrows():
                    input_data = {
                        'customer_name': row['CustomerName'],
                        'gender': row['gender'],
                        'SeniorCitizen': int(float(row['SeniorCitizen'])),
                        'Partner': row['Partner'],
                        'Dependents': row['Dependents'],
                        'tenure': int(float(row['tenure'])),
                        'PhoneService': row['PhoneService'],
                        'MultipleLines': row['MultipleLines'],
                        'InternetService': row['InternetService'],
                        'OnlineSecurity': row['OnlineSecurity'],
                        'OnlineBackup': row['OnlineBackup'],
                        'DeviceProtection': row['DeviceProtection'],
                        'TechSupport': row['TechSupport'],
                        'StreamingTV': row['StreamingTV'],
                        'StreamingMovies': row['StreamingMovies'],
                        'Contract': row['Contract'],
                        'PaperlessBilling': row['PaperlessBilling'],
                        'PaymentMethod': row['PaymentMethod'],
                        'MonthlyCharges': float(row['MonthlyCharges']),
                        'TotalCharges': float(row['TotalCharges']) if row['TotalCharges'] != " " else 0.0
                    }

                    churn_probability = predict_churn(input_data)

                    document = input_data.copy()
                    document['churn_probability'] = churn_probability
                    document['risk_category'] = (
                        'High Risk' if churn_probability > 0.6
                        else 'Medium Risk' if churn_probability > 0.3
                        else 'Low Risk'
                    )
                    document['timestamp'] = datetime.utcnow()

                    collection.insert_one(document)
                    count += 1

                st.success(f"✅ Successfully uploaded {count} customers!")
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(f"Failed to upload customers: {e}")

def customers_page():
    """Customer page with View, Edit for all roles, Delete only for Admin"""
    st.title('Customer Predictions')

    role = st.session_state.get("role", "guest")  # Get user role

    # Clear cached data if requested
    if st.button("Refresh Data", key="refresh_customers"):
        if 'customer_history_df' in st.session_state:
            del st.session_state.customer_history_df
        st.rerun()

    # Load data from MongoDB
    if 'customer_history_df' not in st.session_state:
        try:
            if role == "fibre":
                history = list(collection.find({"InternetService": "Fiber optic"}).sort("timestamp", -1))
            elif role == "dsl":
                history = list(collection.find({"InternetService": "DSL"}).sort("timestamp", -1))
            else:
                history = list(collection.find().sort("timestamp", -1))

            if not history:
                st.warning("No customers found.")
                return

            history_df = pd.DataFrame(history)
            history_df['formatted_timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            st.session_state.customer_history_df = history_df
        except Exception as e:
            st.error(f"Error loading customer data: {e}")
            return
    else:
        history_df = st.session_state.customer_history_df

    # Search bar
    search = st.text_input("Search Customer Name")
    if search:
        history_df = history_df[history_df['customer_name'].str.contains(search, case=False)]

    # Risk filter
    risk_filter = st.multiselect(
        'Filter by Risk Category',
        options=['High Risk', 'Medium Risk', 'Low Risk'],
        default=['High Risk', 'Medium Risk', 'Low Risk']
    )
    history_df = history_df[history_df['risk_category'].isin(risk_filter)]

     # Download button for customer list
    with st.container():
        csv_customer_data = history_df.to_csv(index=False).encode('utf-8')
        st.markdown("""
            <style>
            .small-button button {
                padding: 4px 12px;
                font-size: 0.8rem;
                border-radius: 6px;
            }
            </style>
            <div class='small-button'>
        """, unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Customers CSV",
            data=csv_customer_data,
            file_name='customer_list.csv',
            mime='text/csv'
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Table headers
    st.subheader("Customer List")
    headers = ["Name", "Gender", "Senior", "Partner", "Dependents", "Contract", "Service", "Churn%", "Risk", "Time", "Actions"]
    cols = st.columns([2, 1, 1, 1, 1, 1.5, 1.5, 1.2, 1.5, 2, 2])

    for col, head in zip(cols, headers):
        col.markdown(f"**{head}**")

    # Prepare modals
    view_modal = Modal("👁️ View Customer", key="view_modal", padding=20)
    edit_modal = Modal("✏️ Edit Customer", key="edit_modal", padding=20)
    delete_modal = Modal("⚠️ Confirm Delete", key="delete_modal", padding=20)

    if 'view_data' not in st.session_state:
        st.session_state.view_data = None
    if 'edit_data' not in st.session_state:
        st.session_state.edit_data = None
    if 'delete_data' not in st.session_state:
        st.session_state.delete_data = None

    # Display customers
    for idx, row in history_df.iterrows():
        cols = st.columns([2, 1, 1, 1, 1, 1.5, 1.5, 1.2, 1.5, 2, 2])
        cols[0].write(row['customer_name'])
        cols[1].write(row['gender'])
        cols[2].write("Yes" if row['SeniorCitizen'] == 1 else "No")
        cols[3].write(row['Partner'])
        cols[4].write(row['Dependents'])
        cols[5].write(row['Contract'])
        cols[6].write(row['InternetService'])
        cols[7].write(f"{row['churn_probability']:.2f}")
        cols[8].markdown(
            f"<div style='background-color: {'#ff4d4d' if row['risk_category']=='High Risk' else '#ffa64d' if row['risk_category']=='Medium Risk' else '#85e085'}; border-radius:5px; text-align:center'>{row['risk_category']}</div>",
            unsafe_allow_html=True
        )
        cols[9].write(row['formatted_timestamp'])

        # Modified buttons based on role
        with cols[10]:
            if role == "admin":
                # Admins see all buttons (view, edit, delete)
                b1, b2, b3 = st.columns(3)
                if b1.button("👁️", key=f"view_{idx}"):
                    st.session_state.view_data = row
                    view_modal.open()

                if b2.button("✏️", key=f"edit_{idx}"):
                    st.session_state.edit_data = row
                    edit_modal.open()

                if b3.button("🗑️", key=f"delete_{idx}"):
                    st.session_state.delete_data = row
                    delete_modal.open()
            else:
                # DSL and Fibre employees only see view and edit (no delete)
                b1, b2 = st.columns(2)
                if b1.button("👁️", key=f"view_{idx}"):
                    st.session_state.view_data = row
                    view_modal.open()

                if b2.button("✏️", key=f"edit_{idx}"):
                    st.session_state.edit_data = row
                    edit_modal.open()

    # View Modal
    if view_modal.is_open() and st.session_state.view_data is not None:
        with view_modal.container():
            data = st.session_state.view_data
            st.write(f"**Customer Name:** {data['customer_name']}")
            st.write(f"**Contract:** {data['Contract']}")
            st.write(f"**Internet Service:** {data['InternetService']}")
            st.write(f"**Churn Probability:** {data['churn_probability']:.2f}")
            st.write(f"**Risk Category:** {data['risk_category']}")
            st.write(f"**Timestamp:** {data['formatted_timestamp']}")

            # Risk Factors
            st.subheader("🎯 Risk Factors Analysis")
            risk_factors = []
            if data['Contract'] == 'Month-to-month':
                risk_factors.append("Month-to-month contract")
            if data['InternetService'] == 'Fiber optic':
                risk_factors.append("Fiber optic service")
            if data.get('tenure', 12) < 12:
                risk_factors.append("Low tenure (<12 months)")
            if data.get('MonthlyCharges', 0) > 70:
                risk_factors.append("High monthly charges")
            if data.get('PaymentMethod', '') == 'Electronic check':
                risk_factors.append("Payment by electronic check")
            if data.get('TechSupport', '') == 'No':
                risk_factors.append("No technical support")
            if data.get('OnlineSecurity', '') == 'No':
                risk_factors.append("No online security")

            for factor in risk_factors:
                st.markdown(f"• {factor}")

            # Recommended Actions
            st.subheader("🛡️ Recommended Retention Strategy")
            if data['churn_probability'] > 0.6:
                st.info("🚨 High Risk: Offer strong incentives, schedule service review, assign dedicated support agent.")
            elif data['churn_probability'] > 0.3:
                st.warning("⚠️ Medium Risk: Offer loyalty discounts, enroll in service perks, follow-up monthly.")
            else:
                st.success("✅ Low Risk: Maintain regular communication, enroll in loyalty program, upsell add-ons.")

            if st.button("Close", key="close_view_modal"):
                st.session_state.view_data = None
                view_modal.close()
                st.rerun()
                
    # Edit Modal
    if edit_modal.is_open() and st.session_state.edit_data is not None:
        with edit_modal.container():
            data = st.session_state.edit_data
            with st.form("edit_form"):
                customer_name = st.text_input("Customer Name", value=data.get('customer_name', ''), disabled=True)
                gender = st.text_input("Gender", value=data.get('gender', ''), disabled=True)
                contract = st.selectbox(
                    "Contract",
                    options=["Month-to-month", "One year", "Two year"],
                    index=["Month-to-month", "One year", "Two year"].index(data.get('Contract', 'Month-to-month'))
                )
                tenure = st.number_input('Tenure (months)', min_value=0, max_value=120, value=data.get('tenure', 1))
                paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'], index=0 if data.get('PaperlessBilling') == 'Yes' else 1)
                payment_method = st.selectbox(
                    "Payment Method",
                    options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                    index=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'].index(data.get('PaymentMethod', 'Electronic check'))
                )
                monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=data.get('MonthlyCharges', 90.0))
                total_charges = st.number_input('Total Charges', min_value=0.0, value=data.get('TotalCharges', 0.0))

                col1, col2 = st.columns(2)
                save = col1.form_submit_button("Save Changes")
                cancel = col2.form_submit_button("Cancel")

                if save:
                    updated_data = {
                        'Contract': contract,
                        'tenure': tenure,
                        'PaperlessBilling': paperless_billing,
                        'PaymentMethod': payment_method,
                        'MonthlyCharges': monthly_charges,
                        'TotalCharges': total_charges
                    }
                    collection.update_one(
                        {"customer_name": data['customer_name']},
                        {"$set": updated_data}
                    )
                    st.success("✅ Customer updated!")
                    if 'customer_history_df' in st.session_state:
                        del st.session_state.customer_history_df
                    st.session_state.edit_data = None
                    edit_modal.close()
                    st.rerun()

                if cancel:
                    st.session_state.edit_data = None
                    edit_modal.close()
                    st.rerun()

    # Delete Modal - Only for Admin
    if delete_modal.is_open() and st.session_state.delete_data is not None and st.session_state.role == "admin":
        with delete_modal.container():
            data = st.session_state.delete_data
            st.warning(f"Are you sure you want to delete **{data['customer_name']}**?")
            col1, col2 = st.columns(2)
            confirm = col1.button("✅ Yes, Delete")
            cancel = col2.button("❌ Cancel")

            if confirm:
                try:
                    collection.delete_one({"customer_name": data['customer_name']})
                    st.success("✅ Customer deleted successfully.")
                    if 'customer_history_df' in st.session_state:
                        del st.session_state.customer_history_df
                    st.session_state.delete_data = None
                    delete_modal.close()
                    st.rerun()
                except Exception as e:
                    st.error(f"Deletion failed: {str(e)}")
                    st.session_state.delete_data = None
                    delete_modal.close()
                    st.rerun()

            if cancel:
                st.session_state.delete_data = None
                delete_modal.close()
                st.rerun()

def prediction_history_page():
    """Analytics dashboard page with ultra-compact layout and additional components"""
    st.title('Analytics Dashboard', help='Overview of customer churn predictions')
    
    try:
        role = st.session_state.get("role", "guest")
        history = get_prediction_history(role)
        history_df = pd.DataFrame(history)
        
        if not history_df.empty:
            # Top row - Key metrics in small cards
            st.markdown("""
                <style>
                .metric-card {
                    background-color: #f8f9fa;
                    padding: 4px;
                    border-radius: 5px;
                    text-align: center;
                }
                </style>
            """, unsafe_allow_html=True)
            
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            with m1:
                st.markdown("""<div class="metric-card">""", unsafe_allow_html=True)
                st.metric("Total", value=len(history_df), delta=f"+{len(history_df[history_df['timestamp'] > datetime.now() - timedelta(days=1)])}")
                st.markdown("""</div>""", unsafe_allow_html=True)
            
            with m2:
                high_risk = len(history_df[history_df['risk_category'] == 'High Risk'])
                st.markdown("""<div class="metric-card">""", unsafe_allow_html=True)
                st.metric("High Risk", value=high_risk)
                st.markdown("""</div>""", unsafe_allow_html=True)

            with m3:
                medium_risk = len(history_df[history_df['risk_category'] == 'Medium Risk'])
                st.markdown("""<div class="metric-card">""", unsafe_allow_html=True)
                st.metric("Medium Risk", value=medium_risk)
                st.markdown("""</div>""", unsafe_allow_html=True)

            with m4:
                low_risk = len(history_df[history_df['risk_category'] == 'Low Risk'])
                st.markdown("""<div class="metric-card">""", unsafe_allow_html=True)
                st.metric("Low Risk", value=low_risk)
                st.markdown("""</div>""", unsafe_allow_html=True)

            with m5:
                avg_prob = history_df['churn_probability'].mean() * 100
                st.markdown("""<div class="metric-card">""", unsafe_allow_html=True)
                st.metric("Avg Churn", value=f"{avg_prob:.1f}%")
                st.markdown("""</div>""", unsafe_allow_html=True)
            
            with m6:
                retention_rate = (1 - history_df['churn_probability'].mean()) * 100
                st.markdown("""<div class="metric-card">""", unsafe_allow_html=True)
                st.metric("Retention", value=f"{retention_rate:.1f}%")
                st.markdown("""</div>""", unsafe_allow_html=True)
            
            
            # First row of charts
            chart_row1_1, chart_row1_2, chart_row1_3 = st.columns([1, 1, 1])
            
            with chart_row1_1:
                # Compact risk distribution
                risk_dist = history_df['risk_category'].value_counts()
                fig_risk = px.pie(
                    values=risk_dist.values,
                    names=risk_dist.index,
                    title='Risk Categories',
                    color_discrete_map={'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'},
                    height=200
                )
                fig_risk.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with chart_row1_2:
                # Contract type distribution
                contract_counts = history_df['Contract'].value_counts().reset_index()
                contract_counts.columns = ['Contract', 'Count']
                contract_dist = px.bar(
                    contract_counts,
                    x='Contract',
                    y='Count',
                    title='Contract Types',
                    height=200
                )
                contract_dist.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="",
                    yaxis_title=""
                )
                st.plotly_chart(contract_dist, use_container_width=True)
            
            with chart_row1_3:
                # Payment method distribution
                payment_counts = history_df['PaymentMethod'].value_counts()
                payment_dist = px.pie(
                    values=payment_counts.values,
                    names=payment_counts.index,
                    title='Payment Methods',
                    height=200
                )
                payment_dist.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=False
                )
                st.plotly_chart(payment_dist, use_container_width=True)
            
            # Second row of charts
            chart_row2_1, chart_row2_2 = st.columns([2, 1])
            
            with chart_row2_1:
                # Churn probability by contract type
                fig_prob_contract = px.box(
                    history_df,
                    x='Contract',
                    y='churn_probability',
                    color='risk_category',
                    title='Churn Probability by Contract',
                    height=200,
                    color_discrete_map={'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'}
                )
                fig_prob_contract.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_prob_contract, use_container_width=True)
            
            with chart_row2_2:
                if role == "admin":
                    # Internet Services chart (for Admin only)
                    service_counts = history_df['InternetService'].value_counts().reset_index()
                    service_counts.columns = ['InternetService', 'Count']
                    fig_service = px.bar(
                        service_counts,
                        x='InternetService',
                        y='Count',
                        color='InternetService',
                        title='Internet Services Distribution',
                        height=200
                    )
                    fig_service.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_service, use_container_width=True)
                else:
                    # DSL/Fibre: Churn Risk by Payment Method
                    payment_risk = history_df.groupby('PaymentMethod')['churn_probability'].mean().reset_index()
                    fig_payment_risk = px.bar(
                        payment_risk,
                        x='PaymentMethod',
                        y='churn_probability',
                        title='Churn Risk by Payment Method',
                        labels={'churn_probability': 'Avg Churn Probability'},
                        height=200
                    )          
                    fig_payment_risk.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_payment_risk, use_container_width=True)

            # Third row - Additional metrics and trends
            row3_1, row3_2 = st.columns([1, 1])
            
            with row3_1:
                # Recent predictions table
                st.markdown("### Recent Predictions")
                recent_predictions = history_df.nlargest(5, 'timestamp')[
                    ['customer_name', 'risk_category', 'churn_probability']
                ].copy()
                recent_predictions['churn_probability'] = recent_predictions['churn_probability'].apply(
                    lambda x: f"{x*100:.1f}%"
                )
                st.dataframe(recent_predictions, height=150)
            
            with row3_2:
                # Risk trend
                st.markdown("### Risk Trend")
                history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
                risk_trend = history_df.groupby('date')['churn_probability'].mean().reset_index()
                fig_trend = px.line(
                    risk_trend,
                    x='date',
                    y='churn_probability',
                    title='Average Risk Trend',
                    height=200
                )
                fig_trend.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_trend, use_container_width=True)
                
            # Bottom section - Quick stats
            st.markdown("### Quick Stats")
            stats_1, stats_2, stats_3, stats_4 = st.columns(4)
            
            with stats_1:
                senior_pct = (history_df['SeniorCitizen'] == 1).mean() * 100
                st.markdown(f"**Senior Citizens:** {senior_pct:.1f}%")
            
            with stats_2:
                partner_pct = (history_df['Partner'] == 'Yes').mean() * 100
                st.markdown(f"**With Partner:** {partner_pct:.1f}%")
            
            with stats_3:
                dependents_pct = (history_df['Dependents'] == 'Yes').mean() * 100
                st.markdown(f"**With Dependents:** {dependents_pct:.1f}%")
            
            with stats_4:
                paperless_pct = (history_df['PaperlessBilling'] == 'Yes').mean() * 100
                st.markdown(f"**Paperless Billing:** {paperless_pct:.1f}%")

        else:
            st.warning("No prediction history available for analysis.")
            
    except Exception as e:
        st.error(f"Error generating analytics: {e}")

def interaction_history_page():
    interaction_collection = db["interaction_history"]
    role = st.session_state.get("role", "guest")

    # Reset the interaction history when page loads or explicitly refreshed
    if 'interaction_history_loaded' not in st.session_state or st.button("Refresh Data", key="refresh_interactions"):
        try:
            if role == "dsl":
                all_interactions = list(interaction_collection.find({"service": "DSL"}))
            elif role == "fibre":
                all_interactions = list(interaction_collection.find({"service": "Fiber"}))
            else:  # admin sees all
                all_interactions = list(interaction_collection.find({}))
                
            st.session_state.interaction_history = all_interactions
            st.session_state.interaction_history_loaded = True
        except Exception as e:
            st.error(f"Error loading interaction history: {e}")
            return

    # Get customer names based on role
    if role in ["dsl", "fibre"]:
        service_type = "DSL" if role == "dsl" else "Fiber"
        customer_names = {c.get("customer", "") for c in st.session_state.interaction_history if "customer" in c}
    else:
        customer_names = None  # Admin sees all

    if 'interaction_to_view' not in st.session_state:
        st.session_state.interaction_to_view = None

    if 'interaction_to_edit' not in st.session_state:
        st.session_state.interaction_to_edit = None

    if 'interaction_to_delete' not in st.session_state:
        st.session_state.interaction_to_delete = None

    view_modal = Modal(key="view_modal", title="👁️ View Interaction")
    edit_modal = Modal(key="edit_modal", title="✏️ Edit Interaction")
    delete_modal = Modal(key="delete_modal", title="🔥 Confirm Deletion")

    tab1, tab2 = st.tabs(["View Interactions", "Add Interaction"])

    with tab1:
        st.subheader("Interaction History")

        filtered_interactions = st.session_state.get("interaction_history", [])

        if filtered_interactions:
            try:
                df = pd.DataFrame(filtered_interactions)
                
                with st.container():
                    csv_interaction_data = df.to_csv(index=False).encode('utf-8')
                    st.markdown("""
                        <style>
                        .small-button button {
                            padding: 4px 12px;
                            font-size: 0.8rem;
                            border-radius: 6px;
                        }
                        </style>
                        <div class='small-button'>
                    """, unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Interactions CSV",
                        data=csv_interaction_data,
                        file_name='interaction_list.csv',
                        mime='text/csv'
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Team Member Filter
                employee_list = sorted(df['team_member'].dropna().unique())
                selected_employee = st.selectbox("Filter by Team Member", options=["All"] + employee_list)

                if selected_employee != "All":
                    df = df[df['team_member'] == selected_employee]

                # Table Headers
                headers = st.columns([2, 2, 2, 1.5, 1, 1.5, 1.5, 1, 1.5, 1.5])
                headers[0].markdown("**Customer**")
                headers[1].markdown("**Team Member**")
                headers[2].markdown("**Service**")
                headers[3].markdown("**Type**")
                headers[4].markdown("**Marketing**")
                headers[5].markdown("**Product Update**")
                headers[6].markdown("**Reported Hidden Fees?**")
                headers[7].markdown("**Status**")
                headers[8].markdown("**Next Follow-Up Date**")
                headers[9].markdown("**Actions**")

                # Table Rows - Modified for Delete visibility
                for idx, row in df.iterrows():
                    cols = st.columns([2, 2, 2, 1.5, 1, 1.5, 1.5, 1.5, 1, 1.5])
                    cols[0].write(row.get('customer', ''))
                    cols[1].write(row.get('team_member', ''))
                    cols[2].write(row.get('service', 'Unknown'))
                    cols[3].write(row.get('interaction_type', ''))
                    cols[4].write("Yes" if row.get('marketing_logs') and len(row.get('marketing_logs')) > 0 else "No")
                    cols[5].write("Yes" if row.get('product_update_logs') and len(row.get('product_update_logs')) > 0 else "No")
                    cols[6].write("Yes" if row.get('reported_hidden_fees', False) else "No")

                    status = row.get('status', '')
                    def colored_status_badge(text, bg_color):
                       return f"""
                       <span style='
                           background-color: {bg_color};
                           color: white;
                           padding: 2px 8px;
                           border-radius: 8px;
                           font-weight: bold;
                           font-size: 0.85rem;
                           display: inline-block;
                       '>{text}</span>
                       """

                    if status == "In Progress":
                         cols[7].markdown(colored_status_badge("In Progress", "#FFA500"), unsafe_allow_html=True)
                    elif status == "On Hold":
                         cols[7].markdown(colored_status_badge("On Hold", "#1E90FF"), unsafe_allow_html=True)
                    elif status == "Closed":
                         cols[7].markdown(colored_status_badge("Closed", "#28a745"), unsafe_allow_html=True)
                    else:
                         cols[7].markdown(colored_status_badge(status, "#6c757d"), unsafe_allow_html=True)

                    cols[8].write(row.get('next_follow_up', 'N/A'))

                    with cols[9]:
                        if role == "admin":
                            # Admin sees view, edit, and delete
                            view_col, edit_col, delete_col = st.columns(3)
                            if view_col.button("👁️", key=f"view_{idx}"):
                                st.session_state.interaction_to_view = row.to_dict()
                                view_modal.open()
                            if edit_col.button("✏️", key=f"edit_{idx}"):
                                st.session_state.interaction_to_edit = row.to_dict()
                                edit_modal.open()
                            if delete_col.button("🗑️", key=f"delete_{idx}"):
                                st.session_state.interaction_to_delete = row.to_dict()
                                delete_modal.open()
                        else:
                            # DSL and Fibre only see view and edit
                            view_col, edit_col = st.columns(2)
                            if view_col.button("👁️", key=f"view_{idx}"):
                                st.session_state.interaction_to_view = row.to_dict()
                                view_modal.open()
                            if edit_col.button("✏️", key=f"edit_{idx}"):
                                st.session_state.interaction_to_edit = row.to_dict()
                                edit_modal.open()
            except Exception as e:
                st.error(f"Error displaying interaction table: {e}")

        else:
            st.info("No interaction history to display.")

    # View Modal
    if view_modal.is_open():
        interaction = st.session_state.interaction_to_view
        if interaction:
            with view_modal.container():
                st.markdown(f"### Details for {interaction['customer']}")
                st.write(f"**Team Member:** {interaction.get('team_member', 'N/A')}")
                st.write(f"**Service:** {interaction.get('service', 'Unknown')}")
                st.write(f"**Interaction Type:** {interaction.get('interaction_type', 'N/A')}")
                st.write(f"**Status:** {interaction.get('status', 'N/A')}")
                st.write(f"**Reported Hidden Fees?:** {'Yes' if interaction.get('reported_hidden_fees', False) else 'No'}")
                st.write(f"**Summary:** {interaction.get('summary', 'N/A')}")
                st.write(f"**Interaction Date:** {interaction.get('interaction_date', 'N/A')}")
                st.write(f"**Next Follow-Up:** {interaction.get('next_follow_up', 'N/A')}")

                # Display marketing logs
                if "marketing_logs" in interaction and isinstance(interaction["marketing_logs"], list):
                    st.markdown("### Marketing Logs")
                    for log in interaction["marketing_logs"]:
                       if log.strip():
                          st.write(log)

                # Display product update logs
                if "product_update_logs" in interaction and isinstance(interaction["product_update_logs"], list):
                    st.markdown("### Product Update Logs")
                    for log in interaction["product_update_logs"]:
                       if log.strip():
                          st.write(log)
                
                if st.button("Close", key="close_view_interaction"):
                    st.session_state.interaction_to_view = None
                    view_modal.close()
                    st.rerun()
                
    # Edit Modal
    if edit_modal.is_open():
        interaction = st.session_state.interaction_to_edit
        if interaction:
            with edit_modal.container():
                with st.form("edit_interaction_form"):
                    interaction_type = st.selectbox(
                        "Interaction Type",
                        options=["Call", "Email", "Demo / Meeting", "Support", "Other"],
                        index=["Call", "Email", "Demo / Meeting", "Support", "Other"].index(interaction.get('interaction_type', 'Call'))
                    )
                    reported_hidden_fees = st.checkbox(
                        "Reported Hidden Fees?",
                        value=interaction.get('reported_hidden_fees', False)
                    )
                    summary = st.text_area("Summary", value=interaction.get('summary', ''))
                    marketing_logs = st.text_area("Marketing Logs", value="\n".join(interaction.get('marketing_logs', [])))
                    product_updates = st.text_area("Product Updates", value="\n".join(interaction.get('product_update_logs', [])))
                    status = st.selectbox(
                        "Status",
                        options=["In Progress", "Closed", "On Hold"],
                        index=["In Progress", "Closed", "On Hold"].index(interaction.get('status', 'In Progress'))
                        )
                    interaction_date = st.date_input(  
                        "Interaction Date",
                        value=pd.to_datetime(interaction.get('interaction_date', datetime.now()))
                    )
                    next_follow_up = st.date_input(
                        "Next Follow-Up Date",
                        value=pd.to_datetime(interaction.get('next_follow_up', datetime.now()))
                    )
                    save = st.form_submit_button("Save Changes")
                    if save:
                        try:
                            interaction_collection.update_one(
                                {"_id": interaction["_id"]},
                                {"$set": {
                                    "interaction_type": interaction_type,
                                    "reported_hidden_fees": reported_hidden_fees,
                                    "summary": summary,
                                    "marketing_logs": [log for log in marketing_logs.split("\n") if log.strip()],
                                    "product_update_logs": [log for log in product_updates.split("\n") if log.strip()],
                                    "status": status,
                                    "interaction_date": interaction_date.strftime("%Y-%m-%d"),
                                    "next_follow_up": next_follow_up.strftime("%Y-%m-%d")
                                }}
                            )
                            st.success("✅ Interaction updated successfully!")
                            st.session_state.interaction_to_edit = None
                            st.session_state.interaction_history_loaded = False
                            edit_modal.close()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update interaction: {e}")

    # Delete Modal - Only shown to admin
    if delete_modal.is_open() and st.session_state.role == "admin":
        selected_delete = st.session_state.interaction_to_delete

        if selected_delete is None:
            delete_modal.close()
            st.rerun()

        with delete_modal.container():
            st.warning(f"Are you sure you want to delete **{selected_delete['customer']}**?")

            col1, col2 = st.columns(2)
            if col1.button("Yes, Delete", key="confirm_delete_modal"):
                try:
                    interaction_collection.delete_one({
                        "_id": selected_delete['_id']
                    })
                    
                    st.success("✅ Interaction deleted successfully!")
                    st.session_state.interaction_to_delete = None
                    st.session_state.interaction_history_loaded = False
                    delete_modal.close()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete interaction: {e}")

            if col2.button("Cancel", key="cancel_delete_modal"):
                st.session_state.interaction_to_delete = None
                delete_modal.close()
                st.rerun()

    # Add Interaction Form
    with tab2:
        st.subheader("Add Interaction")
        with st.form("add_interaction_form"):
            customer_name = st.text_input("Customer Name")
            team_member = st.text_input("Team Member")
            
            # Service selection based on role
            if role == "admin":
                service = st.selectbox("Service", options=["DSL", "Fiber"])
            elif role == "fibre":
                service = "Fiber"  # Fixed value for fiber employees
                st.text_input("Service", value=service, disabled=True)
            elif role == "dsl":
                service = "DSL"  # Fixed value for DSL employees
                st.text_input("Service", value=service, disabled=True)
            else:
                service = "Unknown"
            
            interaction_type = st.selectbox("Interaction Type", options=["Call", "Email", "Demo / Meeting", "Support", "Other"])
            reported_hidden_fees = st.checkbox("Reported Hidden Fees?")
            status = st.selectbox("Status", options=["In Progress", "On Hold", "Closed"])
            summary = st.text_area("Summary")
            marketing_logs = st.text_area("Marketing Logs", help="Separate entries with a new line.")
            product_updates = st.text_area("Product Updates", help="Separate entries with a new line.")
            interaction_date = st.date_input("Interaction Date", value=datetime.now())
            next_follow_up = st.date_input("Next Follow-Up Date")
            submit = st.form_submit_button("Add Interaction")
            if submit:
               if not customer_name.strip():
                  st.error("Please enter a customer name")
               elif not team_member.strip():
                  st.error("Please enter team member name")
               else:
                try:
                    interaction_collection.insert_one({
                        "customer": customer_name,
                        "team_member": team_member,
                        "service": service,
                        "interaction_type": interaction_type,
                        "reported_hidden_fees": reported_hidden_fees,
                        "status": status,
                        "summary": summary,
                        "marketing_logs": [log for log in marketing_logs.split("\n") if log.strip()],
                        "product_update_logs": [log for log in product_updates.split("\n") if log.strip()],
                        "interaction_date": interaction_date.strftime("%Y-%m-%d"), 
                        "next_follow_up": next_follow_up.strftime("%Y-%m-%d"),
                    })
                    st.success("✅ Interaction added successfully!")
                    st.session_state.interaction_history_loaded = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add interaction: {e}")

# Main app flow
def main():
    # First check for existing authentication in session state
    if 'is_authenticated' not in st.session_state:
        st.session_state['is_authenticated'] = False
    
    # Check for auth token in session and verify it against DB
    if not st.session_state.get('is_authenticated', False) and 'auth_token' in st.session_state:
        # Try to validate the token from the database
        auth_token = st.session_state['auth_token']
        user = users_collection.find_one({"auth_token": auth_token})
        if user:
            # Auto re-authenticate based on token
            st.session_state['is_authenticated'] = True
            st.session_state['username'] = user.get('email')
            st.session_state['role'] = user.get('role', 'guest')
            st.session_state['user_email'] = user.get('email')
    
    # Check URL parameters 
    params = st.query_params
    if 'page' in params:
        page_from_url = params['page'].lower()
        if page_from_url in ['dashboard', 'prediction', 'customers', 'interaction']:
            st.session_state['page'] = page_from_url.capitalize()
            if page_from_url == 'interaction':
                st.session_state['page'] = 'Interaction History'
    
    # Now check if user is authenticated
    if not st.session_state.get('is_authenticated', False):
        login_page()
        return
    
    # Handle logout
    if st.session_state.get("logout_triggered", False):
        handle_logout()
        st.rerun()
        return

    # Navigation sidebar
    with st.sidebar:
        try:
            st.image("logo.png", width=150)
        except:
            pass

        role = st.session_state.get("role", "guest")

        if role == "admin":
            st.title("Telebond Admin Portal")
        elif role == "dsl":
             st.title("Telebond DSL Portal")
        elif role == "fibre":
             st.title("Telebond Fibre Portal")
        else:
             st.title("Telebond Portal")

        # Navigation buttons with URL parameter updates
        if st.button("Dashboard", use_container_width=True):
            st.session_state['page'] = "Dashboard"
            st.query_params['page'] = 'dashboard'
            st.rerun()
            
        if st.button("Prediction", use_container_width=True):
            st.session_state['page'] = "Prediction"
            st.query_params['page'] = 'prediction'
            st.rerun()
            
        if st.button("Customers", use_container_width=True):
            st.session_state['page'] = "Customers"
            st.query_params['page'] = 'customers'
            st.rerun()
            
        if st.button("Interaction History", use_container_width=True):
            st.session_state['page'] = "Interaction History"
            st.query_params['page'] = 'interaction'
            st.rerun()

        st.markdown("---")
        if st.button("Logout", type="secondary", use_container_width=True):
            st.session_state["logout_triggered"] = True
            st.rerun()

    # Main content based on selected page
    selected_page = st.session_state.get('page', 'Dashboard')
    
    if selected_page == "Dashboard":
        prediction_history_page()
    elif selected_page == "Prediction":
        prediction_page()
    elif selected_page == "Customers":
        customers_page()
    elif selected_page == "Interaction History":
        interaction_history_page()

if __name__ == "__main__":
    main()