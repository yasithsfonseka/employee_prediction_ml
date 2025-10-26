# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👋",
    layout="wide"
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once and cache it for efficiency.
# This is the recommended way for loading models and other heavy resources.
@st.cache_resource
def load_model(model_path):
    """
    Loads the trained scikit-learn pipeline from the specified path.
    Handles errors if the file is not found.
    """
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'")
        st.info("Please ensure the 'employee_attrition_model.joblib' file is in the same directory as the Streamlit app.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the trained model pipeline
model_pipeline = load_model('employee_attrition_model.joblib')

# --- Application Title and Description ---
st.title("👨‍💼 Employee Attrition Prediction App")
st.markdown("""
This application predicts whether an employee is likely to leave the company or stay. 
Please provide the employee's details using the input fields below and click the **Predict** button.
""")

# --- Input Form ---
# Using st.form to group inputs and have a single submission button.
# This prevents the app from rerunning on every widget interaction.
with st.form(key='employee_input_form'):
    st.header("Employee Details")
    
    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        # Feature: Education (Categorical)
        # Options are derived from the unique values in the notebook's dataset.
        education = st.selectbox(
            label='🎓 Education Level',
            options=['Bachelors', 'Masters', 'PHD'],
            index=0  # Default to 'Bachelors'
        )

        # Feature: Joining Year (Numerical)
        # Min/max values are based on the df.describe() output from the notebook.
        joining_year = st.number_input(
            label='🗓️ Joining Year',
            min_value=2012,
            max_value=2018,
            value=2015,
            step=1,
            help="Year the employee joined the company (between 2012 and 2018)."
        )

        # Feature: Payment Tier (Categorical, treated as a number)
        payment_tier = st.selectbox(
            label='💰 Payment Tier',
            options=[1, 2, 3],
            index=2, # Default to tier 3
            help="Company's salary structure tier (1: lowest, 3: highest)."
        )
        
        # Feature: Ever Benched (Categorical)
        ever_benched = st.selectbox(
            label='⏰ Ever Benched?',
            options=['No', 'Yes'],
            index=0
        )

    with col2:
        # Feature: City (Categorical)
        city = st.selectbox(
            label='🏙️ City',
            options=['Bangalore', 'Pune', 'New Delhi'],
            index=0
        )

        # Feature: Age (Numerical)
        age = st.number_input(
            label='🎂 Age',
            min_value=22,
            max_value=41,
            value=29,
            step=1
        )

        # Feature: Gender (Categorical)
        gender = st.selectbox(
            label='👤 Gender',
            options=['Male', 'Female'],
            index=0
        )
        
        # Feature: Experience in Current Domain (Numerical)
        experience = st.number_input(
            label='🔧 Experience in Current Domain (Years)',
            min_value=0,
            max_value=7,
            value=3,
            step=1
        )

    # Submit button for the form
    submit_button = st.form_submit_button(label='Predict Attrition', type="primary", use_container_width=True)


# --- Prediction Logic and Output ---
if submit_button:
    if model_pipeline is not None:
        # Create a dictionary from the user inputs
        # The keys must exactly match the feature names used during model training.
        input_data = {
            'Education': education,
            'JoiningYear': joining_year,
            'City': city,
            'PaymentTier': payment_tier,
            'Age': age,
            'Gender': gender,
            'EverBenched': ever_benched,
            'ExperienceInCurrentDomain': experience
        }

        # Convert the dictionary into a Pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        st.subheader("Input Data")
        st.dataframe(input_df)

        try:
            # Make prediction using the loaded pipeline
            # The pipeline handles both preprocessing (scaling, encoding) and prediction.
            prediction = model_pipeline.predict(input_df)
            prediction_proba = model_pipeline.predict_proba(input_df)

            # Extract the probability of the 'Leave' class (class 1)
            leave_probability = prediction_proba[0][1]

            # Display the prediction result
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error('Prediction: Employee is likely to LEAVE 🚶‍♂️', icon="🚨")
            else:
                st.success('Prediction: Employee is likely to STAY 👍', icon="✅")

            # Display the confidence probability using st.metric
            st.metric(
                label="Confidence (Probability of Leaving)",
                value=f"{leave_probability:.2%}",
                help="This is the model's confidence that the employee will leave."
            )
            
            # Display probabilities for both classes in an expander
            with st.expander("View Detailed Probabilities"):
                st.write({
                    'Probability of Staying': f"{prediction_proba[0][0]:.2%}",
                    'Probability of Leaving': f"{prediction_proba[0][1]:.2%}"
                })

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model is not loaded. Cannot make a prediction.")