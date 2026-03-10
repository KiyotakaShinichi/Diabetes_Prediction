"""
Diabetes Risk Assessment System
Clinical Decision Support Tool for Healthcare Professionals

Run with: streamlit run streamlit_app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Diabetes Risk Assessment System",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# Custom CSS for Medical Theme
# ---------------------------
st.markdown("""
<style>
    /* Medical blue color scheme */
    .stApp {
        background-color: #f8fafc;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0077b6, #00b4d8);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    
    .risk-high {
        background-color: #fee2e2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    
    .risk-low {
        background-color: #dcfce7;
        border-left: 4px solid #16a34a;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    
    .info-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .disclaimer {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    
    .section-header {
        color: #0077b6;
        border-bottom: 2px solid #0077b6;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Model Loading
# ---------------------------
MODEL_BUNDLE_PATH = Path("model_artifacts/model_bundle.pkl")


@st.cache_resource
def load_model():
    if not MODEL_BUNDLE_PATH.exists():
        return None, None, None, None
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    return (
        bundle["pipeline"],
        float(bundle["threshold"]),
        bundle["feature_columns"],
        bundle.get("feature_labels", {}),
    )


pipeline, threshold, feature_columns, feature_labels = load_model()

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="main-header">
    <h1>🏥 Diabetes Risk Assessment System</h1>
    <p>Clinical Decision Support Tool | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Check if model is loaded
if pipeline is None:
    st.error("⚠️ Model not found. Please run `python logisticregression_only.py` to train the model first.")
    st.stop()

# ---------------------------
# Input Mappings
# ---------------------------
genhlth_options = {
    "Excellent": 1,
    "Very Good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5,
}

age_options = {
    "18-24 years": 1,
    "25-29 years": 2,
    "30-34 years": 3,
    "35-39 years": 4,
    "40-44 years": 5,
    "45-49 years": 6,
    "50-54 years": 7,
    "55-59 years": 8,
    "60-64 years": 9,
    "65-69 years": 10,
    "70-74 years": 11,
    "75-79 years": 12,
    "80+ years": 13,
}

education_options = {
    "Never attended school": 1,
    "Elementary (Grades 1-8)": 2,
    "Some high school (Grades 9-11)": 3,
    "High school graduate / GED": 4,
    "Some college / Technical school": 5,
    "College graduate or higher": 6,
}

binary_yes_no = {"No": 0, "Yes": 1}

# ---------------------------
# Tabs
# ---------------------------
tab_assess, tab_info = st.tabs(["📋 Risk Assessment", "ℹ️ Clinical Information"])

with tab_assess:
    st.markdown('<h3 class="section-header">Patient Clinical Data</h3>', unsafe_allow_html=True)
    
    with st.form("assessment_form"):
        # Demographics Section
        st.markdown("**Demographics**")
        col1, col2 = st.columns(2)
        with col1:
            age_label = st.selectbox(
                "Age Group",
                options=list(age_options.keys()),
                index=6,
                help="Patient's age category"
            )
        with col2:
            education_label = st.selectbox(
                "Education Level",
                options=list(education_options.keys()),
                index=5,
                help="Highest education level attained"
            )
        
        # Health Status Section
        st.markdown("**General Health Status**")
        col1, col2, col3 = st.columns(3)
        with col1:
            genhlth_label = st.selectbox(
                "Self-Reported Health",
                options=list(genhlth_options.keys()),
                index=2,
                help="Patient's perception of overall health"
            )
        with col2:
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=10.0,
                max_value=80.0,
                value=27.0,
                step=0.1,
                help="Body Mass Index"
            )
        with col3:
            phys_hlth = st.number_input(
                "Poor Physical Health Days",
                min_value=0,
                max_value=30,
                value=0,
                help="Days in past 30 with poor physical health"
            )
        
        # Cardiovascular Risk Factors
        st.markdown("**Cardiovascular Risk Factors**")
        col1, col2, col3 = st.columns(3)
        with col1:
            high_bp_label = st.selectbox(
                "High Blood Pressure",
                options=list(binary_yes_no.keys()),
                index=0,
                help="History of hypertension"
            )
        with col2:
            high_chol_label = st.selectbox(
                "High Cholesterol",
                options=list(binary_yes_no.keys()),
                index=0,
                help="History of hypercholesterolemia"
            )
        with col3:
            heart_label = st.selectbox(
                "Heart Disease/Attack",
                options=list(binary_yes_no.keys()),
                index=0,
                help="History of coronary heart disease or MI"
            )
        
        # Lifestyle Factors
        st.markdown("**Lifestyle Factors**")
        col1, col2 = st.columns(2)
        with col1:
            phys_activity_label = st.selectbox(
                "Physically Active",
                options=list(binary_yes_no.keys()),
                index=1,
                help="Physical activity in past 30 days (non-work)"
            )
        with col2:
            diff_walk_label = st.selectbox(
                "Difficulty Walking",
                options=list(binary_yes_no.keys()),
                index=0,
                help="Difficulty walking or climbing stairs"
            )
        
        # Submit Button
        st.markdown("---")
        submitted = st.form_submit_button(
            "🔬 Assess Diabetes Risk",
            use_container_width=True,
            type="primary"
        )
    
    if submitted:
        # Build feature vector (order must match SELECTED_FEATURES)
        payload = {
            "GenHlth": genhlth_options[genhlth_label],
            "HighBP": binary_yes_no[high_bp_label],
            "BMI": bmi,
            "HighChol": binary_yes_no[high_chol_label],
            "Age": age_options[age_label],
            "DiffWalk": binary_yes_no[diff_walk_label],
            "HeartDiseaseorAttack": binary_yes_no[heart_label],
            "PhysHlth": phys_hlth,
            "Education": education_options[education_label],
            "PhysActivity": binary_yes_no[phys_activity_label],
        }
        
        input_df = pd.DataFrame([payload])[feature_columns]
        probability = float(pipeline.predict_proba(input_df)[:, 1][0])
        prediction = int(probability >= threshold)
        
        # Results Display
        st.markdown('<h3 class="section-header">Assessment Results</h3>', unsafe_allow_html=True)
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{probability:.1%}")
        col2.metric("Classification Threshold", f"{threshold:.1%}")
        col3.metric("Risk Category", "HIGH" if prediction == 1 else "LOW")
        
        # Result Card
        if prediction == 1:
            st.markdown("""
            <div class="risk-high">
                <strong>⚠️ ELEVATED DIABETES RISK</strong><br>
                The patient's clinical profile indicates an elevated risk of diabetes. 
                Recommend further diagnostic evaluation including fasting glucose, HbA1c, 
                and oral glucose tolerance test (OGTT) as appropriate.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-low">
                <strong>✓ LOWER DIABETES RISK</strong><br>
                The patient's clinical profile indicates a lower risk of diabetes based on 
                the assessed factors. Continue routine screening per clinical guidelines.
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Factor Summary
        with st.expander("📊 Factor Analysis"):
            st.write("**Input Summary:**")
            summary_df = pd.DataFrame([{
                "Factor": k,
                "Value": v,
                "Description": feature_labels.get(k, k)
            } for k, v in payload.items()])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
            <strong>⚕️ Clinical Disclaimer:</strong> This tool is intended for clinical decision 
            support only and should not replace professional medical judgment. All diagnostic and 
            treatment decisions should be made by qualified healthcare providers in consultation 
            with the patient.
        </div>
        """, unsafe_allow_html=True)

with tab_info:
    st.markdown('<h3 class="section-header">About This Assessment Tool</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    This diabetes risk assessment system uses a logistic regression model trained on 
    population health survey data to estimate the probability of diabetes based on 
    clinical and lifestyle factors.
    
    ### Model Methodology
    
    - **Algorithm:** Logistic Regression with L2 regularization
    - **Hyperparameter Optimization:** Optuna (100 trials, 5-fold CV)
    - **Threshold Selection:** Youden's J statistic (maximizes sensitivity + specificity)
    - **Validation:** Separate train/validation/test splits to prevent data leakage
    
    ### Risk Factors Assessed
    
    | Factor | Description |
    |--------|-------------|
    | General Health | Self-reported overall health status (1-5 scale) |
    | High Blood Pressure | History of hypertension diagnosis |
    | BMI | Body Mass Index (kg/m²) |
    | High Cholesterol | History of hypercholesterolemia |
    | Age Category | Age group (13 categories from 18-24 to 80+) |
    | Difficulty Walking | Mobility impairment indicator |
    | Heart Disease/Attack | History of CHD or myocardial infarction |
    | Poor Physical Health Days | Days of poor physical health in past month |
    | Education Level | Highest education attained (6 levels) |
    | Physical Activity | Non-occupational physical activity |
    
    ### Interpretation Guidelines
    
    - **Risk Score:** Probability of diabetes (0-100%)
    - **Threshold:** Model-optimized cutoff for classification (Youden's J)
    - **HIGH Risk:** Score ≥ threshold - recommend confirmatory testing
    - **LOW Risk:** Score < threshold - continue routine screening
    
    ### Limitations
    
    - This tool assesses risk based on survey-derived features only
    - Does not incorporate laboratory values (glucose, HbA1c)
    - Should be used as screening support, not diagnostic confirmation
    - Population-level model may not capture individual variability
    """)
    
    st.markdown("""
    <div class="disclaimer">
        <strong>Data Source:</strong> Model trained on CDC Behavioral Risk Factor 
        Surveillance System (BRFSS) data focusing on diabetes-related health indicators.
    </div>
    """, unsafe_allow_html=True)
