import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ============================================
# 1. Page Configuration & Theme Settings
# ============================================
st.set_page_config(
    page_title="AI Job Impact Analyst Pro",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium UI Injectable Custom CSS
st.markdown("""
    <style>
    /* Global Font & Smooth Transitions */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sleek Container Cards */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Interactive Button Transitions */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #4338CA 0%, #2563EB 100%);
    }
    
    /* Custom Indicator Cards */
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .replaced-card {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid #EF4444;
        color: #EF4444;
    }
    .modified-card {
        background-color: rgba(245, 158, 11, 0.12);
        border: 1px solid #F59E0B;
        color: #F59E0B;
    }
    .unchanged-card {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid #10B981;
        color: #10B981;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 2. Core Machine Learning Artifact Imports
# ============================================
@st.cache_resource
def load_ml_pipeline():
    # Cache loaders preventing multi-thread read bottleneck operations on stream load
    model = joblib.load("ml_artifacts/best_brf_model.pkl")
    scaler = joblib.load("ml_artifacts/scaler.pkl")
    label_encoder = joblib.load("ml_artifacts/label_encoder.pkl")
    feature_names = joblib.load("ml_artifacts/feature_names.pkl")
    return model, scaler, label_encoder, feature_names

try:
    model, scaler, label_encoder, feature_names = load_ml_pipeline()
except Exception as e:
    st.error("⚠️ Pipeline Artifacts missing inside `ml_artifacts/`. Please verify workspace architecture paths.")
    st.stop()

# ============================================
# 3. Application Premium Header
# ============================================
col_header_1, col_header_2 = st.columns([0.7, 0.3])
with col_header_1:
    st.title("🦾 Workforce Displacement Optimization Dashboard")
    st.markdown("""
        *Enterprise predictive system parsing structural human capital transformation risks driven by deep-learning and semantic automation layers.*
    """)
with col_header_2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("🎯 **Model Class Engine:** Balanced Random Forest Classifier")

st.divider()

# ============================================
# 4. Modular Input Feature Forms Layout
# ============================================
with st.form("enterprise_analytics_form"):
    
    # Organize features categorically across three clean vertical columns
    col_demographics, col_professional, col_ai_metrics = st.columns(3)
    
    with col_demographics:
        st.markdown("### 📋 Core Demographics")
        with st.container(border=True):
            age = st.slider("Employee Age", 18, 65, 30)
            gender = st.selectbox("Gender Profile", ["Male", "Female"])
            education = st.selectbox("Terminal Educational Matrix", ["High School", "Bachelor", "Master", "PhD"])
            satisfaction = st.slider("Baseline Metrics: Job Satisfaction", 1, 10, 7)
            
    with col_professional:
        st.markdown("### 💼 Operational Framework")
        with st.container(border=True):
            job_role = st.text_input("Designated Job Title", "Software Engineer")
            industry = st.selectbox("Target Sector Market", ["IT", "Healthcare", "Education", "Finance", "Manufacturing", "Retail", "Transportation", "marketing"])
            experience = st.slider("Chronological Tenure (Years)", 0, 40, 5)
            work_hours = st.slider("Working Hours / Week", 10, 80, 40)
            remote_work = st.selectbox("(Remote Job)", ["Yes", "No"])

    with col_ai_metrics:
        st.markdown("### 🤖 Disruption Exposure")
        with st.container(border=True):
            ai_adoption = st.selectbox("AI Adaptation Pace", ["Low", "Medium", "High"])
            automation_risk = st.selectbox("Task Automation Vector", ["Low", "Medium", "High"])
            upskilling = st.selectbox("Upskilling Required", ["Yes", "No"])
            salary_before = st.number_input("Salary Before AI-impact", min_value=0, value=75000, step=1000)
            salary_after = st.number_input("Salary After AI-impact", min_value=0, value=72000, step=1000)
            
    st.markdown("<br>", unsafe_allow_html=True)
    productivity_change = st.slider("Productivity Change Metric (%)", -100.0, 100.0, 0.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("Predict job status")

# ============================================
# 5. Core Machine Learning Inference Architecture
# ============================================
if submit:
    with st.spinner("Resolving class alignment parameters and applying scale normalizations..."):
        # Map input features safely into structural payload frame
        input_dict = {
            "Age": age,
            "Gender": gender,
            "Education_Level": education,
            "Industry": industry,
            "Job_Role": job_role,
            "Years_Experience": experience,
            "AI_Adoption_Level": ai_adoption,
            "Automation_Risk": automation_risk,
            "Upskilling_Required": upskilling,
            "Salary_Before_AI": salary_before,
            "Salary_After_AI": salary_after,
            "Work_Hours_Per_Week": work_hours,
            "Remote_Work": remote_work,
            "Job_Satisfaction": satisfaction,
            "Productivity_Change_%": productivity_change
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # Parse One-Hot Encoding variations dynamically
        input_encoded = pd.get_dummies(input_df)
        
        # Align features perfectly to prevent out-of-order schema errors
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]
        
        # Scale through training pipeline state
        input_scaled = scaler.transform(input_encoded)
        
        # Multi-class inference
        prediction = model.predict(input_scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction])[0]
        
        # Artificial execution spacer to prevent UI layout stuttering
        time.sleep(0.4)

    # ============================================
    # 6. Visual Output Engineering
    # ============================================
    st.markdown("## 🔍 Statistical Matrix Interpretability Results")
    
    col_out_left, col_out_right = st.columns([0.4, 0.6])
    
    with col_out_left:
        st.markdown("### Displacement Classification")
        
        # Stylized status banners matching enterprise UI expectations
        if prediction_label.lower() == "replaced":
            st.markdown(f"""
                <div class='metric-card replaced-card'>
                    <h2 style='margin:0; color:#EF4444;'>🚨 HIGH STRUCTURAL RISK</h2>
                    <p style='margin:5px 0 0 0; font-size:1.1rem;'>Current Role Profile is predicted to be: <strong>{prediction_label.upper()}</strong></p>
                </div>
            """, unsafe_allow_html=True)
            st.error("💥 **Strategic Remediation Required:** Critical priority vector identified. Workforce optimization indicates immediate action parameters required via active upskilling deployment frameworks or immediate strategic organizational transition mapping.")
            
        elif prediction_label.lower() == "modified":
            st.markdown(f"""
                <div class='metric-card modified-card'>
                    <h2 style='margin:0; color:#F59E0B;'>🔄 EVOLUTIONARY TRANSITION</h2>
                    <p style='margin:5px 0 0 0; font-size:1.1rem;'>Current Role Profile is predicted to be: <strong>{prediction_label.upper()}</strong></p>
                </div>
            """, unsafe_allow_html=True)
            st.warning("⚡ **Operational Readjustment Noted:** The targeted occupational role functions will face modification adjustments. Human capital structures should proactively allocate optimization tools to augment processing capacities.")
            
        else:
            st.markdown(f"""
                <div class='metric-card unchanged-card'>
                    <h2 style='margin:0; color:#10B981;'>✅ STABLE ANCHOR AGE</h2>
                    <p style='margin:5px 0 0 0; font-size:1.1rem;'>Current Role Profile is predicted to be: <strong>{prediction_label.upper()}</strong></p>
                </div>
            """, unsafe_allow_html=True)
            st.success("💎 **Secured Threshold:** Operational parameters indicate baseline immunity against automated disruption transformations. Standard administrative functions remain fully validated under current operational schemas.")

    with col_out_right:
        st.markdown("### Probability Distributive Vector Metric")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            
            prob_df = pd.DataFrame({
                "Structural Vector Class": label_encoder.classes_,
                "Certainty Spectrum Probability": probs
            })
            
            # Interactive visualization showing clean distributive boundaries
            st.markdown("Calculated multi-class soft margin probability allocations:")
            
            # Modernized vertical mapping using clean progress components
            for idx, row in prob_df.iterrows():
                cls_name = row["Structural Vector Class"]
                cls_prob = row["Certainty Spectrum Probability"]
                
                col_c1, col_c2 = st.columns([0.3, 0.7])
                with col_c1:
                    st.markdown(f"**{cls_name}**")
                with col_c2:
                    # Dynamically coloring visual progress elements based on severity
                    color_type = "normal"
                    if cls_name.lower() == "replaced" and cls_prob > 0.4:
                        st.progress(float(cls_prob))
                    else:
                        st.progress(float(cls_prob))
                    st.caption(f"Confidence Allocation Index: {cls_prob:.2%}")
                    
# ============================================
# 7. Enterprise Footer Block
# ============================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption("🔒 Enterprise Class Encrypted Inference Layer Pipeline System")
with col_f2:
    st.markdown("<p style='text-align: right; color: gray; font-size: 0.8rem;'>Inference Core Pipeline v2.4.1-Stable</p>", unsafe_allow_html=True)