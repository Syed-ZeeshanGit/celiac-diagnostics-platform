# app.py
import os
import sys

# Standardize path routing before executing package imports
root_path = os.path.abspath(os.path.dirname(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import streamlit as st
import pandas as pd
import numpy as np         # Added missing import required for Tab 2 matrix execution
import plotly.express as px
from PIL import Image

# Core architectural modules - Fixed the duplicate class name typo on the last line here
from src.data_processing import MedicalDataPipeline
from src.model_engine import VectorizedLogisticRegression, MedicalImageMatrixEngine
from src.clinical_flow import ClinicalValidationEngine, DiagnosticAuditJournal

# Force page configuration setup
st.set_page_config(page_title="Clinical Diagnostics & Analytics Platform", layout="wide")

# Instantiation of persistent tracking stores across page rebuilds
if "audit_logger" not in st.session_state:
    st.session_state.audit_logger = DiagnosticAuditJournal()

st.title("🔬 Clinical Diagnostics & Analytics Platform")
st.markdown("An end-to-end medical software system optimizing **Celiac Disease prediction pipelines**.")
st.write("---")

# Initialize and fit model instantly on application boot sequence
@st.cache_resource
def execute_system_training():
    pipeline = MedicalDataPipeline("data/celiac_disease.csv")
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess()
    model = VectorizedLogisticRegression(learning_rate=0.1, epochs=1000)
    costs = model.fit(X_train, y_train)
    
    # Calculate performance matrices
    preds = model.predict(X_test)
    metrics = ClinicalValidationEngine.compute_evaluation_metrics(y_test, preds)
    return pipeline, model, metrics, costs

pipeline, model, metrics, costs = execute_system_training()

# Define layout presentation elements
tab_is, tab_cs, tab_se = st.tabs([
    "📊 Patient Triage & Institutional Analytics",
    "🧮 Algorithmic Core & Matrix Core",
    "💻 System Pipeline & Validation Blueprint"
])

# =========================================================================
# TAB 1: INFORMATION SYSTEMS (Clinical Workflows & Real-time Triage)
# =========================================================================
with tab_is:
    st.header("Patient Triage Engine & Operational Auditing")
    
    col_input, col_diag = st.columns([2, 2])
    
    with col_input:
        st.subheader("📋 Ingest Current Patient Profile")
        age = st.slider("Patient Age", 1, 90, 25)
        gender = st.selectbox("Biological Gender", ["Male", "Female"])
        diabetes = st.selectbox("Comorbid Diabetes Diagnosis", ["no", "Yes"])
        db_type = st.selectbox("Diabetes Variant Type", ["None", "Type 1", "Type 2"])
        diarrhoea = st.selectbox("Primary Diarrhoea Symptom Manifestation", ["inflammatory", "fatty", "watery"])
        abdominal = st.radio("Experiencing Abdominal Distress Pain", ["no", "yes"])
        short_stature = st.selectbox("Idiopathic Short Stature Category", ["PSS", "Variant", "DSS"])
        sticky_stool = st.radio("Sticky Stool Manifestation Signs", ["no", "yes"])
        weight_loss = st.radio("Unexplained Rapid Weight Loss Signs", ["no", "yes"])
        
        st.markdown("**Serology Quantitative Blood Biomarkers**")
        iga = st.slider("Serum Total IgA Level (g/L)", 0.1, 10.0, 1.5)
        igg = st.slider("Serum Anti-tTG IgG Antibody (U/mL)", 4.0, 16.0, 9.5)
        igm = st.slider("Serum IgM Antibody Parameter (g/L)", 0.1, 3.0, 1.1)
        
        if st.button("🚀 Process Clinical Record Verification"):
            input_dict = {
                "Age": age, "Gender": gender, "Diabetes": diabetes, "Diabetes Type": db_type,
                "Diarrhoea": diarrhoea, "Abdominal": abdominal, "Short_Stature": short_stature,
                "Sticky_Stool": sticky_stool, "Weight_loss": weight_loss, "IgA": iga, "IgG": igg, "IgM": igm
            }
            
            # Map features, pass vector through pure NumPy equations
            processed_vector = pipeline.transform_single_input(input_dict)
            risk_probability = float(model.predict_proba(processed_vector)[0])
            verdict = "Positive" if risk_probability >= 0.5 else "Negative"
            
            st.session_state.last_prob = risk_probability
            st.session_state.last_verdict = verdict
            st.session_state.audit_logger.sign_record(age, gender, risk_probability, verdict)

    with col_diag:
        st.subheader("🩺 Diagnostic System Verdict Output")
        if "last_verdict" in st.session_state:
            prob = st.session_state.last_prob
            verd = st.session_state.last_verdict
            
            if verd == "Positive":
                st.error(f"🚨 HIGH RISK PATHWAY IDENTIFIED: Celiac Target Matches (Probability: {prob*100:.2f}%)")
                st.warning("Clinical Advisory Action: Schedule patient instantly for structural intestinal biopsy extraction confirmation.")
            else:
                st.success(f"✅ BASELINE SAFETY BOUNDS CONFIRMED: Low Risk Target (Probability: {prob*100:.2f}%)")
                st.info("Clinical Advisory Action: Discharge profile back into routine observation monitoring cycles.")
        else:
            st.info("Awaiting clinical profile submission inputs to execute vector evaluation path.")

    st.write("---")
    st.subheader("🗂️ Active Compliance Audit Logging Journal (GDPR/HIPAA Traceable)")
    if st.session_state.audit_logger.journal_records:
        st.table(pd.DataFrame(st.session_state.audit_logger.journal_records))
    else:
        st.caption("No operational access traces logged during current tracking frame session context.")

# =========================================================================
# TAB 2: COMPUTER SCIENCE (Custom Binary Matrix Convolution & Pooling)
# =========================================================================
with tab_cs:
    st.header("Advanced Processing Engine Matrix Computations")
    
    st.subheader("🖼️ Tier 2 Core: High-Performance Image Downsampling Simulation")
    st.markdown("""
    To model tissue imagery diagnostics safely inside standard programming environments, upload any sample 
    file below. The custom backend vectorizes the pixels into a 2D matrix structure and applies structural **2D Max Pooling** transformations via pure nested array loops to extract regional feature bounds.
    """)
    
    img_uploader = st.file_uploader("Upload Intestinal Biopsy Image Scan File (.png, .jpg)", type=["png", "jpg", "jpeg"])
    
    if img_uploader:
        pil_img = Image.open(img_uploader).convert("L")  # Convert input bytes directly to Grayscale
        raw_pixels = np.array(pil_img)
        
        c_img_l, c_img_r = st.columns(2)
        c_img_l.image(pil_img, caption=f"Raw Grayscale Pixel Array Target Structure ({raw_pixels.shape[0]}x{raw_pixels.shape[1]})", use_container_width=True)
        
        # Execute raw matrix operation from scratch
        pooled_pixels = MedicalImageMatrixEngine.compute_max_pooling2d(raw_pixels, stride_dimension=4)
        
        # Normalize back into standard array image display ranges
        display_pooled = ((pooled_pixels - pooled_pixels.min()) / (pooled_pixels.max() - pooled_pixels.min()) * 255).astype(np.uint8)
        c_img_r.image(display_pooled, caption=f"NumPy Compressed Feature Map Grid ({display_pooled.shape[0]}x{display_pooled.shape[1]})", use_container_width=True)

# =========================================================================
# TAB 3: SOFTWARE ENGINEERING (Cost Functions & Structural Metrics Validation)
# =========================================================================
with tab_se:
    st.header("Validation Blueprints & Mathematical Verifications")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("🧮 Model Formulation & Vector Trajectory Loss")
        st.markdown("Optimization is driven by standard Binary Cross Entropy Cost minimize configurations:")
        st.latex(r"J(w) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]")
        
        # Plot cost decay vector sequence
        cost_df = pd.DataFrame({"Optimization Training Epoches": range(len(costs)), "Loss Index Magnitude": costs})
        fig_loss = px.line(cost_df, x="Optimization Training Epoches", y="Loss Index Magnitude", title="Gradient Descent Cost Function Convex Curve Decay Convergence Profile", template="plotly_dark")
        st.plotly_chart(fig_loss, use_container_width=True)
        
    with col_v2:
        st.subheader("🏁 Pipeline Performance Metrics (80/20 Split Assessment)")
        st.markdown("Validation characteristics calculated directly across out-of-sample population segments:")
        
        for k, v in metrics.items():
            st.metric(k, f"{v * 100:.2f}%")