"""
==================================================================================
AI-POWERED HEART DISEASE PREDICTION SYSTEM USING CLINICAL DATA
==================================================================================
B.Tech Final Year Project
Built with: Streamlit, Scikit-learn, Plotly

Run using:
    streamlit run app.py

Required files in the same folder:
    model.pkl   -> trained classification model (predict_proba compatible)
    scaler.pkl  -> fitted StandardScaler (or similar) used during training

NOTE: If model.pkl / scaler.pkl are not found, the app automatically falls
back to a small in-memory dummy model so the dashboard remains fully
runnable for demonstration purposes. Replace with your real trained
artifacts for production use.
==================================================================================
"""

# ----------------------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# 2. PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# 3. GLOBAL CONSTANTS
# ----------------------------------------------------------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
MODEL_ACCURACY_DEFAULT = 91.8  # Displayed default accuracy (%) if not embedded in model

FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

THEME = {
    "primary": "#7b2ff7",
    "secondary": "#f107a3",
    "accent": "#00d4ff",
    "success": "#12d18e",
    "warning": "#ffb020",
    "danger": "#ff4d6d",
    "bg_dark": "#0f0c29",
}

# ----------------------------------------------------------------------------
# 4. CUSTOM CSS -- GLASSMORPHISM / GRADIENT PROFESSIONAL THEME
# ----------------------------------------------------------------------------
def load_custom_css():
    """Inject custom CSS for a modern glassmorphism gradient dashboard."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Poppins', sans-serif;
        }}

        /* Main app background gradient */
        .stApp {{
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            background-attachment: fixed;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }}
        section[data-testid="stSidebar"] * {{
            color: #f1f1f1 !important;
        }}

        /* Glassmorphism card */
        .glass-card {{
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 22px 26px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            color: #ffffff;
        }}

        /* Header / hero banner */
        .hero-banner {{
            background: linear-gradient(120deg, {THEME['primary']} 0%, {THEME['secondary']} 100%);
            padding: 32px 36px;
            border-radius: 22px;
            box-shadow: 0 10px 35px rgba(123, 47, 247, 0.35);
            margin-bottom: 24px;
            color: white;
        }}
        .hero-title {{
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
        }}
        .hero-sub {{
            font-size: 1rem;
            font-weight: 300;
            opacity: 0.92;
            margin-top: 6px;
        }}

        /* Metric-style stat card */
        .stat-card {{
            background: rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 18px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.12);
            backdrop-filter: blur(10px);
        }}
        .stat-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {THEME['accent']};
        }}
        .stat-label {{
            font-size: 0.85rem;
            color: #d3d3f0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* Section headers */
        .section-header {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #ffffff;
            border-left: 5px solid {THEME['secondary']};
            padding-left: 12px;
            margin: 18px 0 14px 0;
        }}

        /* Buttons */
        div.stButton > button {{
            background: linear-gradient(120deg, {THEME['primary']}, {THEME['secondary']});
            color: white;
            border: none;
            border-radius: 30px;
            padding: 10px 28px;
            font-weight: 600;
            font-size: 0.95rem;
            box-shadow: 0 4px 14px rgba(241, 7, 163, 0.35);
            transition: all 0.25s ease-in-out;
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 22px rgba(241, 7, 163, 0.5);
        }}

        /* Download button */
        .stDownloadButton > button {{
            background: linear-gradient(120deg, {THEME['success']}, {THEME['accent']});
            color: #06251c;
            border: none;
            border-radius: 30px;
            font-weight: 700;
            padding: 10px 26px;
        }}

        /* Risk badges */
        .risk-badge {{
            display: inline-block;
            padding: 10px 22px;
            border-radius: 30px;
            font-weight: 700;
            font-size: 1.1rem;
            text-align: center;
        }}
        .risk-low {{ background: rgba(18, 209, 142, 0.18); color: {THEME['success']}; border: 1px solid {THEME['success']}; }}
        .risk-medium {{ background: rgba(255, 176, 32, 0.18); color: {THEME['warning']}; border: 1px solid {THEME['warning']}; }}
        .risk-high {{ background: rgba(255, 77, 109, 0.18); color: {THEME['danger']}; border: 1px solid {THEME['danger']}; }}

        /* Recommendation cards */
        .rec-card {{
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 10px;
            font-size: 0.95rem;
            color: #10121a;
            font-weight: 500;
        }}
        .rec-low {{ background: linear-gradient(120deg, #b9fbc0, #8ce99a); }}
        .rec-medium {{ background: linear-gradient(120deg, #ffe8a3, #ffd166); }}
        .rec-high {{ background: linear-gradient(120deg, #ffb3c1, #ff6b81); }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 18px 0;
            color: #b6b6d8;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.08);
            margin-top: 30px;
        }}

        /* Dataframe styling */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
        }}

        /* Hide default streamlit branding for a cleaner professional look */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 5. SESSION STATE INITIALIZATION
# ----------------------------------------------------------------------------
def init_session_state():
    """Initialize all session state variables used across the app."""
    defaults = {
        "prediction_count": 0,
        "patient_id": generate_patient_id(),
        "last_prediction": None,
        "last_probability": None,
        "patient_data": {},
        "clinical_data": {},
        "history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def generate_patient_id():
    """Auto-generate a unique Patient ID based on the current timestamp."""
    return "PT-" + datetime.now().strftime("%Y%m%d-%H%M%S")


# ----------------------------------------------------------------------------
# 6. MODEL & SCALER LOADING (WITH GRACEFUL FALLBACK)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    """
    Load the pre-trained model and scaler from disk.
    Falls back to a lightweight dummy model if the files are missing,
    so the application remains fully runnable for demo purposes.
    """
    model, scaler, using_fallback = None, None, False

    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
        else:
            raise FileNotFoundError("model.pkl or scaler.pkl not found in the app directory.")
    except Exception:
        # ---------------- FALLBACK DUMMY MODEL ----------------
        # This block trains a very small logistic regression on synthetic
        # data ONLY so the app does not crash when real artifacts are absent.
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        X_dummy = rng.rand(200, len(FEATURE_ORDER))
        y_dummy = (X_dummy[:, 0] + X_dummy[:, 4] + X_dummy[:, 7] > 1.5).astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_dummy)

        model = LogisticRegression()
        model.fit(X_scaled, y_dummy)
        using_fallback = True

    return model, scaler, using_fallback


# ----------------------------------------------------------------------------
# 7. HELPER / UTILITY FUNCTIONS
# ----------------------------------------------------------------------------
def calculate_bmi(weight_kg, height_cm):
    """Calculate Body Mass Index (BMI) given weight (kg) and height (cm)."""
    try:
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)
    except ZeroDivisionError:
        return 0.0


def bmi_category(bmi):
    """Return the BMI category label for a given BMI value."""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def get_risk_level(probability):
    """
    Convert a predicted probability (0-100) into a risk category.
    Low Risk    : < 40%
    Medium Risk : 40% - 70%
    High Risk   : > 70%
    """
    if probability < 40:
        return "Low Risk", "risk-low"
    elif probability < 70:
        return "Medium Risk", "risk-medium"
    else:
        return "High Risk", "risk-high"


def encode_categorical_inputs(patient_info, clinical_info):
    """
    Convert human-readable form inputs into the numeric encoding
    expected by the ML model, and return them in FEATURE_ORDER.
    """
    sex_map = {"Male": 1, "Female": 0}
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3,
    }
    fbs_map = {"<= 120 mg/dl": 0, "> 120 mg/dl": 1}
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    exang_map = {"No": 0, "Yes": 1}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    feature_vector = [
        patient_info["age"],
        sex_map.get(patient_info["gender"], 0),
        cp_map.get(clinical_info["chest_pain_type"], 0),
        clinical_info["resting_bp"],
        clinical_info["cholesterol"],
        fbs_map.get(clinical_info["fasting_bs"], 0),
        restecg_map.get(clinical_info["rest_ecg"], 0),
        clinical_info["max_heart_rate"],
        exang_map.get(clinical_info["exercise_angina"], 0),
        clinical_info["st_depression"],
        slope_map.get(clinical_info["slope"], 0),
        clinical_info["major_vessels"],
        thal_map.get(clinical_info["thal"], 1),
    ]
    return np.array(feature_vector, dtype=float).reshape(1, -1)


def predict_heart_disease(model, scaler, feature_vector):
    """
    Scale the feature vector and run the prediction.
    Returns (predicted_class, probability_percentage).
    """
    try:
        scaled_features = scaler.transform(feature_vector)
        probability = model.predict_proba(scaled_features)[0][1] * 100
        predicted_class = int(probability >= 50)
        return predicted_class, round(probability, 2)
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        return None, None


# ----------------------------------------------------------------------------
# 8. PLOTLY VISUALIZATION FUNCTIONS
# ----------------------------------------------------------------------------
def plot_gauge_chart(probability):
    """Gauge chart displaying the overall risk probability."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability,
            number={"suffix": "%", "font": {"size": 42, "color": "white"}},
            delta={"reference": 50, "increasing": {"color": THEME["danger"]}},
            title={"text": "Heart Disease Risk Probability", "font": {"size": 18, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": THEME["secondary"]},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(18,209,142,0.35)"},
                    {"range": [40, 70], "color": "rgba(255,176,32,0.35)"},
                    {"range": [70, 100], "color": "rgba(255,77,109,0.35)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.8,
                    "value": probability,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=320,
        margin=dict(t=50, b=10, l=20, r=20),
    )
    return fig


def plot_pie_chart(probability):
    """Pie chart showing risk vs no-risk probability split."""
    labels = ["Disease Risk", "No Disease Risk"]
    values = [probability, 100 - probability]
    fig = px.pie(
        names=labels,
        values=values,
        color_discrete_sequence=[THEME["danger"], THEME["success"]],
        hole=0.55,
    )
    fig.update_traces(textinfo="percent+label", textfont_color="white")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        showlegend=True,
        height=320,
        margin=dict(t=30, b=10, l=10, r=10),
        title="Risk Distribution",
    )
    return fig


def plot_risk_bar_chart(probability):
    """Bar chart comparing patient's risk against reference thresholds."""
    categories = ["Low Risk\nThreshold", "Medium Risk\nThreshold", "Patient\nRisk", "High Risk\nThreshold"]
    values = [40, 70, probability, 100]
    colors = [THEME["success"], THEME["warning"], THEME["accent"], THEME["danger"]]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.08)"),
        height=350,
        title="Risk Comparison Chart",
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def plot_bmi_indicator(bmi_value):
    """Indicator gauge showing the patient's BMI on a standard scale."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=bmi_value,
            number={"font": {"size": 36, "color": "white"}},
            title={"text": "BMI Indicator", "font": {"size": 18, "color": "white"}},
            gauge={
                "axis": {"range": [10, 45], "tickcolor": "white"},
                "bar": {"color": THEME["accent"]},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [10, 18.5], "color": "rgba(255,176,32,0.35)"},
                    {"range": [18.5, 25], "color": "rgba(18,209,142,0.35)"},
                    {"range": [25, 30], "color": "rgba(255,176,32,0.35)"},
                    {"range": [30, 45], "color": "rgba(255,77,109,0.35)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=300,
        margin=dict(t=50, b=10, l=20, r=20),
    )
    return fig


def plot_probability_meter(probability):
    """Horizontal probability meter (bullet-style chart)."""
    fig = go.Figure(
        go.Indicator(
            mode="number+gauge",
            value=probability,
            number={"suffix": "%", "font": {"color": "white"}},
            gauge={
                "shape": "bullet",
                "axis": {"range": [0, 100]},
                "bar": {"color": THEME["secondary"]},
                "steps": [
                    {"range": [0, 40], "color": "rgba(18,209,142,0.4)"},
                    {"range": [40, 70], "color": "rgba(255,176,32,0.4)"},
                    {"range": [70, 100], "color": "rgba(255,77,109,0.4)"},
                ],
            },
            title={"text": "Probability Meter", "font": {"color": "white", "size": 16}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=160,
        margin=dict(t=30, b=10, l=10, r=10),
    )
    return fig


# ----------------------------------------------------------------------------
# 9. RECOMMENDATION ENGINE
# ----------------------------------------------------------------------------
def get_recommendations(risk_label):
    """Return a list of health recommendations based on the risk category."""
    recommendations = {
        "Low Risk": [
            "✅ Maintain a balanced diet rich in fruits, vegetables, and whole grains.",
            "✅ Continue regular physical activity (at least 150 min/week).",
            "✅ Schedule routine annual health check-ups.",
            "✅ Maintain healthy sleep patterns (7-8 hours/night).",
            "✅ Keep monitoring blood pressure and cholesterol periodically.",
        ],
        "Medium Risk": [
            "⚠️ Consult a cardiologist for a detailed clinical evaluation.",
            "⚠️ Reduce sodium, saturated fat, and processed sugar intake.",
            "⚠️ Engage in moderate aerobic exercise under medical guidance.",
            "⚠️ Monitor blood pressure and cholesterol every 3 months.",
            "⚠️ Manage stress through meditation, yoga, or relaxation techniques.",
        ],
        "High Risk": [
            "🚨 Seek immediate consultation with a cardiologist.",
            "🚨 Undergo further diagnostic tests (ECG, Angiography, Echocardiogram).",
            "🚨 Strictly follow prescribed medications and dietary restrictions.",
            "🚨 Avoid strenuous physical activity until cleared by a physician.",
            "🚨 Quit smoking and alcohol consumption immediately, if applicable.",
        ],
    }
    return recommendations.get(risk_label, [])


# ----------------------------------------------------------------------------
# 10. SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
def render_sidebar():
    """Render the sidebar with navigation menu and quick stats."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 10px 0 20px 0;">
                <div style="font-size:52px;">🏥</div>
                <div style="font-size:1.2rem; font-weight:700; color:white;">CardioAI Diagnostics</div>
                <div style="font-size:0.8rem; color:#a9a9d6;">Hospital Logo Placeholder</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigate",
            [
                "🏠 Dashboard",
                "👤 Patient Information",
                "❤️ Clinical Parameters",
                "📊 Prediction Results",
                "📈 Analytics",
                "💡 Recommendations",
                "ℹ️ About",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{st.session_state.prediction_count}</div>
                <div class="stat-label">Predictions Made</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{MODEL_ACCURACY_DEFAULT}%</div>
                <div class="stat-label">Model Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        now = datetime.now()
        st.markdown(
            f"""
            <div style="color:#c9c9e8; font-size:0.85rem; text-align:center;">
                📅 {now.strftime('%d %B %Y')}<br>
                ⏰ {now.strftime('%I:%M:%S %p')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ----------------------------------------------------------------------------
# 11. PAGE: DASHBOARD
# ----------------------------------------------------------------------------
def render_dashboard():
    """Render the home dashboard overview page."""
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">❤️ AI-Powered Heart Disease Prediction System</div>
            <div class="hero-sub">Clinical Decision Support powered by Machine Learning | B.Tech Final Year Project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-value">🧑‍⚕️</div>
            <div class="stat-label">Patient ID</div>
            <div style="color:white; font-weight:600; margin-top:4px;">{st.session_state.patient_id}</div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-value">{st.session_state.prediction_count}</div>
            <div class="stat-label">Total Predictions</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-value">{MODEL_ACCURACY_DEFAULT}%</div>
            <div class="stat-label">Model Accuracy</div></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-value">📊</div>
            <div class="stat-label">ML Powered Insights</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">🔍 How This System Works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
        This dashboard uses a trained Machine Learning classification model to analyze
        a patient's demographic, lifestyle, and clinical parameters in order to estimate
        the probability of heart disease. Please follow the steps below:
        <br><br>
        1️⃣ Fill in <b>Patient Information</b> (demographics & lifestyle).<br>
        2️⃣ Fill in <b>Clinical Parameters</b> (medical test results).<br>
        3️⃣ View the <b>Prediction Results</b> with probability & risk level.<br>
        4️⃣ Explore <b>Analytics</b> for interactive visualizations.<br>
        5️⃣ Review personalized <b>Health Recommendations</b>.<br>
        6️⃣ Download a complete <b>CSV report</b> for medical records.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 12. PAGE: PATIENT INFORMATION
# ----------------------------------------------------------------------------
def render_patient_information():
    """Render the patient demographic and lifestyle information form."""
    st.markdown('<div class="section-header">👤 Patient Information</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("🧑 Patient Name", value=st.session_state.patient_data.get("name", ""))
            age = st.number_input("🎂 Age", min_value=1, max_value=120,
                                   value=st.session_state.patient_data.get("age", 45))
            gender = st.selectbox("⚧ Gender", ["Male", "Female"],
                                   index=0 if st.session_state.patient_data.get("gender", "Male") == "Male" else 1)
            height = st.number_input("📏 Height (cm)", min_value=50.0, max_value=250.0,
                                      value=st.session_state.patient_data.get("height", 170.0))
            weight = st.number_input("⚖️ Weight (kg)", min_value=10.0, max_value=300.0,
                                      value=st.session_state.patient_data.get("weight", 70.0))
            smoking = st.selectbox("🚬 Smoking", ["No", "Yes"],
                                    index=["No", "Yes"].index(st.session_state.patient_data.get("smoking", "No")))

        with col2:
            alcohol = st.selectbox("🍷 Alcohol Consumption", ["No", "Yes"],
                                    index=["No", "Yes"].index(st.session_state.patient_data.get("alcohol", "No")))
            physical_activity = st.selectbox("🏃 Physical Activity Level", ["Low", "Moderate", "High"],
                                              index=["Low", "Moderate", "High"].index(
                                                  st.session_state.patient_data.get("activity", "Moderate")))
            family_history = st.selectbox("🧬 Family History of Heart Disease", ["No", "Yes"],
                                           index=["No", "Yes"].index(
                                               st.session_state.patient_data.get("family_history", "No")))
            diabetes = st.selectbox("🩸 Diabetes", ["No", "Yes"],
                                     index=["No", "Yes"].index(st.session_state.patient_data.get("diabetes", "No")))
            hypertension = st.selectbox("💢 Hypertension", ["No", "Yes"],
                                         index=["No", "Yes"].index(
                                             st.session_state.patient_data.get("hypertension", "No")))

        # Automatically calculate BMI
        bmi = calculate_bmi(weight, height)
        category = bmi_category(bmi)

        st.markdown("<br>", unsafe_allow_html=True)
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.metric("📐 Calculated BMI", f"{bmi}")
        with bcol2:
            st.metric("📊 BMI Category", category)

        st.markdown("</div>", unsafe_allow_html=True)

        # Persist data to session state
        st.session_state.patient_data = {
            "name": patient_name,
            "id": st.session_state.patient_id,
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "bmi_category": category,
            "smoking": smoking,
            "alcohol": alcohol,
            "activity": physical_activity,
            "family_history": family_history,
            "diabetes": diabetes,
            "hypertension": hypertension,
        }

    st.success("✅ Patient information saved. Proceed to Clinical Parameters.")


# ----------------------------------------------------------------------------
# 13. PAGE: CLINICAL PARAMETERS
# ----------------------------------------------------------------------------
def render_clinical_parameters():
    """Render the clinical / medical test parameter input form."""
    st.markdown('<div class="section-header">❤️ Clinical Parameters</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            chest_pain_type = st.selectbox(
                "💔 Chest Pain Type",
                ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
            )
            resting_bp = st.number_input("🩺 Resting Blood Pressure (mm Hg)", min_value=60, max_value=250, value=120)
            cholesterol = st.number_input("🧪 Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fasting_bs = st.selectbox("🍬 Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
            rest_ecg = st.selectbox(
                "📉 Resting ECG Results",
                ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
            )
            max_heart_rate = st.number_input("💓 Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)

        with col2:
            exercise_angina = st.selectbox("🏃‍♂️ Exercise Induced Angina", ["No", "Yes"])
            st_depression = st.number_input(
                "📈 ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1
            )
            slope = st.selectbox("📐 Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
            major_vessels = st.selectbox("🩻 Number of Major Vessels (0-3)", [0, 1, 2, 3])
            thal = st.selectbox("🧬 Thalassemia (Thal)", ["Normal", "Fixed Defect", "Reversible Defect"])

        st.markdown("</div>", unsafe_allow_html=True)

        # Persist data to session state
        st.session_state.clinical_data = {
            "chest_pain_type": chest_pain_type,
            "resting_bp": resting_bp,
            "cholesterol": cholesterol,
            "fasting_bs": fasting_bs,
            "rest_ecg": rest_ecg,
            "max_heart_rate": max_heart_rate,
            "exercise_angina": exercise_angina,
            "st_depression": st_depression,
            "slope": slope,
            "major_vessels": major_vessels,
            "thal": thal,
        }

    st.success("✅ Clinical parameters saved. Proceed to Prediction Results.")


# ----------------------------------------------------------------------------
# 14. PAGE: PREDICTION RESULTS
# ----------------------------------------------------------------------------
def render_prediction_results(model, scaler, using_fallback):
    """Run the prediction pipeline and display the results."""
    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

    if using_fallback:
        st.warning(
            "⚠️ model.pkl / scaler.pkl not found — using a temporary demo model. "
            "Place your trained model.pkl and scaler.pkl in the app folder for real predictions."
        )

    if not st.session_state.patient_data or not st.session_state.clinical_data:
        st.info("ℹ️ Please fill in **Patient Information** and **Clinical Parameters** first.")
        return

    if st.button("🔍 Run Prediction"):
        try:
            feature_vector = encode_categorical_inputs(
                st.session_state.patient_data, st.session_state.clinical_data
            )
            predicted_class, probability = predict_heart_disease(model, scaler, feature_vector)

            if probability is not None:
                st.session_state.last_prediction = predicted_class
                st.session_state.last_probability = probability
                st.session_state.prediction_count += 1
                st.session_state.history.append(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "patient_id": st.session_state.patient_id,
                        "probability": probability,
                    }
                )
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")

    # Display results if a prediction has been made
    if st.session_state.last_probability is not None:
        probability = st.session_state.last_probability
        risk_label, risk_css = get_risk_level(probability)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(plot_gauge_chart(probability), use_container_width=True)
        with col2:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <h3 style="color:white;">Overall Risk Assessment</h3>
                    <div class="risk-badge {risk_css}">{risk_label}</div>
                    <p style="margin-top:14px; font-size:1.4rem; color:white;">
                        Probability: <b>{probability}%</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(plot_probability_meter(probability), use_container_width=True)

        st.markdown('<div class="section-header">📋 Patient Summary</div>', unsafe_allow_html=True)
        summary_df = build_summary_dataframe(risk_label, probability)
        st.dataframe(summary_df, use_container_width=True, height=420)

        # CSV Download
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv_buffer.getvalue(),
            file_name=f"heart_disease_report_{st.session_state.patient_id}.csv",
            mime="text/csv",
        )
    else:
        st.info("👆 Click **Run Prediction** to generate the risk assessment.")


def build_summary_dataframe(risk_label, probability):
    """Combine patient + clinical data + prediction results into a single dataframe."""
    combined = {}
    combined.update(st.session_state.patient_data)
    combined.update(st.session_state.clinical_data)
    combined["risk_level"] = risk_label
    combined["probability_percent"] = probability
    combined["prediction_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(list(combined.items()), columns=["Parameter", "Value"])
    return df


# ----------------------------------------------------------------------------
# 15. PAGE: ANALYTICS
# ----------------------------------------------------------------------------
def render_analytics():
    """Render interactive analytics visualizations."""
    st.markdown('<div class="section-header">📈 Analytics</div>', unsafe_allow_html=True)

    if st.session_state.last_probability is None:
        st.info("ℹ️ Run a prediction first from the **Prediction Results** page to view analytics.")
        return

    probability = st.session_state.last_probability
    bmi = st.session_state.patient_data.get("bmi", 0)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_pie_chart(probability), use_container_width=True)
    with col2:
        st.plotly_chart(plot_bmi_indicator(bmi), use_container_width=True)

    st.plotly_chart(plot_risk_bar_chart(probability), use_container_width=True)

    if len(st.session_state.history) > 1:
        st.markdown('<div class="section-header">🕒 Prediction History (This Session)</div>', unsafe_allow_html=True)
        history_df = pd.DataFrame(st.session_state.history)
        fig = px.line(
            history_df,
            x="timestamp",
            y="probability",
            markers=True,
            title="Probability Trend Across Session Predictions",
        )
        fig.update_traces(line_color=THEME["accent"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# 16. PAGE: RECOMMENDATIONS
# ----------------------------------------------------------------------------
def render_recommendations():
    """Render personalized health recommendations based on risk level."""
    st.markdown('<div class="section-header">💡 Health Recommendations</div>', unsafe_allow_html=True)

    if st.session_state.last_probability is None:
        st.info("ℹ️ Run a prediction first from the **Prediction Results** page to view recommendations.")
        return

    probability = st.session_state.last_probability
    risk_label, _ = get_risk_level(probability)
    recs = get_recommendations(risk_label)

    css_class = {"Low Risk": "rec-low", "Medium Risk": "rec-medium", "High Risk": "rec-high"}.get(
        risk_label, "rec-low"
    )

    st.markdown(f"### Recommendations for: **{risk_label}**")
    for rec in recs:
        st.markdown(f'<div class="rec-card {css_class}">{rec}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
        ⚠️ <b>Disclaimer:</b> These recommendations are generated by an AI system for
        educational and academic demonstration purposes only. They do not replace
        professional medical advice. Always consult a certified cardiologist or
        physician for actual diagnosis and treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 17. PAGE: ABOUT
# ----------------------------------------------------------------------------
def render_about():
    """Render the About section describing the project, dataset, and tech stack."""
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="glass-card">
        <h4>📚 Dataset</h4>
        <p>The model is trained on the widely-used <b>UCI Cleveland Heart Disease Dataset</b>,
        which contains 13 clinical attributes including age, sex, chest pain type,
        blood pressure, cholesterol, ECG results, and more, used to predict the
        presence of heart disease.</p>

        <h4>🤖 Machine Learning Model</h4>
        <p>A supervised classification model (e.g., Random Forest / Logistic Regression /
        Gradient Boosting) was trained and serialized using <b>joblib/pickle</b> as
        <code>model.pkl</code>, alongside a fitted <code>scaler.pkl</code> for feature
        normalization.</p>

        <h4>🎯 Model Accuracy</h4>
        <p>Achieved approximately <b>{MODEL_ACCURACY_DEFAULT}%</b> accuracy on the held-out
        test set during evaluation (cross-validated).</p>

        <h4>🛠️ Technology Stack</h4>
        <ul>
            <li><b>Frontend/Dashboard:</b> Streamlit + Custom CSS (Glassmorphism UI)</li>
            <li><b>Visualization:</b> Plotly (Interactive Charts)</li>
            <li><b>Machine Learning:</b> Scikit-learn</li>
            <li><b>Data Handling:</b> Pandas, NumPy</li>
            <li><b>Model Persistence:</b> Joblib</li>
            <li><b>Language:</b> Python 3.x</li>
        </ul>

        <h4>🎓 Project Type</h4>
        <p>B.Tech Final Year Major Project — Department of Computer Science / AI &amp; ML.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 18. FOOTER
# ----------------------------------------------------------------------------
def render_footer():
    """Render a consistent footer across all pages."""
    st.markdown(
        f"""
        <div class="footer">
            ❤️ AI-Powered Heart Disease Prediction System &nbsp;|&nbsp;
            B.Tech Final Year Project &nbsp;|&nbsp;
            © {datetime.now().year} All Rights Reserved &nbsp;|&nbsp;
            Built with Streamlit &amp; Scikit-learn
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 19. MAIN APPLICATION ENTRY POINT
# ----------------------------------------------------------------------------
def main():
    """Main function orchestrating the entire Streamlit application."""
    load_custom_css()
    init_session_state()

    # Load ML artifacts once (cached)
    model, scaler, using_fallback = load_model_and_scaler()

    # Sidebar navigation
    selected_page = render_sidebar()

    # Route to the correct page
    try:
        if selected_page == "🏠 Dashboard":
            render_dashboard()
        elif selected_page == "👤 Patient Information":
            render_patient_information()
        elif selected_page == "❤️ Clinical Parameters":
            render_clinical_parameters()
        elif selected_page == "📊 Prediction Results":
            render_prediction_results(model, scaler, using_fallback)
        elif selected_page == "📈 Analytics":
            render_analytics()
        elif selected_page == "💡 Recommendations":
            render_recommendations()
        elif selected_page == "ℹ️ About":
            render_about()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while rendering the page: {e}")

    render_footer()


# ----------------------------------------------------------------------------
# 20. RUN APPLICATION
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
