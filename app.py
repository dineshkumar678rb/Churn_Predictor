import json
import os
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# -------------------- BLACK & MUSTARD THEME --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0B0B0B;
    color: #F5F5F5;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #F5F5F5 !important;
    font-weight: 800 !important;
}

p, label, .stMarkdown {
    color: #C9C9C9;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #111111;
}

/* Main cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #151515;
    border: 1px solid #303030;
    border-radius: 18px;
    padding: 18px;
}

/* Form */
[data-testid="stForm"] {
    background: #151515;
    border: 1px solid #303030;
    border-radius: 20px;
    padding: 25px;
}

/* Inputs */
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: #222222 !important;
    color: #FFFFFF !important;
    border-color: #444444 !important;
    border-radius: 10px !important;
}

.stSlider [data-testid="stSliderTrack"] {
    background: #D4A928 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: #D4A928 !important;
    color: #111111 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    padding: 0.65rem 1.5rem !important;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #F0C84B !important;
    transform: translateY(-2px);
}

/* Metrics */
[data-testid="stMetric"] {
    background: #202020;
    border: 1px solid #383838;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #BDBDBD !important;
}

[data-testid="stMetricValue"] {
    color: #E6BE45 !important;
    font-weight: 800;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: #D4A928;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #151515;
    border: 1px solid #333333;
    border-radius: 14px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #333333;
    border-radius: 12px;
    overflow: hidden;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Mustard accent line */
.accent-line {
    height: 4px;
    width: 75px;
    background: #D4A928;
    border-radius: 10px;
    margin: 10px 0 22px 0;
}

/* Header */
.hero {
    background: linear-gradient(120deg, #191919, #101010);
    border: 1px solid #333333;
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero-tag {
    color: #D4A928;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero-title {
    font-size: clamp(28px, 4vw, 42px);
    color: #FFFFFF;
    font-weight: 800;
    margin: 12px 0;
}

.hero-description {
    color: #A9A9A9;
    font-size: 15px;
}

.gold {
    color: #E6BE45;
}

/* Result cards */
.result-card {
    background: linear-gradient(145deg, #1D1D1D, #121212);
    border: 1px solid #393939;
    border-radius: 20px;
    padding: 24px;
    min-height: 170px;
}

.result-label {
    color: #AFAFAF;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-value {
    color: #E6BE45;
    font-size: 34px;
    font-weight: 800;
    margin-top: 15px;
}

.result-description {
    color: #999999;
    font-size: 13px;
    margin-top: 8px;
}

.section-heading {
    font-size: 21px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 5px;
}

/* AI advisor text */
.advisor-box h3 {
    color: #E6BE45 !important;
    font-size: 17px !important;
    margin-top: 14px !important;
}

.advisor-box li, .advisor-box p {
    color: #D0D0D0;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #666666;
    font-size: 12px;
    padding-top: 30px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.joblib")


@st.cache_data
def load_metrics():
    with open("models/metrics.json") as f:
        return json.load(f)


# -------------------- LLM ADVISOR --------------------

def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def ask_advisor(customer, label, confidence):
    client = Groq(api_key=get_api_key())
    prompt = f"""You are a customer retention analyst for a telecom company.
A machine learning model made this prediction. Do not change it, only explain it.

Customer details: {json.dumps(customer)}
Prediction: {label} (confidence {confidence:.0%})

Respond in markdown with exactly these sections:
### Explanation
(2-3 sentences explaining the prediction)
### Key Risk Factors
(3-4 bullets taken from the customer details)
### Possible Reasons for Churn
(2-3 bullets)
### Retention Suggestions
(exactly 3 short, practical actions)

Keep it under 250 words."""
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content


model = load_model()
metrics = load_metrics()
best = metrics["best_model"]


# -------------------- HEADER --------------------

st.markdown(f"""
<div class="hero">
    <div class="hero-tag">CUSTOMER ANALYTICS / AI PREDICTION</div>
    <div class="hero-title">
        Customer <span class="gold">Churn</span> Predictor
    </div>
    <div class="hero-description">
        Understand customer retention. Identify churn risk.
        Make data-driven business decisions.
    </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("MODEL", best)

with m2:
    st.metric(
        "MODEL RECALL",
        f"{metrics['results'][best]['recall']:.0%}"
    )

with m3:
    st.metric("PREDICTION THRESHOLD", "50%")


# -------------------- CUSTOMER INPUT FORM --------------------

st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="section-heading">01 / Customer Profile</div>',
    unsafe_allow_html=True
)

st.caption("Enter customer information to generate a churn prediction.")

with st.form("customer_form"):

    c1, c2, c3 = st.columns(3)

    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 5)
        monthly = st.slider("Monthly Charges", 18.0, 120.0, 70.0)
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with c2:
        internet = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with c3:
        security = st.selectbox("Online Security", ["No", "Yes"])
        techsupport = st.selectbox("Tech Support", ["No", "Yes"])
        backup = st.selectbox("Online Backup", ["No", "Yes"])

    with st.expander("＋ More Customer Details"):

        d1, d2, d3 = st.columns(3)

        with d1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["No", "Yes"])

        with d2:
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["No", "Yes"])

        with d3:
            protection = st.selectbox("Device Protection", ["No", "Yes"])
            stv = st.selectbox("Streaming TV", ["No", "Yes"])
            smovies = st.selectbox("Streaming Movies", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "⚡  ANALYZE CUSTOMER",
        type="primary",
        width="stretch"
    )


# -------------------- PREDICTION --------------------

if submitted:

    no_net = "No internet service"
    has_net = internet != "No"

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple if phone == "Yes" else "No phone service",
        "InternetService": internet,
        "OnlineSecurity": security if has_net else no_net,
        "OnlineBackup": backup if has_net else no_net,
        "DeviceProtection": protection if has_net else no_net,
        "TechSupport": techsupport if has_net else no_net,
        "StreamingTV": stv if has_net else no_net,
        "StreamingMovies": smovies if has_net else no_net,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": round(tenure * monthly, 2),
    }

    proba = model.predict_proba(
        pd.DataFrame([customer])
    )[0][1]

    churn = proba >= 0.5

    label = "Likely to Churn" if churn else "Likely to Stay"
    confidence = proba if churn else 1 - proba

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">02 / Prediction Results</div>',
        unsafe_allow_html=True
    )

    st.caption("AI-generated customer churn assessment.")

    # -------------------- RESULT CARDS --------------------

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Prediction Status</div>
            <div class="result-value">{label}</div>
            <div class="result-description">
                {"Customer may discontinue the service."
                if churn else
                "Customer is predicted to remain."}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Prediction Confidence</div>
            <div class="result-value">{confidence:.0%}</div>
            <div class="result-description">
                Confidence in the predicted class
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Churn Probability</div>
            <div class="result-value">{proba:.0%}</div>
            <div class="result-description">
                Estimated probability of customer churn
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.progress(
        float(proba),
        text=f"Churn Probability — {proba:.0%}"
    )

    # -------------------- DETAILS + AI ADVISOR --------------------

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:

        st.markdown(
            '<div class="section-heading">03 / Customer Details</div>',
            unsafe_allow_html=True
        )

        st.caption("Summary of the information used for prediction.")

        st.dataframe(
            pd.DataFrame(
                customer.items(),
                columns=["Field", "Value"]
            ).astype(str),
            hide_index=True,
            width="stretch"
        )

    with right:

        st.markdown(
            '<div class="section-heading">04 / AI Business Advisor</div>',
            unsafe_allow_html=True
        )

        st.caption("Explanation, risk factors and retention actions from the LLM.")

        with st.container(border=True):
            if not get_api_key():
                st.warning("Add GROQ_API_KEY to your .env file to enable the advisor.")
            else:
                with st.spinner("Analysing customer..."):
                    try:
                        advice = ask_advisor(customer, label, confidence)
                        st.markdown(
                            '<div class="advisor-box">',
                            unsafe_allow_html=True
                        )
                        st.markdown(advice)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"LLM error: {e}")


# -------------------- FOOTER --------------------

st.markdown("""
<div class="footer">
    CUSTOMER INTELLIGENCE • POWERED BY MACHINE LEARNING
</div>
""", unsafe_allow_html=True)