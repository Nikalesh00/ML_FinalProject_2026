import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Phishing Threat Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
)


# Load Models & Assets
@st.cache_resource
def load_assets():
  dt_model = joblib.load("model_decision_tree.pkl")
  rf_model = joblib.load("model_random_forest.pkl")
  knn_model = joblib.load("model_knn.pkl")
  scaler = joblib.load("scaler.pkl")
  feature_names = joblib.load("feature_names.pkl")
  return dt_model, rf_model, knn_model, scaler, feature_names


dt_model, rf_model, knn_model, scaler, feature_names = load_assets()

# Top Header Banner
st.markdown(
    "### 🛡️ Cyber Threat Intelligence & Phishing Website Detection System"
)
st.markdown(
    "Multi-Algorithm Evaluation • 100% Offline & Self-Contained • 3 ML"
    " Models Active"
)
st.markdown("---")

# Navigation Tabs
tab1, tab2 = st.tabs(["🔍 Threat Predictor", "📊 Model Governance & Metrics"])

with tab1:
  # Quick Presets Bar
  st.subheader("⚡ Quick Test Presets")
  preset_col1, preset_col2, preset_col3 = st.columns(3)

  preset_choice = "Custom"
  with preset_col1:
    if st.button("Load Safe Website Profile"):
      preset_choice = "Safe"
  with preset_col2:
    if st.button("Load Phishing Threat Profile"):
      preset_choice = "Phishing"

  st.markdown("---")

  # Layout: Left side inputs, Right side results dashboard
  col_input, col_results = st.columns([1.2, 1.8])

  with col_input:
    st.subheader("URL & Feature Attributes")

    # Select primary model to query
    selected_algo = st.selectbox(
        "Select Primary Classifier",
        ["Decision Tree", "Random Forest", "KNN"],
    )

    input_data = {}
    with st.expander(
        "Configure Website Parameters (30 Features)", expanded=True
    ):
      for feature in feature_names:
        default_val = 0
        if preset_choice == "Safe":
          default_val = 1 if feature not in ["having_IP_Address"] else 0
        elif preset_choice == "Phishing":
          default_val = (
              -1
              if feature in ["SSLfinal_State", "age_of_domain", "web_traffic"]
              else 0
          )

        input_data[feature] = st.number_input(
            label=feature, value=int(default_val), step=1, format="%d"
        )

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    analyze_btn = st.button(
        "Execute Security Scan", type="primary", use_container_width=True
    )

  with col_results:
    st.subheader("Security Payout & Threat Evaluation")

    if analyze_btn:
      # Pick correct model and run prediction
      if selected_algo == "Decision Tree":
        active_model = dt_model
        pred = active_model.predict(input_df)[0]
        prob = active_model.predict_proba(input_df)[0][pred]
      elif selected_algo == "Random Forest":
        active_model = rf_model
        pred = active_model.predict(input_df)[0]
        prob = active_model.predict_proba(input_df)[0][pred]
      else:
        active_model = knn_model
        pred = active_model.predict(input_scaled)[0]
        prob = active_model.predict_proba(input_scaled)[0][pred]

      # Display Big Result Card
      if pred == 1:
        st.success(
            "### ✅ Status: Legitimate & Safe Website\nConfidence Score: "
            f"**{prob * 100:.2f}%**"
        )
        st.balloons()
      else:
        st.error(
            "### ⚠️ Status: MALICIOUS PHISHING THREAT DETECTED!\nConfidence"
            f" Score: **{prob * 100:.2f}%**"
        )

      st.markdown("---")
      st.subheader("Cross-Algorithm Consensus Predictions")

      # Compare all 3 models side by side
      dt_pred = dt_model.predict(input_df)[0]
      rf_pred = rf_model.predict(input_df)[0]
      knn_pred = knn_model.predict(input_scaled)[0]

      res_col1, res_col2, res_col3 = st.columns(3)
      res_col1.metric("Decision Tree", "Safe" if dt_pred == 1 else "Phishing")
      res_col2.metric("Random Forest", "Safe" if rf_pred == 1 else "Phishing")
      res_col3.metric("KNN", "Safe" if knn_pred == 1 else "Phishing")
    else:
      st.info(
          "👈 Adjust parameters or select a preset on the left, then click"
          " **Execute Security Scan** to generate threat analytics."
      )

with tab2:
  st.subheader("Model Governance & Algorithm Specifications")
  st.write(
      "This system integrates three distinct supervised machine learning"
      " architectures evaluated on standard cybersecurity benchmarks:"
  )
  st.markdown(
      "- **Decision Tree Classifier:** Rule-based hierarchical splitting"
      " model for rapid feature evaluation."
  )
  st.markdown(
      "- **Random Forest Classifier:** Ensemble bagging method combining"
      " multiple decision trees to minimize overfitting and maximize"
      " accuracy."
  )
  st.markdown(
      "- **K-Nearest Neighbors (KNN):** Distance-metric spatial classification"
      " utilizing standardized feature vectors."
  )