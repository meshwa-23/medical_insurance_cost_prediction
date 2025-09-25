import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# Configuration (tweak these)
# -----------------------------
USD_TO_INR = 83                # currency conversion used
COST_ADJUSTMENT = 0.15         # additional factor to bring US-based charges closer to typical Indian ranges
MODEL_OUTPUT_IN_THOUSANDS = None
#   - Set to True if your model was trained on (charges / 1000) -> i.e. model.predict returns values like 1.2 meaning $1200
#   - Set to False if your model.predict returns raw USD (e.g. 12000)
#   - Set to None to use a simple heuristic auto-detection

CLIP_MIN_INR = 50_000          # minimum realistic insurance cost (INR)
CLIP_MAX_INR = 500_000         # maximum realistic insurance cost (INR)

# -----------------------------
# Load model
# -----------------------------
try:
    model = joblib.load("best_model_pipeline.joblib")
except Exception as e:
    st.error(f"Could not load model: {e}")
    raise

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="💰", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💰 Medical Insurance Cost Predictor</h1>", unsafe_allow_html=True)
st.write("Fill in the details below to estimate your **health insurance cost** (in INR).")

# -----------------------------
# Sidebar for inputs
# -----------------------------
st.sidebar.header("User Input Features")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"]) 

# Advanced options (for debugging / tweaking)
with st.sidebar.expander("Advanced / Tuning (edit constants in code for persistent change)"):
    st.write("Current config used at runtime:")
    st.write({
        "USD_TO_INR": USD_TO_INR,
        "COST_ADJUSTMENT": COST_ADJUSTMENT,
        "MODEL_OUTPUT_IN_THOUSANDS": MODEL_OUTPUT_IN_THOUSANDS,
        "CLIP_MIN_INR": CLIP_MIN_INR,
        "CLIP_MAX_INR": CLIP_MAX_INR,
    })
    show_debug = st.checkbox("Show prediction debug info")

# -----------------------------
# Prediction helpers
# -----------------------------

def predict_inr(input_df, return_details=False):
    """Predict using pipeline and convert to INR, scaled into 50k–1L range.

    - Handles model output (raw USD or in thousands)
    - Applies USD->INR and COST_ADJUSTMENT
    - Rescales everything smoothly into 50k–1L
    """

    raw_pred = model.predict(input_df)[0]
    details = {"raw_pred": float(raw_pred)}

    # decide whether raw_pred represents 'thousands of USD' or raw USD
    if MODEL_OUTPUT_IN_THOUSANDS is None:
        if raw_pred < 100:
            usd_cost = float(raw_pred) * 1000.0
            details["inferred_scale"] = "thousands -> *1000"
        else:
            usd_cost = float(raw_pred)
            details["inferred_scale"] = "raw USD"
    else:
        if MODEL_OUTPUT_IN_THOUSANDS:
            usd_cost = float(raw_pred) * 1000.0
            details["inferred_scale"] = "forced thousands"
        else:
            usd_cost = float(raw_pred)
            details["inferred_scale"] = "forced raw USD"

    details["usd_cost"] = usd_cost

    # convert to INR and apply India adjustment factor
    inr_cost = usd_cost * USD_TO_INR
    details["inr_before_adjustment"] = inr_cost

    adjusted_inr = inr_cost * COST_ADJUSTMENT
    details["inr_after_adjustment"] = adjusted_inr

    # -------------------------
    # Rescale into 50k–1L range
    # -------------------------
    CLIP_MIN_INR = 50_000
    CLIP_MAX_INR = 100_000

    # Define an expected raw range before scaling
    min_val, max_val = 0, 500_000   # you can tune max_val based on your dataset
    scaled_inr = CLIP_MIN_INR + (adjusted_inr - min_val) * (CLIP_MAX_INR - CLIP_MIN_INR) / (max_val - min_val)

    # Keep final value inside bounds
    final_inr = float(np.clip(scaled_inr, CLIP_MIN_INR, CLIP_MAX_INR))
    details["final_inr_scaled"] = final_inr

    if return_details:
        return round(final_inr, 2), details
    return round(final_inr, 2)

# -----------------------------
# Run prediction and show results
# -----------------------------
if st.sidebar.button("💡 Predict"):
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Ensure dtypes match typical training dtypes (helps pipelines)
    input_df["age"] = input_df["age"].astype(int)
    input_df["bmi"] = input_df["bmi"].astype(float)
    input_df["children"] = input_df["children"].astype(int)

    # Predict
    try:
        if show_debug:
            inr_cost, details = predict_inr(input_df, return_details=True)
        else:
            inr_cost = predict_inr(input_df)

        st.markdown(
            f"<h2 style='text-align: center; color: green;'>Estimated Insurance Cost: ₹{inr_cost:,.0f}</h2>",
            unsafe_allow_html=True
        )

        if show_debug:
            st.write("---")
            st.write("**Debug details (model raw output & conversion steps)**")
            st.json(details)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # -------------------------
    # Feature Importance (best-effort)
    # -------------------------
    st.subheader("🌟 Feature Importance (Overall Model)")
    try:
        # find preprocessor (ColumnTransformer) and regressor (estimator with feature_importances_)
        preprocessor = None
        regressor = None
        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():
                if preprocessor is None and hasattr(step, "named_transformers_"):
                    preprocessor = step
                if regressor is None and hasattr(step, "feature_importances_"):
                    regressor = step

        if preprocessor is None or regressor is None:
            raise ValueError("Could not locate preprocessor or regressor inside the pipeline. Names may differ.")

        # Attempt to reconstruct feature names for OneHot encoded categorical columns
        try:
            # This assumes your preprocessor has a transformer named 'cat' that contains an OneHotEncoder
            cat_ohe = preprocessor.named_transformers_["cat"].named_steps.get("onehot") if hasattr(preprocessor.named_transformers_["cat"], "named_steps") else preprocessor.named_transformers_["cat"]
            cat_feature_names = cat_ohe.get_feature_names_out(["sex", "smoker", "region"])
        except Exception:
            # fallback: try direct access if "onehot" is not inside a pipeline
            try:
                cat_ohe = preprocessor.named_transformers_["cat"]
                cat_feature_names = cat_ohe.get_feature_names_out(["sex", "smoker", "region"])
            except Exception:
                cat_feature_names = np.array(["sex_*", "smoker_*", "region_*"])

        num_features = ["age", "bmi", "children"]
        feature_names = np.concatenate([num_features, cat_feature_names])

        importances = regressor.feature_importances_

        # if lengths mismatch, gracefully trim or pad
        if len(importances) != len(feature_names):
            # try to align by taking min length
            m = min(len(importances), len(feature_names))
            importances = importances[:m]
            feature_names = feature_names[:m]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        importance_df["Highlight"] = ["Top 5" if i < 5 else "Other" for i in range(len(importance_df))]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            hue="Highlight",
            dodge=False,
            ax=ax
        )
        ax.set_title("Feature Importance (RandomForest)")
        ax.legend(title="Features")
        st.pyplot(fig)

    except Exception as e:
        st.warning("Feature importance plot skipped: " + str(e))

    # -------------------------
    # Comparative Graphs
    # -------------------------
    st.subheader("⚖️ Comparative Predictions")

    try:
        smokers = input_df.copy()
        smokers["smoker"] = "yes"
        nonsmokers = input_df.copy()
        nonsmokers["smoker"] = "no"
        smoker_cost = predict_inr(smokers)
        nonsmoker_cost = predict_inr(nonsmokers)

        sex_male = input_df.copy()
        sex_male["sex"] = "male"
        sex_female = input_df.copy()
        sex_female["sex"] = "female"
        male_cost = predict_inr(sex_male)
        female_cost = predict_inr(sex_female)

        st.write("### Smoker vs Non-Smoker")
        smoker_df = pd.DataFrame({"Cost": [smoker_cost, nonsmoker_cost]}, index=["Smoker", "Non-Smoker"])
        st.bar_chart(smoker_df)

        st.write("### Male vs Female")
        gender_df = pd.DataFrame({"Cost": [male_cost, female_cost]}, index=["Male", "Female"])
        st.bar_chart(gender_df)

    except Exception as e:
        st.warning("Comparative graphs could not be generated: " + str(e))

    # -------------------------
    # What-if Analysis
    # -------------------------
    st.subheader("🔮 What-if Simulation")
    new_bmi = st.slider("Simulate new BMI", 10.0, 50.0, bmi)
    quit_smoking = st.checkbox("Simulate quitting smoking")

    sim_input = input_df.copy()
    sim_input["bmi"] = new_bmi
    if quit_smoking:
        sim_input["smoker"] = "no"

    try:
        new_cost = predict_inr(sim_input)
        st.success(f"Simulated Insurance Cost: ₹{new_cost:,.0f}")
    except Exception as e:
        st.warning("Simulation failed: " + str(e))

# -----------------------------
# Footer / Notes
# -----------------------------
st.markdown("---")
st.write("**Notes:**\n- The app applies a conversion + adjustment factor to convert US-based charges into a rough Indian-range estimate.\n- For the most accurate results, retrain the model with target values converted to INR (or directly with Indian data).\n- To change how aggressive the adjustment is, edit COST_ADJUSTMENT at the top of this file.")
