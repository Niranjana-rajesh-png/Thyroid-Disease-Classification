import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="ThyroCare AI | Thyroid Classification",
    page_icon="🦋",
    layout="wide"
)

# ---------- Theme / styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: #F4F7FB; border: 1px solid #E2E8F0;
    padding: 14px; border-radius: 12px;
}
.hero {
    padding: 26px; border-radius: 18px;
    background: linear-gradient(120deg, #14243A, #24556A);
    color: white; margin-bottom: 18px;
}
.hero h1 {color: white; margin: 0;}
.hero p {color: #D7E7EF; margin: 8px 0 0 0;}
.small-note {color: #64748B; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🦋 ThyroCare AI</h1>
  <p>Thyroid Disease Classification using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

BUNDLE_PATH = "thyroid_model_bundle.joblib"

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

try:
    bundle = load_bundle(BUNDLE_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{BUNDLE_PATH}`. Run the model-export cell in your Jupyter "
        "notebook first, then place the generated file in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model bundle: {exc}")
    st.stop()

models = bundle["models"]
class_names = bundle["class_names"]
dataset = bundle.get("dataset")
metrics = bundle.get("metrics", {})
raw_defaults = bundle["raw_defaults"]
field_options = bundle["field_options"]
flag_cols = bundle["flag_cols"]
lab_medians = bundle["lab_medians"]
sex_mode = bundle["sex_mode"]
num_cols = bundle["num_cols"]
feature_columns = bundle["feature_columns"]
scaler = bundle["scaler"]
referral_categories = bundle.get("referral_categories", [])

# ---------- Preprocessing: must mirror the notebook ----------
def prepare_patient(raw_row):
    """Convert one raw patient record into the exact model input format."""
    data = pd.DataFrame([raw_row]).copy()

    for col, value in lab_medians.items():
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(value)

    if "sex" in data.columns:
        data["sex"] = data["sex"].fillna(sex_mode)

    for col in flag_cols:
        if col in data.columns:
            data[col] = data[col].map(
                lambda v: 1 if v in ["t", 1, True, "True"] else
                          0 if v in ["f", 0, False, "False"] else v
            )

    if "sex" in data.columns:
        data["sex"] = data["sex"].map({"F": 1, "M": 0}).fillna(data["sex"])

    if "referral_source" in data.columns:
        data["referral_source"] = pd.Categorical(
            data["referral_source"], categories=referral_categories
        )
        data = pd.get_dummies(
            data, columns=["referral_source"], drop_first=True, dtype=int
        )

    if "TSH" in data.columns:
        data["TSH"] = np.log1p(pd.to_numeric(data["TSH"], errors="coerce"))

    data = data.reindex(columns=feature_columns, fill_value=0)

    # Apply the scaler fitted in the notebook, to the same numeric columns.
    scale_cols = [c for c in num_cols if c in data.columns]
    if scale_cols:
        data.loc[:, scale_cols] = scaler.transform(data[scale_cols])

    return data


# ---------- Single-page patient prediction dashboard ----------
st.markdown("## Patient Details")
st.write("Enter the available patient information below to generate thyroid-class predictions.")
st.info(
    "Academic project demonstration only. Predictions are not a medical diagnosis "
    "and must not guide treatment or clinical decisions."
)

with st.form("patient_form"):
    st.markdown("### Enter patient information")
    raw_input = {}
    columns = st.columns(2)

    for idx, (field, default) in enumerate(raw_defaults.items()):
        container = columns[idx % 2]
        with container:
            label = field.replace("_", " ").title()
            if field in field_options:
                options = field_options[field]
                if field == "sex":
                    options = [v for v in ["F", "M"] if v in options] or options
                chosen_default = default if default in options else options[0]
                raw_input[field] = st.selectbox(
                    label, options, index=options.index(chosen_default)
                )
            elif field in flag_cols:
                default_bool = default in [1, "t", "T", "True", True]
                raw_input[field] = "t" if st.checkbox(label, value=default_bool) else "f"
            else:
                try:
                    numeric_default = float(default)
                except (TypeError, ValueError):
                    numeric_default = 0.0
                step = 1.0 if field == "age" else 0.1
                raw_input[field] = st.number_input(
                    label, value=numeric_default, step=step, format="%.3f"
                )

    submitted = st.form_submit_button("Predict thyroid class", type="primary", use_container_width=True)

if submitted:
    try:
        patient_features = prepare_patient(raw_input)
        predictions = {}
        for model_name, model in models.items():
            predictions[model_name] = str(model.predict(patient_features)[0])

        from collections import Counter
        counts = Counter(predictions.values())
        highest_count = max(counts.values())
        tied_classes = [name for name, count in counts.items() if count == highest_count]

        if len(tied_classes) == 1:
            final_prediction = tied_classes[0]
            decision_note = "Selected by majority vote across the available trained models."
        else:
            # Resolve a vote tie using the saved test accuracy of models predicting
            # one of the tied classes. If metrics are unavailable, use class order.
            accuracy_by_class = {}
            for model_name, predicted_class in predictions.items():
                if predicted_class in tied_classes:
                    accuracy = metrics.get(model_name, {}).get("test_accuracy", 0)
                    accuracy_by_class[predicted_class] = max(
                        accuracy_by_class.get(predicted_class, 0), float(accuracy or 0)
                    )
            final_prediction = max(
                tied_classes,
                key=lambda name: (accuracy_by_class.get(name, 0), -list(class_names).index(name)
                                  if name in class_names else 0)
            )
            decision_note = "Vote was tied; tie-break used the saved test accuracy of the models."

        st.markdown("## Prediction Results")
        st.success(f"Final predicted class: **{final_prediction}**")
        st.caption(decision_note)

        results_df = pd.DataFrame([
            {"Model": model_name, "Predicted class": prediction}
            for model_name, prediction in predictions.items()
        ])
        st.markdown("### Individual model predictions")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        counts_df = pd.DataFrame(
            [{"Class": name, "Votes": count} for name, count in counts.most_common()]
        )
        st.markdown("### Model voting summary")
        st.dataframe(counts_df, use_container_width=True, hide_index=True)

        download_df = results_df.copy()
        download_df.loc[len(download_df)] = ["Final majority-vote result", final_prediction]
        st.download_button(
            "Download prediction report",
            data=download_df.to_csv(index=False).encode("utf-8"),
            file_name="thyroid_prediction_report.csv",
            mime="text/csv"
        )
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        st.info(
            "Check that the model bundle was exported after the final preprocessing "
            "and model-training cells, and that it is in the same folder as app.py."
        )

st.markdown("---")
st.caption("ThyroCare AI · Academic project demonstration · Not intended for clinical diagnosis")
