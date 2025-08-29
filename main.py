# stacking_app.py
# Run: streamlit run stacking_app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==============================
# Config
# ==============================
st.set_page_config(page_title="Aria — Stacking Ensemble", layout="wide")
st.title("Aria — Stacking Ensemble")

ID_COL = "customer_no"
TARGET_COL = "Donor_Category"
POS_LABEL = "Patron+"
NEG_LABEL = "Under-Patron"

# ==============================
# Helper: rebuild pipeline from metadata
# ==============================
def build_pipeline_from_meta(meta, available_cols=None):
    model_choice = meta.get("model_choice")
    hyperparams = dict(meta.get("hyperparams", {}))

    # Strip bad hyperparams that break sklearn clone
    for bad_key in ["sparse", "sparse_output"]:
        if bad_key in hyperparams:
            hyperparams.pop(bad_key)

    if model_choice == "Logistic Regression":
        clf = LogisticRegression(**hyperparams)
    elif model_choice == "Random Forest":
        clf = RandomForestClassifier(**hyperparams)
    elif model_choice == "AdaBoost":
        clf = AdaBoostClassifier(**hyperparams)
    elif model_choice == "KNN":
        clf = KNeighborsClassifier(**hyperparams)
    elif model_choice == "Naive Bayes":
        clf = GaussianNB(**hyperparams)
    else:
        return None

    num_cols = meta.get("numeric_columns", [])
    cat_cols = meta.get("categorical_columns_used", [])

    if available_cols is not None:
        missing_cols = [c for c in num_cols + cat_cols if c not in available_cols]
        if missing_cols:
            import streamlit as st  # Ensure this is imported at the top
            st.warning(f"The following expected columns were not found in the uploaded CSV: {missing_cols}")

        num_cols = [c for c in num_cols if c in available_cols]
        cat_cols = [c for c in cat_cols if c in available_cols]

        # Debug type issues in categorical columns
    if available_cols is not None:
        bad_type_cols = []
        for col in cat_cols:
            if col in available_cols:
                unique_types = set(type(val) for val in available_cols[col].dropna())
                if len(unique_types) > 1:
                    bad_type_cols.append((col, unique_types))

        if bad_type_cols:
            for col, types_found in bad_type_cols:
                st.error(f"❌ Column '{col}' has mixed types: {types_found}. Fix this column in your CSV.")
            raise ValueError("One or more categorical columns contain mixed types.")

        for col in cat_cols:
            if col in available_cols:
                available_cols[col] = available_cols[col].astype(str)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer([
        ("num", MinMaxScaler(), num_cols),
        ("cat", ohe, cat_cols)
    ], remainder="drop")

    return Pipeline([("prep", pre), ("clf", clf)])


# ==============================
# Meta-model choice (now with knobs)
# ==============================
st.subheader("Choose Final Meta-Model")

meta_model_choice = st.selectbox(
    "Final Meta-Model",
    ["Logistic Regression", "Random Forest", "AdaBoost", "KNN", "Naive Bayes"]
)

if meta_model_choice == "Logistic Regression":
    max_iter = st.slider("Max Iterations", 200, 5000, 1000, 100)
    final_estimator = LogisticRegression(max_iter=max_iter, class_weight="balanced")

elif meta_model_choice == "Random Forest":
    n_estimators = st.slider("Number of Trees (n_estimators)", 50, 1000, 200, 50)
    max_depth = st.slider("Max Depth (None = unlimited)", 1, 50, 10, 1)
    final_estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None if max_depth == 50 else max_depth,
        class_weight="balanced_subsample",
        random_state=42
    )

elif meta_model_choice == "AdaBoost":
    n_estimators = st.slider("Number of Estimators", 50, 800, 200, 50)
    learning_rate = st.slider("Learning Rate", 0.01, 2.0, 1.0, 0.01)
    final_estimator = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

elif meta_model_choice == "KNN":
    k = st.slider("Number of Neighbors (k)", 3, 50, 10, 1)
    final_estimator = KNeighborsClassifier(n_neighbors=k, weights="distance")

else:  # Naive Bayes
    final_estimator = GaussianNB()


# ==============================
# Upload base models
# ==============================
st.subheader("Upload Base Models")
n_models = st.number_input("Number of base models", min_value=1, max_value=10, step=1)

uploaded_models = []
for i in range(int(n_models)):
    uploaded = st.file_uploader(f"Upload model {i+1} (.pkl)", type=["pkl"], key=f"pkl_{i}")
    if uploaded:
        try:
            bundle = joblib.load(uploaded)

            if isinstance(bundle, dict) and "model_choice" in bundle:
                pipe = build_pipeline_from_meta(bundle, available_cols=df.columns)
                if pipe is not None:
                    uploaded_models.append((f"model{i+1}", pipe))
                    st.success(f"Loaded metadata bundle for Model {i+1}")

            elif isinstance(bundle, dict) and "pipeline" in bundle:
                pipe = bundle["pipeline"]
                uploaded_models.append((f"model{i+1}", pipe))
                st.success(f"Loaded full pipeline for Model {i+1}")

            elif isinstance(bundle, Pipeline):
                uploaded_models.append((f"model{i+1}", bundle))
                st.success(f"Loaded sklearn Pipeline for Model {i+1}")

            else:
                st.error(f".pkl {i+1} not recognized (must contain pipeline or metadata).")

        except Exception as e:
            st.error(f"Could not load .pkl {i+1}: {e}")


# ==============================
# Upload training CSV (meta-model training)
# ==============================
train_file = st.file_uploader("Upload Training CSV (must include Donor_Category)", type=["csv"])

if train_file and st.button("Train Meta-Model"):
    df = pd.read_csv(train_file)

    st.write("Column types:")
    st.write(df.dtypes)

    if TARGET_COL not in df.columns or ID_COL not in df.columns:
        st.error(f"CSV must include both '{ID_COL}' and '{TARGET_COL}'")
        st.stop()

    # Force all object columns to strings to avoid OneHotEncoder crash
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    X_train = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y_train = df[TARGET_COL]

    X_train = X_train.fillna(0.5)

    if len(uploaded_models) == 0:
        st.error("Please upload at least one base model.")
        st.stop()

    stack = StackingClassifier(
        estimators=uploaded_models,
        final_estimator=final_estimator,
        passthrough=False,
        n_jobs=-1
    )

    stack.fit(X_train, y_train)
    st.session_state["meta_model"] = stack
    st.success("Meta-model trained successfully!")

    # Show base model weights if available
    st.subheader("Base Model Weights in Meta-Model")
    # Show normalized (percentage) weights if available
    if hasattr(stack.final_estimator_, "coef_"):
        weights = stack.final_estimator_.coef_.flatten()
        abs_sum = np.sum(np.abs(weights))
        if abs_sum > 0:
            st.subheader("Base Model Relative Importance (%)")
            for name, weight in zip([name for name, _ in uploaded_models], weights):
                pct = 100 * np.abs(weight) / abs_sum
                st.write(f"**{name}**: {pct:.1f}%")
        else:
            st.info("All coefficients are zero; cannot compute relative importance.")

    preds = stack.predict(X_train)
    st.subheader("Classification Report")
    st.text(classification_report(y_train, preds))

    cm = confusion_matrix(y_train, preds, labels=[NEG_LABEL, POS_LABEL])
    fig, ax = plt.subplots(figsize = (2.5,2.5))
    ConfusionMatrixDisplay(cm, display_labels=[NEG_LABEL, POS_LABEL]).plot(ax=ax)
    st.pyplot(fig)


# ==============================
# Prediction on new CSV (no target column)
# ==============================
st.subheader("Predict with Meta-Model")
pred_file = st.file_uploader("Upload new CSV for prediction (no Donor_Category)", type=["csv"], key="pred")

thr = st.slider("Decision Threshold for Patron+", 0.0, 1.0, 0.5, 0.01)

if pred_file and "meta_model" in st.session_state:
    new_df = pd.read_csv(pred_file)
        # Clean up Excel-like error values before processing
    error_values = ["#DIV/0!", "#N/A", "#VALUE!", "N/A", "na", "NA", "null"]
    new_df.replace(error_values, np.nan, inplace=True)

    # Optional: force all object columns to strings to avoid OneHotEncoder crash
    for col in new_df.select_dtypes(include="object").columns:
        new_df[col] = new_df[col].astype(str)

    if ID_COL not in new_df.columns:
        st.error(f"Prediction CSV must include '{ID_COL}'")
    else:
        X_new = new_df.drop(columns=[ID_COL, TARGET_COL], errors="ignore")
        X_new = X_new.fillna(0.5)

        probs = st.session_state["meta_model"].predict_proba(X_new)
        pos_idx = list(st.session_state["meta_model"].classes_).index(POS_LABEL)
        pos_probs = probs[:, pos_idx]
        preds = np.where(pos_probs >= thr, POS_LABEL, NEG_LABEL)

        out = pd.DataFrame({
            ID_COL: new_df[ID_COL],
            f"proba_{POS_LABEL}": pos_probs,
            "prediction": preds
        })

        st.dataframe(out.head(20))
        st.download_button(
            "Download Predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="aria_meta_predictions.csv",
            mime="text/csv"
        )
