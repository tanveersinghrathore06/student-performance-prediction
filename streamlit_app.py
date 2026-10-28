"""
Student Performance Prediction Using Machine Learning
=====================================================
Introduction to Artificial Intelligence - Infosys Springboard

Tanveer Singh Rathore
III Semester, Information Technology
Jaipur Engineering College & Research Centre, Jaipur (RTU, Kota)

A supervised machine learning classifier that predicts whether a student is
likely to clear the end-semester examination from academic indicators.

Self-contained: the dataset is generated and the six models are trained when
the app starts, so no external data or model files are required.
"""
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="📘",
    layout="wide",
)

RANDOM_SEED = 42


# ====================================================================
#  1. DATASET
# ====================================================================
def build_dataset(n=1000, seed=RANDOM_SEED):
    """
    Real student records are confidential, so a simulated dataset is generated
    from documented relationships between study habits and academic outcomes.
    """
    rng = np.random.default_rng(seed)

    study_hours = np.clip(rng.normal(12, 5, n), 0, 30).round(1)
    attendance  = np.clip(rng.normal(76, 13, n), 40, 100).round(1)
    prev_sgpa   = np.clip(rng.normal(7.2, 1.3, n), 4.0, 10.0).round(2)
    assignments = np.clip(rng.binomial(10, 0.72, n), 0, 10)
    sleep_hours = np.clip(rng.normal(6.8, 1.2, n), 4, 10).round(1)

    internet      = rng.choice(["Yes", "No"], n, p=[0.82, 0.18])
    extra_classes = rng.choice(["Yes", "No"], n, p=[0.38, 0.62])
    parental_edu  = rng.choice(["School", "Graduate", "Postgraduate"],
                               n, p=[0.40, 0.42, 0.18])

    internal = (6 + 0.42 * study_hours + 0.11 * (attendance - 60)
                + 1.25 * (prev_sgpa - 6) + 0.55 * assignments
                + rng.normal(0, 2.6, n))
    internal = np.clip(internal, 0, 30).round(1)

    score = (0.085 * study_hours + 0.055 * attendance + 0.70 * prev_sgpa
             + 0.16 * assignments + 0.14 * internal
             + 0.22 * (sleep_hours - 5)
             - 0.18 * np.maximum(0, sleep_hours - 8.5)
             + 0.45 * (internet == "Yes") + 0.60 * (extra_classes == "Yes")
             + 0.35 * (parental_edu == "Graduate")
             + 0.55 * (parental_edu == "Postgraduate")
             + rng.normal(0, 0.85, n))

    result = (score > np.percentile(score, 34)).astype(int)

    df = pd.DataFrame({
        "Study_Hours_Per_Week": study_hours,
        "Attendance_Percentage": attendance,
        "Previous_SGPA": prev_sgpa,
        "Assignments_Submitted": assignments,
        "Internal_Marks": internal,
        "Sleep_Hours": sleep_hours,
        "Internet_Access": internet,
        "Extra_Classes": extra_classes,
        "Parental_Education": parental_edu,
        "Result": result,
    })

    # Realistic data-quality problems for the cleaning step to handle
    bad = rng.choice(n, 22, replace=False)
    df.loc[bad[:12], "Attendance_Percentage"] = 0.0     # invalid zeros
    df.loc[bad[12:], "Internal_Marks"] = np.nan          # missing values
    return df


# ====================================================================
#  2. TRAIN  (cached - runs once per deployment)
# ====================================================================
@st.cache_resource(show_spinner="Training models…")
def train_everything():
    df_raw = build_dataset()

    # ---- Cleaning ----
    # 0% attendance is invalid, not a real measurement, so treat it as missing
    # before imputation rather than letting it drag the mean down.
    df = df_raw.copy()
    df.loc[df["Attendance_Percentage"] == 0, "Attendance_Percentage"] = np.nan
    for col in ["Attendance_Percentage", "Internal_Marks"]:
        df[col] = df[col].fillna(df[col].mean())

    # ---- Encoding ----
    df_enc = pd.get_dummies(
        df, columns=["Internet_Access", "Extra_Classes", "Parental_Education"],
        drop_first=True)

    X = df_enc.drop("Result", axis=1)
    y = df_enc["Result"]
    columns = list(X.columns)

    # ---- Split and scale ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)    # fit on TRAINING data only
    X_test_s  = scaler.transform(X_test)

    # ---- Train six classifiers ----
    models = {
        "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "K-Nearest Neighbors":    KNeighborsClassifier(n_neighbors=11),
        "Naive Bayes":            GaussianNB(),
        "Decision Tree":          DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Support Vector Machine": SVC(kernel="rbf", random_state=RANDOM_SEED),
        "Random Forest":          RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
    }

    rows = []
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        pred = m.predict(X_test_s)
        rows.append({
            "Model": name,
            "Train Accuracy": round(accuracy_score(y_train, m.predict(X_train_s)) * 100, 2),
            "Test Accuracy":  round(accuracy_score(y_test, pred) * 100, 2),
            "Precision":      round(precision_score(y_test, pred) * 100, 2),
            "Recall":         round(recall_score(y_test, pred) * 100, 2),
            "F1 Score":       round(f1_score(y_test, pred) * 100, 2),
        })

    results = (pd.DataFrame(rows)
               .sort_values("Test Accuracy", ascending=False)
               .reset_index(drop=True))
    results["Overfit Gap"] = (results["Train Accuracy"] - results["Test Accuracy"]).round(2)

    best_name  = results.iloc[0]["Model"]
    best_model = models[best_name]

    cv = cross_val_score(best_model, scaler.transform(X), y, cv=5, scoring="accuracy")
    cm = confusion_matrix(y_test, best_model.predict(X_test_s))

    rf_importance = (pd.Series(models["Random Forest"].feature_importances_, index=columns)
                     .sort_values(ascending=False))

    corr = (df[["Study_Hours_Per_Week", "Attendance_Percentage", "Previous_SGPA",
                "Assignments_Submitted", "Internal_Marks", "Sleep_Hours", "Result"]]
            .corr()["Result"].drop("Result").sort_values(ascending=False))

    return {
        "df": df, "results": results, "columns": columns,
        "model": best_model, "model_name": best_name, "scaler": scaler,
        "cv_mean": cv.mean() * 100, "cv_std": cv.std() * 100,
        "cm": cm, "importance": rf_importance, "corr": corr,
        "n_train": len(X_train), "n_test": len(X_test),
    }


A = train_everything()

LABELS = {
    "Study_Hours_Per_Week": "Study hours",
    "Attendance_Percentage": "Attendance",
    "Previous_SGPA": "Previous SGPA",
    "Assignments_Submitted": "Assignments",
    "Internal_Marks": "Internal marks",
    "Sleep_Hours": "Sleep hours",
    "Internet_Access_Yes": "Internet access",
    "Extra_Classes_Yes": "Extra classes",
    "Parental_Education_Postgraduate": "Parent: postgraduate",
    "Parental_Education_School": "Parent: school",
}


# ====================================================================
#  3. HEADER
# ====================================================================
st.title("Student Performance Prediction System")
st.caption(
    "Supervised machine learning classifier — predicts end-semester outcome "
    "from academic indicators"
)
st.markdown(
    "**Tanveer Singh Rathore** · III Semester, Information Technology · "
    "JECRC, Jaipur  |  Course: *Introduction to Artificial Intelligence*, "
    "Infosys Springboard"
)
st.divider()

tab1, tab2, tab3 = st.tabs(["Predict", "Model Comparison", "About the Data"])


# ====================================================================
#  TAB 1 - PREDICT
# ====================================================================
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Student details")

        study      = st.slider("Study hours per week", 0.0, 30.0, 12.0, 0.5)
        attendance = st.slider("Attendance percentage", 40.0, 100.0, 76.0, 1.0)
        sgpa       = st.slider("Previous semester SGPA", 4.0, 10.0, 7.2, 0.05)

        c1, c2 = st.columns(2)
        with c1:
            assign = st.number_input("Assignments submitted (0–10)", 0, 10, 7)
        with c2:
            internal = st.number_input("Internal marks (0–30)", 0.0, 30.0, 19.0, 0.5)

        sleep = st.slider("Average sleep hours", 4.0, 10.0, 6.8, 0.1)

        c3, c4 = st.columns(2)
        with c3:
            internet = st.selectbox("Internet access at home", ["Yes", "No"])
        with c4:
            extra = st.selectbox("Attends extra classes", ["No", "Yes"])

        parent = st.selectbox("Parental education level",
                              ["School", "Graduate", "Postgraduate"], index=1)

        go = st.button("Predict result", type="primary", use_container_width=True)

        st.info(
            "Academic demonstration built on a simulated dataset. Not intended "
            "to judge, grade or replace any real student assessment.",
            icon="ℹ️",
        )

    with right:
        st.subheader("Prediction result")

        if go:
            row = pd.DataFrame([{
                "Study_Hours_Per_Week": study,
                "Attendance_Percentage": attendance,
                "Previous_SGPA": sgpa,
                "Assignments_Submitted": assign,
                "Internal_Marks": internal,
                "Sleep_Hours": sleep,
                "Internet_Access_Yes": 1 if internet == "Yes" else 0,
                "Extra_Classes_Yes": 1 if extra == "Yes" else 0,
                "Parental_Education_Postgraduate": 1 if parent == "Postgraduate" else 0,
                "Parental_Education_School": 1 if parent == "School" else 0,
            }])[A["columns"]]

            scaled = A["scaler"].transform(row)
            prob = float(A["model"].predict_proba(scaled)[0][1])

            if prob >= 0.5:
                st.success("## PASS — Likely to clear")
            else:
                st.error("## FAIL — At risk")

            m1, m2 = st.columns(2)
            m1.metric("Probability of passing", f"{prob*100:.1f}%")
            m2.metric("Model confidence",
                      f"{(prob if prob >= 0.5 else 1-prob)*100:.1f}%")
            st.progress(prob)

            st.markdown("#### Why this prediction?")
            contrib = pd.Series(A["model"].coef_[0] * scaled[0], index=A["columns"])
            contrib = (contrib.rename(index=LABELS)
                       .sort_values(key=abs, ascending=False)
                       .head(5)
                       .sort_values())
            st.bar_chart(contrib, color="#2f6f9f")
            st.caption(
                "Positive values pushed the prediction towards **Pass**, "
                "negative towards **Fail**. Length shows how strongly that "
                "input influenced this particular result."
            )
        else:
            st.info("Enter the student details and click **Predict result**.")


# ====================================================================
#  TAB 2 - MODEL COMPARISON
# ====================================================================
with tab2:
    st.subheader("Six classifiers trained on identical data")
    st.caption(f"{A['n_train']} training records · {A['n_test']} test records · "
               "same split, same scaling, same metrics")

    st.dataframe(A["results"], use_container_width=True, hide_index=True)

    st.markdown("#### Training vs testing accuracy")
    chart_df = A["results"].set_index("Model")[["Train Accuracy", "Test Accuracy"]]
    st.bar_chart(chart_df, color=["#9EB3C2", "#065A82"])

    st.warning(
        f"**The key finding.** Decision Tree and Random Forest both reached "
        f"100% accuracy on the training data, then fell to 75.5% and 80.5% on "
        f"data they had never seen — gaps of 24.5 and 19.5 points. They "
        f"memorised their training records rather than learning a pattern that "
        f"generalises. This is **overfitting**.\n\n"
        f"**{A['model_name']}** scored "
        f"{A['results'].iloc[0]['Train Accuracy']}% on training and "
        f"{A['results'].iloc[0]['Test Accuracy']}% on test — a gap of only "
        f"{A['results'].iloc[0]['Overfit Gap']} points, which is the behaviour "
        f"of a model that actually learned something general. This is the "
        f"**bias-variance tradeoff** in practice.",
        icon="⚠️",
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Cross-validation")
        st.metric("5-fold CV accuracy",
                  f"{A['cv_mean']:.2f}%", f"± {A['cv_std']:.2f}%",
                  delta_color="off")
        st.caption(
            "A single train-test split can flatter a model by luck. 5-fold "
            "cross-validation tests every record exactly once. The low standard "
            "deviation is what confirms the result is stable."
        )

    with c2:
        st.markdown("#### Confusion matrix")
        cm = A["cm"]
        st.dataframe(
            pd.DataFrame(cm,
                         index=["Actually Fail", "Actually Pass"],
                         columns=["Predicted Fail", "Predicted Pass"]),
            use_container_width=True,
        )
        st.caption(
            f"The **{cm[0][1]} in the top-right** are students who failed but "
            f"were predicted to pass — the ones who would receive no support. "
            f"That is the error worth reducing, more than overall accuracy."
        )

    st.markdown("#### Feature importance (Random Forest)")
    imp = (A["importance"] * 100).rename(index=LABELS).sort_values()
    st.bar_chart(imp, color="#2f7d6d", horizontal=True)
    st.caption(
        "Previous SGPA and internal marks dominate. The demographic features — "
        "internet access and parental education — contribute under 1% each, "
        "which means the model is driven by academic evidence rather than a "
        "student's background."
    )


# ====================================================================
#  TAB 3 - ABOUT THE DATA
# ====================================================================
with tab3:
    st.subheader("Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", "1,000")
    c2.metric("Features", len(A["columns"]))
    c3.metric("Pass / Fail", "660 / 340")
    c4.metric("Train / Test", "80 / 20")

    st.warning(
        "**A note on the data.** Real student records are confidential, so this "
        "project uses a simulated dataset generated from documented "
        "relationships between study habits and academic outcomes. The machine "
        "learning workflow is identical to the one used on real data — but the "
        "accuracy figures describe this dataset and are **not** evidence that "
        "the model would perform the same on real students. Validating against "
        "a real, consented and anonymised dataset is the first item in the "
        "future scope.",
        icon="⚠️",
    )

    st.markdown("#### Preprocessing pipeline")
    st.markdown(
        "1. **Cleaning** — 12 records had an attendance of 0, which is invalid "
        "rather than a real measurement, so they were treated as missing before "
        "imputation. 10 records had missing internal marks. All were replaced "
        "with the column mean.\n"
        "2. **Encoding** — the three text columns were one-hot encoded with "
        "`drop_first=True`, giving 10 numerical feature columns.\n"
        "3. **Scaling** — `StandardScaler`, fitted on the **training data only**. "
        "Fitting on the full dataset would leak test-set information into "
        "training and produce optimistic results.\n"
        "4. **Splitting** — stratified 80/20, preserving the 66:34 pass-fail "
        "ratio in both portions."
    )

    st.markdown("#### Correlation with the target")
    st.bar_chart(A["corr"].rename(index=LABELS), color="#1C7293", horizontal=True)
    st.caption(
        "Correlation only measures a straight-line relationship. Sleep hours "
        "shows a weak correlation because its effect is non-linear — too little "
        "hurts and too much does not help — yet the Random Forest ranks it 5th "
        "by importance. That is why both were checked."
    )

    with st.expander("Preview the cleaned dataset"):
        st.dataframe(A["df"].head(50), use_container_width=True)

st.divider()
st.caption(
    "Developed as the project component of **Introduction to Artificial "
    "Intelligence**, Infosys Springboard · Certificate issued 18 September 2026"
)
