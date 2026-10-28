# Student Performance Prediction Using Machine Learning

A supervised machine learning system that predicts whether a student is likely to clear
the end-semester examination, so that support can be given before the exam instead of
after the result.

Built as the project for the **Introduction to Artificial Intelligence** course
(Infosys Springboard).

**Tanveer Singh Rathore** · III Semester, Information Technology
Jaipur Engineering College & Research Centre, Jaipur · Rajasthan Technical University

---

## The problem

Colleges usually find out which students were struggling only after the result is
declared, when it is too late to help them. The warning signs are already in the college
records — attendance, internal marks, assignments submitted, previous SGPA — but they are
never analysed together.

This project treats that as a **supervised classification problem**.

---

## Results

Six classification algorithms were trained on identical data and compared on the same
test set.

| Model | Train Accuracy | Test Accuracy | F1 Score | Gap |
|---|---|---|---|---|
| **Logistic Regression** | 86.12% | **84.00%** | **88.06%** | **2.12** |
| Support Vector Machine | 88.50% | 82.00% | 86.57% | 6.50 |
| K-Nearest Neighbors | 85.88% | 81.50% | 86.83% | 4.38 |
| Naive Bayes | 84.88% | 81.50% | 86.25% | 3.38 |
| Random Forest | 100.00% | 80.50% | 85.50% | 19.50 |
| Decision Tree | 100.00% | 75.50% | 81.08% | 24.50 |

**Selected model:** Logistic Regression, validated at **85.30% (± 1.03%)** by 5-fold
cross-validation.

### Key finding

Decision Tree and Random Forest both reached **100% accuracy on the training data**, then
fell to 75.5% and 80.5% on data they had never seen. They memorised their training records
instead of learning a pattern that generalises — a direct demonstration of **overfitting**.

Logistic Regression scored 86.12% on training and 84.00% on test, a gap of only 2.12
points. The simplest model generalised best. This is the **bias-variance tradeoff** in
practice: on 1,000 records with 10 features, the flexibility of the complex models cost
more in variance than it saved in bias.

---

## A note on the data

Real student records are confidential, so this project uses a **simulated dataset** of
1,000 records generated from documented relationships between study habits and academic
outcomes.

The machine learning workflow is identical to the one used on real data. However, the
accuracy figures above describe performance on this simulated dataset and are **not**
evidence that the model would perform the same on real students. Validating against a
real, consented and anonymised dataset is the first item in the future scope.

---

## Dataset

1,000 records · 9 input features · binary target (660 Pass / 340 Fail)

| Feature | Type | Range |
|---|---|---|
| Study_Hours_Per_Week | Numerical | 0 – 30 |
| Attendance_Percentage | Numerical | 40 – 100 |
| Previous_SGPA | Numerical | 4.0 – 10.0 |
| Assignments_Submitted | Numerical | 0 – 10 |
| Internal_Marks | Numerical | 0 – 30 |
| Sleep_Hours | Numerical | 4 – 10 |
| Internet_Access | Categorical | Yes / No |
| Extra_Classes | Categorical | Yes / No |
| Parental_Education | Categorical | School / Graduate / Postgraduate |
| **Result** | **Target** | **0 = Fail, 1 = Pass** |

---

## Method

1. **Cleaning** — 12 records had an attendance of 0, which is invalid rather than a real
   measurement, so they were treated as missing before imputation. 10 records had missing
   internal marks. All were replaced with the column mean.
2. **Encoding** — the three text columns were one-hot encoded, giving 10 numerical columns.
3. **Splitting** — stratified 80/20, preserving the 66:34 pass-fail ratio in both parts.
4. **Scaling** — `StandardScaler`, fitted on the **training data only**. Fitting on the
   full dataset would leak test information into training.
5. **Training** — six algorithms, identical data, identical metrics.
6. **Evaluation** — accuracy, precision, recall, F1 and the confusion matrix, plus 5-fold
   cross-validation on the selected model.

### Feature importance

| Feature | Importance |
|---|---|
| Previous_SGPA | 27.2% |
| Internal_Marks | 25.9% |
| Attendance_Percentage | 15.0% |
| Study_Hours_Per_Week | 12.7% |
| Sleep_Hours | 9.1% |
| *(demographic features)* | *< 1.5% each* |

The demographic features contribute almost nothing, which means the model is driven by
academic evidence rather than a student's background.

---

## Running it

### Online

The application is deployed on Streamlit Community Cloud. The link is in the repository
description.

### Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The dataset is generated and the models are trained when the app starts, so there is
nothing to download.

---

## Files

```
streamlit_app.py    the complete application - generates data, trains 6 models, serves the UI
requirements.txt    the libraries needed
README.md           this file
```

---

## Future scope

1. **Validate on real data** — a consented, anonymised institutional dataset.
2. Hyperparameter tuning with `GridSearchCV`; pruning to control the overfitting observed.
3. SMOTE or class weighting to raise recall on the failing class.
4. Risk bands instead of a binary label, and SHAP values for per-prediction explanations.

### Responsible use

This system is intended only to **direct supportive attention** towards students who may
need it. It must never be used for grading, ranking, or any decision affecting a student's
standing, and a human should remain in the loop for every prediction.

---

## Built with

Python · scikit-learn · pandas · NumPy · Streamlit
