<h1 align="center">❤️ Heart Disease Prediction</h1>

> A multi-model classification project to predict the presence of heart disease from clinical patient data — built to benchmark 7 ML algorithms and identify the best-performing approach.

---

# 📌 Problem

Heart disease is a leading cause of mortality worldwide. Early and accurate detection from clinical indicators can assist doctors in triaging high-risk patients faster. This project frames the problem as a **binary classification task**: predict whether a patient has heart disease (`1`) or not (`0`).

---

# 📂 Dataset

| Property | Details |
|----------|---------|
| Source | [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Records | 908 patients |
| Features | 12 clinical features |
| Target | `num` → binarized: `0` = No Disease, `1` = Disease Present |
| Missing Values | `oldpeak` had 62 nulls → filled with **median** (right-skewed + outliers) |

**Features used:**

| Feature | Description |
|---------|-------------|
| `age` | Patient age |
| `sex` | Gender (Male/Female) |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results |
| `thalch` | Max heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `thal` | Thalassemia type |

---

# 🔍 Approach

1. **EDA** — Checked distributions (histplot, boxplot) across all numeric features; flagged `oldpeak` as right-skewed with outliers
2. **Missing Value Treatment** — Median imputation on `oldpeak` (preferred over mean due to skew + outliers)
3. **Encoding** — One-hot encoding (`pd.get_dummies`, `drop_first=True`) on all categorical & boolean columns
4. **Target Binarization** — `num` ∈ {0,1,2,3,4} → mapped to binary: `0` = No Disease, `1` = Disease
5. **Train/Test Split** — 80/20, `random_state=42`
6. **Modeling** — 7 classifiers trained and compared
7. **Evaluation** — Accuracy score + bar chart comparison across all models

---

# 🤖 Model / Algorithm

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Linear | No tuning — strong baseline |
| Gaussian Naive Bayes | Probabilistic | Assumes feature independence |
| Random Forest | Ensemble (Bagging) | 100 trees, `random_state=42` |
| XGBoost | Ensemble (Boosting) | `random_state=42` |
| Decision Tree | Tree-based | No pruning — prone to overfitting |
| SVM | Kernel-based | No feature scaling applied |
| KNN | Distance-based | No feature scaling — known limitation |

> ⚠️ SVM and KNN are scale-sensitive. No `StandardScaler` was applied in this version — a clear improvement area.

---

# 📊 Results

| Rank | Model | Accuracy |
|------|-------|----------|
| 🥇 1 | Logistic Regression | **83.52%** |
| 2 | Gaussian Naive Bayes | 82.42% |
| 3 | Random Forest | 81.32% |
| 4 | XGBoost | 79.12% |
| 5 | Decision Tree | 78.02% |
| 6 | SVM | 71.43% |
| 7 | KNN | 68.68% |

**Best Model:** Logistic Regression — interpretable, robust on small structured datasets, no scaling required.

**Key Observations:**
- Simple models can outperform complex ensembles on small, structured clinical data
- KNN and SVM suffered without feature scaling — adding `StandardScaler` is the obvious next step
- XGBoost underperforming Random Forest signals the need for hyperparameter tuning

---

# ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Chaithanya449/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook Heart_Disease_prediction.ipynb
```

> Ensure `heart_disease.csv` is in the same directory as the notebook before running.

---

# 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-lightgrey?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-blue?logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-9cf)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-yellow)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

# 📁 Project Structure

```
heart-disease-prediction/
├── Heart_Disease_prediction.ipynb   # Main analysis notebook
├── heart_disease.csv                # Dataset (908 records, 13 columns)
├── requirements.txt                 # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

# 🔮 Next Steps

- [ ] Apply `StandardScaler` — expected to significantly boost KNN and SVM scores
- [ ] Hyperparameter tuning on XGBoost and Random Forest (GridSearchCV / Optuna)
- [ ] Expand metrics: AUC-ROC, Precision, Recall, F1 — accuracy alone is misleading for medical data
- [ ] Handle class imbalance with SMOTE or `class_weight='balanced'`
- [ ] Deploy best model via FastAPI + Docker

---

# 👤 Author

**Chaithanya Krishna** · [LinkedIn](https://www.linkedin.com/in/chaitanyakrishna-profile) · [GitHub](https://github.com/Chaithanya449)
