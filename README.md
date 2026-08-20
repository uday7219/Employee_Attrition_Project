# 📊 Employee Attrition Prediction

> **End-to-End Machine Learning Project \| XGBoost \| Streamlit
> Deployment**

A professional machine learning project that predicts whether an
employee is likely to leave an organization based on demographic,
professional, compensation, work-environment, engagement, satisfaction,
and employee-support attributes.

The project covers the complete ML lifecycle: **data understanding → EDA
→ cleaning → preprocessing → baseline modeling → XGBoost →
hyperparameter tuning → evaluation → model persistence → Streamlit
deployment**.

------------------------------------------------------------------------

## 🎯 Project Objective

Employee attrition can create significant recruitment, training,
productivity, and operational costs. The objective of this project is to
build a **binary classification system** that estimates whether an
employee is likely to experience attrition.

### Target Variable

  Value   Meaning
  ------- --------------------
  `Yes`   Employee attrition
  `No`    Employee remains

**Machine Learning Task:** Binary Classification

The model is intended to support HR analysis and early risk
identification. It should **not** be used as the sole basis for
employment decisions.

------------------------------------------------------------------------

## ✨ Key Features

-   Complete end-to-end machine learning workflow
-   Exploratory Data Analysis and visualization
-   Duplicate and text-value cleaning
-   Automatic numerical and categorical feature detection
-   Numerical imputation and standardization
-   Categorical imputation and One-Hot Encoding
-   Multiple baseline classification models
-   XGBoost classification
-   5-fold Stratified Cross-Validation
-   GridSearchCV hyperparameter optimization
-   Evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
-   Confusion matrix and ROC analysis
-   Saved production-ready model using Joblib
-   Interactive Streamlit prediction application
-   Attrition probability and risk-level display

------------------------------------------------------------------------

## 🗂️ Dataset

The project uses the provided employee attrition workbook:

``` text
employee_attrition.xlsx
```

### Dataset Summary

-   **Rows:** 1,191
-   **Columns:** 35
-   **Target:** `Attrition`
-   **Identifier:** `ID`

The notebook documents the dataset as **user-provided**. The workbook
itself does not establish an external source URL or provenance, so the
original source should be verified before claiming a public dataset
source.

### Feature Categories

The application uses employee attributes covering areas such as:

-   Gender
-   Age
-   Marital status
-   Academic degree
-   Years of experience
-   Experience at previous organization
-   Sector
-   Department
-   Job title
-   Monthly salary
-   Allowances
-   Medical insurance
-   Bonus
-   Overtime
-   Overtime payment
-   Rewards and wages satisfaction
-   Promotion
-   Training
-   Business travel
-   Job support
-   Recognition
-   Emotional commitment
-   Job engagement
-   Distance to work
-   Work-life balance
-   Physical stress
-   Psychological exhaustion
-   Job stability
-   Health issues
-   Environment satisfaction
-   Job satisfaction
-   Job opportunities

------------------------------------------------------------------------

## 🧠 Machine Learning Workflow

``` text
Raw Employee Dataset
        │
        ▼
Data Understanding
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Data Cleaning
        │
        ▼
Feature / Target Separation
        │
        ▼
Train-Test Split
        │
        ▼
Preprocessing Pipeline
 ┌──────┴─────────┐
 ▼                ▼
Numerical       Categorical
Imputation      Imputation
   +               +
Scaling        One-Hot Encoding
 └──────┬─────────┘
        ▼
Baseline Models
        │
        ▼
XGBoost Selection
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Final Evaluation
        │
        ▼
Joblib Model
        │
        ▼
Streamlit Application
```

------------------------------------------------------------------------

## 🤖 Models Used

Four classifiers were evaluated using the same preprocessing framework:

1.  Logistic Regression
2.  Decision Tree
3.  Random Forest
4.  XGBoost

### Baseline Model Performance

 ------------------------------------------------------------------------------
  Model:             Accuracy,    Precision,       Recall,           F1,      ROC-AUC
  ------------- ------------ ------------ ------------ ------------ ------------
  **Tuned XGBoost**   **82.85%**   **84.44%**   **73.79%**   **78.76%**   **89.33%**

  **XGBoost**         **80.33%**   **79.79%**   **72.82%**   **76.14%**   **88.80%**

  **Logistic Regression**   **79.50%**   **77.55%**   **73.79%**   **75.62%**   **87.95%**

  **Decision Tree**   **78.24%**   **77.42%**   **69.90%**   **73.47%**   **80.00%**

  **Random Forest**   **71.97%**   **63.85%**   **80.58%**   **71.24%**   **80.08%**
------------------------------------------------------------------------------

**Best baseline:** Tuned XGBoost

## 🚀 XGBoost Hyperparameter Tuning

The selected XGBoost pipeline was optimized using **GridSearchCV** with
**5-fold Stratified Cross-Validation**.

### Search Space

-   `n_estimators`: 150, 200
-   `learning_rate`: 0.03, 0.05
-   `max_depth`: 2, 3, 4
-   `min_child_weight`: 5, 8
-   `subsample`: 0.7, 0.8
-   `colsample_bytree`: 0.7, 0.8
-   `gamma`: 0.2, 0.4
-   `reg_alpha`: 0.1, 0.5
-   `reg_lambda`: 2.0, 3.0

This produced **768 parameter combinations × 5 folds = 3,840 fits**.

### Best Parameters

``` text
colsample_bytree = 0.8
gamma             = 0.4
learning_rate     = 0.05
max_depth         = 4
min_child_weight  = 5
n_estimators      = 200
reg_alpha         = 0.1
reg_lambda        = 2.0
subsample         = 0.8
```

**Best cross-validation F1:** `0.7757`

------------------------------------------------------------------------

## 📈 Final Tuned Model Performance

The tuned XGBoost model was evaluated on the held-out test set.

  Metric          Score
  ----------- ---------
  Accuracy      **80%**
  Precision     **79%**
  Recall        **72%**
  F1-score      **76%**
  ROC-AUC       **88%**

### Classification Performance

  Class            Precision   Recall   F1-score
  -------------- ----------- -------- ----------
  No Attrition          0.82     0.90       0.86
  Attrition             0.84     0.74       0.79

The tuned model improved the baseline XGBoost result from approximately
**76.14% F1 / 88.80% ROC-AUC** to **78.76% F1 / 89.33% ROC-AUC** on the
recorded evaluation.

------------------------------------------------------------------------

## ⚙️ Preprocessing

The project uses a `ColumnTransformer` and `Pipeline` to keep
preprocessing and modeling together.

### Numerical Features

1.  Missing values → median imputation
2.  Features → StandardScaler

### Categorical Features

1.  Missing values → most-frequent imputation
2.  Categories → OneHotEncoder
3.  Unknown categories are safely ignored

This approach ensures that the same transformations used during training
are applied during inference.

------------------------------------------------------------------------

## 🏗️ Project Structure

``` text
Employee-Attrition-Prediction/
│
├── employee_attrition.xlsx
├── employee_attrition_prediction.ipynb
├── train_model.py
├── app.py
├── requirements.txt
│
├── models/
│   └── employee_attrition_model.joblib
│
└── README.md
```

> The exact filenames can be renamed to cleaner production names such as
> `train_model.py`, `app.py`, and `requirements.txt` without changing
> the project concept.

------------------------------------------------------------------------

## 💻 Installation

### 1. Clone the repository

``` bash
git clone <https://github.com/uday7219/Employee_Attrtion_Project>
cd Employee-Attrition-Prediction
```

### 2. Create a virtual environment

#### Windows

``` bash
python -m venv .employe
employee\Scripts\activate.bat
```

#### macOS / Linux

``` bash
python3 -m .employee venv
source employee/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

The project requirements include:

-   pandas
-   numpy
-   scikit-learn
-   xgboost
-   joblib
-   matplotlib
-   streamlit
-   scipy
-   seaborn
-   ipython
-   python-dotenv

------------------------------------------------------------------------

## 🏋️ Train the Model

Make sure the dataset is available in the project root:

``` text
employee_attrition.xlsx
```

Then run:

``` bash
python train_model.py
```

The training pipeline:

1.  Loads the Excel dataset
2.  Cleans column names and text values
3.  Removes duplicate rows
4.  Separates `Attrition` from the input features
5.  Removes `ID` from the feature set
6.  Performs an 80/20 stratified train-test split
7.  Builds the preprocessing pipeline
8.  Trains and evaluates models
9.  Tunes XGBoost using GridSearchCV
10. Saves the best estimator

The trained model is saved as:

``` text
models/employee_attrition_model.joblib
```

------------------------------------------------------------------------

## 🌐 Run the Streamlit Application

After training the model, run:

``` bash
streamlit run app.py
```

The application loads the saved Joblib model and provides an interactive
interface for entering employee attributes.

The application then displays:

-   Predicted attrition outcome
-   Estimated attrition probability
-   Risk level
-   Probability progress indicator

The current application uses `st.cache_resource` to load the saved model
efficiently.

------------------------------------------------------------------------

## 🖥️ Application Workflow

``` text
Open Streamlit App
        │
        ▼
Enter Employee Information
        │
        ▼
Validate Required Features
        │
        ▼
Load Saved ML Pipeline
        │
        ▼
Generate Prediction
        │
        ├───────────────┐
        ▼               ▼
Attrition = Yes     Attrition = No
        │               │
        ▼               ▼
HIGH Risk           LOW Risk
        │               │
        └───────┬───────┘
                ▼
   Display Probability + Result
```

------------------------------------------------------------------------

## 🔐 Responsible Use

Employee attrition prediction is a **decision-support application**, not
an automated employment decision system.

Predictions should be interpreted carefully because employee-related
data can contain bias, historical patterns, privacy concerns, and
contextual factors that are not captured by the model.

The application itself states that its output should support HR analysis
rather than replace human judgment or employment decisions.

------------------------------------------------------------------------

## 📊 Evaluation Metrics

### Accuracy

Measures the proportion of total predictions that are correct.

### Precision

Measures how many employees predicted as attrition actually belong to
the attrition class.

### Recall

Measures how many actual attrition cases are successfully identified.

### F1-score

Provides a balance between precision and recall and is particularly
useful when both false positives and false negatives matter.

### ROC-AUC

Measures the model's ability to distinguish between attrition and
non-attrition cases across classification thresholds.

For this project, **F1-score and ROC-AUC are especially useful alongside
accuracy**, because the business objective is to identify potential
attrition cases rather than simply maximize overall correctness.

------------------------------------------------------------------------

## 🧰 Technologies

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Core programming language

  Pandas                              Data manipulation

  NumPy                               Numerical operations

  Scikit-learn                        Preprocessing, pipelines,
                                      validation, metrics, baseline
                                      models

  XGBoost                             Final boosting classifier

  Matplotlib                          Visualization

  Seaborn                             Statistical visualization

  Joblib                              Model serialization

  Streamlit                           Web application

  Jupyter Notebook                    Experimentation and documentation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 📌 Important Files

### `employee_attrition_prediction.ipynb`

Complete project notebook containing the end-to-end workflow, EDA, model
comparison, tuning, evaluation, and final model selection.

### `train_model.py`

Training script that loads the Excel dataset, builds the preprocessing +
XGBoost pipeline, performs GridSearchCV, evaluates the model, and saves
the trained estimator.

### `app.py`

Streamlit application for interactive employee attrition prediction.

### `employee_attrition.xlsx`

User-provided employee attrition dataset used by the project.

### `requirements.txt`

Python dependency list required to run the project.

------------------------------------------------------------------------

## 🧪 Reproducibility

The project uses:

``` python
RANDOM_STATE = 42
```

The train-test split is stratified, and the hyperparameter search uses
`StratifiedKFold` with 5 folds.

This makes the training and validation process substantially more
reproducible across runs, subject to library versions and execution
environment.

------------------------------------------------------------------------

## 🔮 Future Improvements

Potential extensions include:

-   Probability-threshold optimization based on HR objectives
-   SHAP-based model explainability
-   Feature importance dashboard
-   Calibration analysis
-   Bias and fairness analysis
-   Cross-validation performance reporting
-   Automated experiment tracking
-   Docker containerization
-   Cloud deployment
-   CI/CD pipeline
-   Batch employee-risk prediction
-   Secure authentication and access control
-   Model monitoring and drift detection

------------------------------------------------------------------------

## 👨‍💻 Project Status

**Status:** ✅ Completed ML Prototype + Streamlit Deployment

**Model:** Tuned XGBoost Classifier

**Task:** Binary Employee Attrition Classification

**Deployment:** Streamlit

**Model Format:** Joblib

------------------------------------------------------------------------

## ⭐ Project Highlights

``` text
✔ 1,191 employee records
✔ 35 columns
✔ Binary classification
✔ 4 baseline models
✔ XGBoost selected as strongest baseline
✔ 5-fold cross-validation
✔ 768 XGBoost parameter combinations
✔ 3,840 total CV fits
✔ Tuned XGBoost
✔ 83% recorded test accuracy
✔ 79% recorded test F1
✔ 89% recorded test ROC-AUC
✔ Joblib model persistence
✔ Interactive Streamlit application
```

------------------------------------------------------------------------

## 📜 Disclaimer

This repository is intended for **educational, portfolio, and
machine-learning demonstration purposes**. Model predictions are
estimates based on the supplied dataset and should not be interpreted as
definitive judgments about an individual employee.
