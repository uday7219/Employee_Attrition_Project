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
  Model             Accuracy    Precision       Recall           F1      ROC-AUC
  ------------- ------------ ------------ ------------ ------------ ------------
  **XGBoost**     **82.01%**   **81.25%**   **75.73%**   **78.39%**   **90.16%**

  Logistic            79.08%       77.32%       72.82%       75.00%       88.61%
  Regression                                                        

  Random Forest       77.82%       73.58%       75.73%       74.64%       86.21%

  Decision Tree       75.31%       73.91%       66.02%       69.74%       77.76%
  ------------------------------------------------------------------------------

**Best baseline:** XGBoost

------------------------------------------------------------------------

## 🚀 XGBoost Hyperparameter Tuning

The selected XGBoost pipeline was optimized using **GridSearchCV** with
**5-fold Stratified Cross-Validation**.

### Search Space

-   `n_estimators`: 200, 300, 400
-   `learning_rate`: 0.05, 0.1
-   `max_depth`: 3, 5, 7
-   `min_child_weight`: 1, 2, 4
-   `subsample`: 0.8, 1.0
-   `colsample_bytree`: 0.8, 1.0
-   `gamma`: 0, 0.2

This produced **432 parameter combinations × 5 folds = 2,160 fits**.

### Best Parameters

``` text
colsample_bytree = 0.8
gamma             = 0
learning_rate     = 0.05
max_depth         = 3
min_child_weight  = 2
n_estimators      = 400
subsample         = 0.8
```

**Best cross-validation F1:** `0.7876`

------------------------------------------------------------------------

## 📈 Final Tuned Model Performance

The tuned XGBoost model was evaluated on the held-out test set.

  Metric          Score
  ----------- ---------
  Accuracy      **83%**
  Precision     **83%**
  Recall        **77%**
  F1-score      **80%**
  ROC-AUC       **91%**

### Classification Performance

  Class            Precision   Recall   F1-score
  -------------- ----------- -------- ----------
  No Attrition          0.83     0.88       0.86
  Attrition             0.83     0.77       0.80

The tuned model improved the baseline XGBoost result from approximately
**78.39% F1 / 90.16% ROC-AUC** to **80% F1 / 91% ROC-AUC** on the
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
├── employee_attrition_prediction(3).ipynb
├── train_model(2).py
├── app(1).py
├── requirements(1).txt
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
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Employee-Attrition-Prediction
```

### 2. Create a virtual environment

#### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

``` bash
python3 -m venv venv
source venv/bin/activate
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

### `employee_attrition_prediction(3).ipynb`

Complete project notebook containing the end-to-end workflow, EDA, model
comparison, tuning, evaluation, and final model selection.

### `train_model(2).py`

Training script that loads the Excel dataset, builds the preprocessing +
XGBoost pipeline, performs GridSearchCV, evaluates the model, and saves
the trained estimator.

### `app(1).py`

Streamlit application for interactive employee attrition prediction.

### `employee_attrition.xlsx`

User-provided employee attrition dataset used by the project.

### `requirements(1).txt`

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
✔ 432 XGBoost parameter combinations
✔ 2,160 total CV fits
✔ Tuned XGBoost
✔ 83% recorded test accuracy
✔ 80% recorded test F1
✔ 91% recorded test ROC-AUC
✔ Joblib model persistence
✔ Interactive Streamlit application
```

------------------------------------------------------------------------

## 📜 Disclaimer

This repository is intended for **educational, portfolio, and
machine-learning demonstration purposes**. Model predictions are
estimates based on the supplied dataset and should not be interpreted as
definitive judgments about an individual employee.

------------------------------------------------------------------------

## ⭐ If You Found This Project Useful

Consider giving the repository a ⭐ on GitHub and using the project as a
foundation for further experimentation in **HR analytics,
classification, explainable AI, and responsible machine learning**.
