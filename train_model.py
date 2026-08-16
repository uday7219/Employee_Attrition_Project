import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE=42
TARGET="Attrition"

df = pd.read_excel("employee_attrition.xlsx")

# Normalize columns and text values.
df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.replace("\xa0", " ", regex=False).str.strip()

df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=[TARGET, "ID"])
y = df[TARGET].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

param = {
    "model__n_estimators": [200, 300, 400],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5, 7],
    "model__min_child_weight": [1, 2, 4],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__gamma": [0, 0.2]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = GridSearchCV(
    pipeline,
    param_grid=param,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

pred = search.best_estimator_.predict(X_test)
prob = search.best_estimator_.predict_proba(X_test)[:, 1]

print("\nFinal Test Metrics")
print("------------------")
print("Accuracy :", round(accuracy_score(y_test, pred)*100, 2))
print("Precision:", round(precision_score(y_test, pred, zero_division=0)*100, 2))
print("Recall   :", round(recall_score(y_test, pred, zero_division=0)*100, 2))
print("F1       :", round(f1_score(y_test, pred, zero_division=0)*100, 2))
print("ROC-AUC  :", round(roc_auc_score(y_test, prob)*100, 2))

joblib.dump(search.best_estimator_, "models/employee_attrition_model.joblib")

print("\nBest parameters:")
print(search.best_params_)
print(f"\nSaved model: {"models/employee_attrition_model.joblib"}")
