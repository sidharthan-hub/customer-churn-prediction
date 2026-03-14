import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("data/churn.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns)

# 2. Data cleaning
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 3. Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# 4. Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 5. Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Train model
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 9. Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Risk categorization
results = X_test.copy()
results["ActualChurn"] = y_test.values
results["PredictedChurn"] = y_pred
results["RiskScore"] = y_prob

def risk_category(score):
    if score >= 0.8:
        return "Critical Risk"
    elif score >= 0.5:
        return "Moderate Risk"
    else:
        return "Low Risk"

results["RiskCategory"] = results["RiskScore"].apply(risk_category)

print("\nSample Risk Predictions:")
print(results[["RiskScore", "RiskCategory"]].head(10))

# 11. Save outputs
results.to_csv("data/churn_predictions.csv", index=False)
joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")
print("Predictions saved as data/churn_predictions.csv")
