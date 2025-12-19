import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # project folder
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"



# 1) load
csv_path = DATA_DIR / "Stress Dataset.csv"
df = pd.read_csv(csv_path)



# 2) target (exact from your printout)
target_col = "Have you recently experienced stress in your life?"

# Encode target labels as integers 0..(n_classes-1) for XGBoost
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col].astype(str))
print("Encoded classes:", le.classes_)  # optional, for you to see the mapping

# 3) feature types
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in num_cols:
    num_cols.remove(target_col)
cat_cols = [c for c in df.columns if c not in num_cols + [target_col]]

# 4) preprocess
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

# 5) split
X = df.drop(columns=[target_col])
y = df[target_col]   # already integer-encoded

Xtr, Xte, ytr, yte = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() <= 50 else None
)


# 6) models
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        multi_class="ovr"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        learning_rate=0.1,
        n_estimators=300,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}




# 7) train + evaluate
rows = []
trained_models = {}

for name, mdl in models.items():
    pipe = Pipeline([
        ("pp", preprocess),
        ("m", mdl)
    ])
    
    # Train model
    pipe.fit(Xtr, ytr)
    trained_models[name] = pipe   # save trained model
    
    # Predictions
    y_pred = pipe.predict(Xte)
    
    try:
        y_proba = pipe.predict_proba(Xte)
        auc = roc_auc_score(
            yte,
            y_proba,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        auc = float("nan")
    
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(yte, y_pred),
        "Macro F1": f1_score(yte, y_pred, average="macro"),
        "ROC-AUC OvR": auc
    })

# ==== RESULTS DATAFRAME ====
results_df = pd.DataFrame(rows)
print("\n=== Model Performance ===")
print(results_df)

# Ensure folder exists
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Save metrics
results_df.to_csv(RESULTS_DIR / "model_metrics_confirmed_target.csv", index=False)
print(f"Saved -> {RESULTS_DIR / 'model_metrics_confirmed_target.csv'}")
# ==== SELECT BEST MODEL ====
best_model_name = results_df.sort_values("Macro F1", ascending=False).iloc[0]["Model"]
print("\nBest model:", best_model_name)

best_model = trained_models[best_model_name]

# ==== CONFUSION MATRIX ====
y_pred_best = best_model.predict(Xte)
cm = confusion_matrix(yte, y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig(RESULTS_DIR / "confusion_matrix_best_model.png", dpi=300, bbox_inches="tight")

plt.close()

print("Saved -> results/confusion_matrix_best_model.png")


# ==== FEATURE IMPORTANCE ====
# We will compute feature importances for RandomForest and XGBoost
# Using the fitted preprocessing pipeline to get the transformed feature names

# Get feature names from the preprocessing step (from any fitted pipeline)
# Use the best_model's preprocessor since it is already fitted
preprocessor_fitted = best_model.named_steps["pp"]
feature_names = preprocessor_fitted.get_feature_names_out()

# ---- Random Forest feature importance ----
if "RandomForest" in trained_models:
    rf_pipe = trained_models["RandomForest"]
    rf_model = rf_pipe.named_steps["m"]

    try:
        rf_importances = rf_model.feature_importances_
        # Take top 10
        idx_rf = np.argsort(rf_importances)[-10:]
        top_rf_importances = rf_importances[idx_rf]
        top_rf_features = feature_names[idx_rf]

        plt.figure()
        plt.barh(range(len(idx_rf)), top_rf_importances)
        plt.yticks(range(len(idx_rf)), top_rf_features)
        plt.xlabel("Importance")
        plt.title("Top 10 Feature Importances - Random Forest")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_importances_randomforest.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved -> results/feature_importances_randomforest.png")
    except AttributeError:
        print("RandomForest model does not provide feature_importances_")

# ---- XGBoost feature importance ----
if "XGBoost" in trained_models:
    xgb_pipe = trained_models["XGBoost"]
    xgb_model = xgb_pipe.named_steps["m"]

    try:
        xgb_importances = xgb_model.feature_importances_
        idx_xgb = np.argsort(xgb_importances)[-10:]
        top_xgb_importances = xgb_importances[idx_xgb]
        top_xgb_features = feature_names[idx_xgb]

        plt.figure()
        plt.barh(range(len(idx_xgb)), top_xgb_importances)
        plt.yticks(range(len(idx_xgb)), top_xgb_features)
        plt.xlabel("Importance")
        plt.title("Top 10 Feature Importances - XGBoost")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_importances_xgboost.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved -> results/feature_importances_xgboost.png")
    except AttributeError:
        print("XGBoost model does not provide feature_importances_")

# ==== LIGHT HYPERPARAMETER TUNING (RF & XGB) ====
print("\n=== Hyperparameter tuning (RandomizedSearchCV) ===")

# --- Random Forest tuning ---
if "RandomForest" in trained_models:
    rf_pipe = Pipeline([
        ("pp", preprocess),
        ("m", RandomForestClassifier(random_state=42))
    ])

    rf_param_dist = {
        "m__n_estimators": [200, 300, 500],
        "m__max_depth": [None, 5, 10],
        "m__max_features": ["sqrt", "log2"]
    }

    rf_search = RandomizedSearchCV(
        rf_pipe,
        rf_param_dist,
        n_iter=5,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rf_search.fit(Xtr, ytr)
    print("Best RF params:", rf_search.best_params_)

    y_pred_rf_tuned = rf_search.best_estimator_.predict(Xte)
    try:
        proba_rf_tuned = rf_search.best_estimator_.predict_proba(Xte)
        auc_rf_tuned = roc_auc_score(
            yte,
            proba_rf_tuned,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        auc_rf_tuned = float("nan")

    rows.append({
        "Model": "RandomForest_tuned",
        "Accuracy": accuracy_score(yte, y_pred_rf_tuned),
        "Macro F1": f1_score(yte, y_pred_rf_tuned, average="macro"),
        "ROC-AUC OvR": auc_rf_tuned
    })

# --- XGBoost tuning ---
if "XGBoost" in trained_models:
    xgb_pipe = Pipeline([
        ("pp", preprocess),
        ("m", XGBClassifier(
            objective="multi:softprob",
            num_class=len(le.classes_),
            eval_metric="mlogloss",
            random_state=42
        ))
    ])

    xgb_param_dist = {
        "m__n_estimators": [200, 300, 400],
        "m__max_depth": [3, 4, 5],
        "m__learning_rate": [0.05, 0.1, 0.2],
        "m__subsample": [0.7, 0.8, 1.0],
        "m__colsample_bytree": [0.7, 0.8, 1.0]
    }

    xgb_search = RandomizedSearchCV(
        xgb_pipe,
        xgb_param_dist,
        n_iter=6,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    xgb_search.fit(Xtr, ytr)
    print("Best XGB params:", xgb_search.best_params_)

    y_pred_xgb_tuned = xgb_search.best_estimator_.predict(Xte)
    proba_xgb_tuned = xgb_search.best_estimator_.predict_proba(Xte)
    auc_xgb_tuned = roc_auc_score(
        yte,
        proba_xgb_tuned,
        multi_class="ovr",
        average="macro"
    )

    rows.append({
        "Model": "XGBoost_tuned",
        "Accuracy": accuracy_score(yte, y_pred_xgb_tuned),
        "Macro F1": f1_score(yte, y_pred_xgb_tuned, average="macro"),
        "ROC-AUC OvR": auc_xgb_tuned
    })

# Rebuild and save updated results table with tuned models
results_df = pd.DataFrame(rows)
print("\n=== Model Performance (including tuned models) ===")
print(results_df)

os.makedirs("results", exist_ok=True)
results_df.to_csv(RESULTS_DIR / "model_metrics_with_tuning.csv", index=False)

print("Saved -> results/model_metrics_with_tuning.csv")


#visualizations 
# 1) class distribution
df[target_col].value_counts().plot(kind="bar", color="skyblue")
plt.title("Have you recently experienced stress in your life?")
plt.xlabel("Response")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()

# 2) numeric correlations
# 2) numeric correlations (clean summary instead of messy heatmap)
num_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[num_cols].corr().abs().unstack().sort_values(ascending=False)

# Filter: only strong correlations (0.45 < corr < 1.0)
top_corr = corr_matrix[(corr_matrix < 1.0) & (corr_matrix > 0.45)].dropna().head(15)

# Reset for readability
top_corr = top_corr.reset_index()
top_corr.columns = ["Variable 1", "Variable 2", "Correlation"]

print("\nTop 15 strongest correlations:")
print(top_corr)

# Save as CSV if you want to include in appendix/report
top_corr.to_csv("top_correlations.csv", index=False)
# 3) categorical examples
sns.countplot(data=df, x="Gender", hue=target_col)
plt.title("Stress frequency by Gender")
plt.tight_layout()
plt.savefig("stress_by_gender.png", dpi=300)
plt.show()

sns.countplot(data=df, x="Do you face any sleep problems or difficulties falling asleep?", hue=target_col)
plt.title("Stress frequency vs. Sleep problems")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("stress_by_sleep.png", dpi=300)
plt.show()

# Stress vs Gender
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Gender", hue=target_col, palette="coolwarm")
plt.title("Stress Levels by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Stress Level", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("stress_by_gender.png", dpi=300)

corr_matrix = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
top_corr = corr_matrix[(corr_matrix < 1.0) & (corr_matrix > 0.4)].dropna().head(15)
print(top_corr)
plt.show()
# Stress vs Sleep Problems
sleep_col = "Do you face any sleep problems or difficulties falling asleep?"
plt.figure(figsize=(7,4))
sns.countplot(data=df, x=sleep_col, hue=target_col, palette="mako")
plt.title("Stress vs Sleep Difficulties")
plt.xlabel("Sleep Problems (Scale 1–5)")
plt.ylabel("Count")
plt.legend(title="Stress Level", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("stress_by_sleep.png", dpi=300)
plt.show()

# Top correlations table. 
# Recalculate correlations with a slightly lower threshold
corr_matrix = df[num_cols].corr().abs().unstack().sort_values(ascending=False)
top_corr = corr_matrix[(corr_matrix < 1.0) & (corr_matrix > 0.35)].dropna().head(15)

# Reset for readability
top_corr = top_corr.reset_index()
top_corr.columns = ["Variable 1", "Variable 2", "Correlation"]

# Print in terminal for confirmation
print("\n=== Top 15 Correlations (|r| > 0.35) ===")
print(top_corr)

# Save table as CSV
top_corr.to_csv("top_correlations.csv", index=False)

# --- Plot the correlations ---
plt.figure(figsize=(8, 5))
sns.barplot(
    data=top_corr,
    x="Correlation",
    y="Variable 1",
    orient="h",
    palette="viridis"
)
plt.title("Top 15 Correlated Variable Pairs")
plt.xlabel("Correlation Strength (|r|)")
plt.ylabel("Variable 1")
plt.tight_layout()
plt.savefig("top_correlations_chart.png", dpi=300)
plt.show()
