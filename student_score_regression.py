import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import statsmodels.api as sm

# Load data
DATA_PATH = "student_performance_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Data cleaning
df = df.drop(columns=["student_id", "final_grade", "parental_education"])

# Check and remove exact duplicate rows
n_before = len(df)
df = df.drop_duplicates() #clear dupes
n_after = len(df)
print(f"Removed {n_before - n_after} duplicate rows ({n_after} rows remain)")

# Check missing values across all columns
print("\nMissing values per column:")
print(df.isnull().sum())

# Sanity-check plausible ranges for numeric columns
range_checks = {
    "study_time_hours": (0, 24),
    "attendance_percent": (0, 100),
    "sleep_hours": (0, 24),
    "previous_grade": (0, 100),
    "final_exam_score": (0, 100),
}
for col, (low, high) in range_checks.items():
    out_of_range = df[(df[col] < low) | (df[col] > high)]
    if len(out_of_range) > 0:
        print(f"\nWARNING: {len(out_of_range)} rows in '{col}' fall outside [{low}, {high}]")

# Encode categorical variables
binary_map = {"Yes": 1, "No": 0}
binary_map_gender = {"Male": 1, "Female": 0}
binary_cols = ["internet_access", "extracurricular_activities", "part_time_job"]
for col in binary_cols:
    df[col] = df[col].map(binary_map)
df["gender"] = df["gender"].map(binary_map_gender)

# Save State of Data
df.to_csv("student_performance_cleaned.csv", index=False)


# Train/validation split (80/20, fixed seed for reproducibility)
X = df.drop(columns=["final_exam_score"])
y = df["final_exam_score"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scatterplots with regression line
numeric_features = ["study_time_hours", "attendance_percent", "sleep_hours", "previous_grade"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, feat in zip(axes.flatten(), numeric_features):
    sns.regplot(x=df[feat], y=df["final_exam_score"], ax=ax,
                scatter_kws={"alpha": 0.4, "s": 10}, line_kws={"color": "red"})
    ax.set_xlabel(feat)
    ax.set_ylabel("final_exam_score")
    ax.set_title(f"{feat} vs. final_exam_score")

plt.tight_layout()
plt.savefig("feature_scatterplots.png", dpi=150)
print("\nSaved scatterplots to feature_scatterplots.png")

# Box plots
categorical_features = ["gender", "internet_access", "extracurricular_activities",
                         "part_time_job"]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, feat in zip(axes.flatten(), categorical_features):
    df.boxplot(column="final_exam_score", by=feat, ax=ax)
    ax.set_xlabel(feat)
    ax.set_ylabel("final_exam_score")
    ax.set_title(f"final_exam_score by {feat}")

for ax in axes.flatten()[len(categorical_features):]:
    ax.axis("off")
plt.suptitle("")  # remove default pandas title
plt.tight_layout()
plt.savefig("feature_boxplots.png", dpi=150)
print("\nSaved box plots to feature_boxplots.png")
