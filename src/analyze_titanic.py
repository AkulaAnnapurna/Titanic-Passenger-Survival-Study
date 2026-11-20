from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "titanic.csv"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\nDownload the Titanic training CSV, "
            "rename it to titanic.csv, and place it in the data folder."
        )

    OUT.mkdir(exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)

    required = {"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    print("\n--- DATASET OVERVIEW ---")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(df.head())
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    # Work on a copy; fill numeric missing values with medians and Embarked with mode.
    clean = df.copy()
    clean["Age"] = clean["Age"].fillna(clean["Age"].median())
    clean["Fare"] = clean["Fare"].fillna(clean["Fare"].median())
    if clean["Embarked"].isnull().any():
        clean["Embarked"] = clean["Embarked"].fillna(clean["Embarked"].mode()[0])
    clean["FamilySize"] = clean["SibSp"] + clean["Parch"] + 1
    clean["IsAlone"] = (clean["FamilySize"] == 1).astype(int)
    clean.to_csv(OUT / "titanic_cleaned.csv", index=False)

    print("\n--- SURVIVAL SUMMARY ---")
    survival = clean["Survived"].value_counts().sort_index()
    survival_summary = pd.DataFrame({"status": ["Did not survive", "Survived"], "passengers": [int(survival.get(0, 0)), int(survival.get(1, 0))]})
    survival_summary["percentage"] = survival_summary["passengers"] / len(clean) * 100
    print(survival_summary.to_string(index=False))
    survival_summary.to_csv(OUT / "survival_summary.csv", index=False)

    by_sex = clean.groupby("Sex", observed=False)["Survived"].agg(["count", "mean"]).reset_index()
    by_sex["survival_percent"] = by_sex["mean"] * 100
    by_sex.to_csv(OUT / "survival_by_sex.csv", index=False)
    by_class = clean.groupby("Pclass")["Survived"].agg(["count", "mean"]).reset_index()
    by_class["survival_percent"] = by_class["mean"] * 100
    by_class.to_csv(OUT / "survival_by_class.csv", index=False)
    print("\nSurvival rate by sex (%):\n", by_sex[["Sex", "survival_percent"]].to_string(index=False))
    print("\nSurvival rate by passenger class (%):\n", by_class[["Pclass", "survival_percent"]].to_string(index=False))

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 5))
    sns.countplot(data=clean, x="Survived")
    plt.xticks([0, 1], ["Did not survive", "Survived"])
    plt.title("Titanic Passenger Survival Count")
    plt.xlabel("Survival status")
    plt.ylabel("Number of passengers")
    plt.tight_layout(); plt.savefig(CHARTS / "01_survival_count.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 5))
    sns.barplot(data=clean, x="Sex", y="Survived", errorbar=None)
    plt.title("Survival Rate by Sex"); plt.ylabel("Survival rate")
    plt.tight_layout(); plt.savefig(CHARTS / "02_survival_by_sex.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 5))
    sns.barplot(data=clean, x="Pclass", y="Survived", errorbar=None)
    plt.title("Survival Rate by Passenger Class"); plt.xlabel("Ticket class"); plt.ylabel("Survival rate")
    plt.tight_layout(); plt.savefig(CHARTS / "03_survival_by_class.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=clean, x="Age", hue="Survived", bins=25, multiple="stack")
    plt.title("Passenger Age Distribution by Survival")
    plt.tight_layout(); plt.savefig(CHARTS / "04_age_distribution.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=clean, x="Survived", y="Fare")
    plt.xticks([0, 1], ["Did not survive", "Survived"])
    plt.title("Ticket Fare by Survival Status")
    plt.tight_layout(); plt.savefig(CHARTS / "05_fare_by_survival.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=clean, x="FamilySize", y="Survived", errorbar=None)
    plt.title("Survival Rate by Family Size"); plt.xlabel("Family size (including passenger)"); plt.ylabel("Survival rate")
    plt.tight_layout(); plt.savefig(CHARTS / "06_family_size_survival.png", dpi=150); plt.close()

    numeric = clean.select_dtypes(include=np.number)
    plt.figure(figsize=(9, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout(); plt.savefig(CHARTS / "07_correlation_heatmap.png", dpi=150); plt.close()

    print(f"\nAnalysis completed. Cleaned data and summaries saved in: {OUT}")
    print(f"Charts saved in: {CHARTS}")


if __name__ == "__main__":
    main()
