# ============================================================
# DECODELABS - DATA SCIENCE PROJECT 1
# Advanced EDA & Feature Engineering
# Batch: 2026
# Dataset: Titanic Dataset
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

print("=" * 70)
print("DECODELABS - DATA SCIENCE PROJECT 1")
print("Advanced EDA & Feature Engineering")
print("=" * 70)


# ============================================================
# 2. LOAD DATASET
# ============================================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

print("\nDataset loaded successfully!")
print("Dataset Shape:", df.shape)


# ============================================================
# 3. BASIC EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("BASIC EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 4. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE VALUE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")


# ============================================================
# 5. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

print("\nMissing values in each column:")
print(missing_values)

print("\nMissing value percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.round(2))


# ============================================================
# 6. VISUALIZE MISSING VALUES
# ============================================================

plt.figure(figsize=(12, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title("Missing Values Heatmap")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE IMPUTATION")
print("=" * 70)

# Numerical column: Median Imputation
# Median is useful because it is less affected by extreme values.

age_median = df["Age"].median()

df["Age"] = df["Age"].fillna(age_median)

print("Missing Age values replaced using Median.")
print("Age Median:", age_median)


# Categorical column: Mode Imputation

embarked_mode = df["Embarked"].mode()[0]

df["Embarked"] = df["Embarked"].fillna(embarked_mode)

print("Missing Embarked values replaced using Mode.")
print("Embarked Mode:", embarked_mode)


# Cabin has many missing values.
# Instead of replacing cabin numbers with artificial values,
# create a feature indicating whether cabin information exists.

df["Has_Cabin"] = df["Cabin"].notna().astype(int)

print("Created 'Has_Cabin' feature.")


# Drop original Cabin column

df.drop("Cabin", axis=1, inplace=True)

print("Original 'Cabin' column removed.")


# Check missing values again

print("\nMissing values after imputation:")
print(df.isnull().sum())


# ============================================================
# 8. BASIC VISUALIZATIONS
# ============================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA VISUALIZATION")
print("=" * 70)


# Survival Distribution

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Survived"
)

plt.title("Passenger Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()


# Age Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    bins=30,
    kde=True,
    color="steelblue"
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# Fare Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Fare",
    bins=30,
    kde=True,
    color="green"
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()


# Survival by Gender

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Sex",
    hue="Survived"
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()


# Survival by Passenger Class

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Pclass",
    hue="Survived"
)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()


# ============================================================
# 9. OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER DETECTION USING IQR")
print("=" * 70)


def detect_iqr_outliers(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    print("\nColumn:", column)
    print("Q1:", round(Q1, 2))
    print("Q3:", round(Q3, 2))
    print("IQR:", round(IQR, 2))
    print("Lower Bound:", round(lower_bound, 2))
    print("Upper Bound:", round(upper_bound, 2))
    print("Number of Outliers:", len(outliers))

    return lower_bound, upper_bound


age_lower, age_upper = detect_iqr_outliers(df, "Age")

fare_lower, fare_upper = detect_iqr_outliers(df, "Fare")


# ============================================================
# 10. VISUALIZE OUTLIERS BEFORE TREATMENT
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

sns.boxplot(
    y=df["Age"],
    color="orange"
)

plt.title("Age - Before Outlier Treatment")


plt.subplot(1, 2, 2)

sns.boxplot(
    y=df["Fare"],
    color="lightgreen"
)

plt.title("Fare - Before Outlier Treatment")

plt.tight_layout()
plt.show()


# ============================================================
# 11. OUTLIER TREATMENT USING IQR CLIPPING
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER TREATMENT")
print("=" * 70)

# Clip extreme Age values

df["Age"] = df["Age"].clip(
    lower=age_lower,
    upper=age_upper
)

# Clip extreme Fare values

df["Fare"] = df["Fare"].clip(
    lower=fare_lower,
    upper=fare_upper
)

print("Age outliers treated using IQR clipping.")
print("Fare outliers treated using IQR clipping.")


# ============================================================
# 12. VERIFY OUTLIER TREATMENT
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

sns.boxplot(
    y=df["Age"],
    color="orange"
)

plt.title("Age - After Outlier Treatment")


plt.subplot(1, 2, 2)

sns.boxplot(
    y=df["Fare"],
    color="lightgreen"
)

plt.title("Fare - After Outlier Treatment")

plt.tight_layout()
plt.show()


# ============================================================
# 13. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# Feature 1: Family Size
# SibSp = siblings/spouses
# Parch = parents/children

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("Feature 1 created: FamilySize")


# Feature 2: Is Alone
# 1 = Passenger travelling alone
# 0 = Passenger travelling with family

df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

print("Feature 2 created: IsAlone")


# Feature 3: Fare Per Person
# This calculates the approximate fare per family member.

df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

print("Feature 3 created: FarePerPerson")


# Feature 4: Age Group

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[
        "Child",
        "Teenager",
        "Young Adult",
        "Adult",
        "Senior"
    ]
)

print("Feature 4 created: AgeGroup")


# Feature 5: Family Fare

df["FamilyFare"] = df["Fare"] * df["FamilySize"]

print("Feature 5 created: FamilyFare")


# Display engineered features

print("\nNewly engineered features:")

print(
    df[
        [
            "FamilySize",
            "IsAlone",
            "FarePerPerson",
            "AgeGroup",
            "FamilyFare"
        ]
    ].head(10)
)


# ============================================================
# 14. ANALYZE ENGINEERED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS OF ENGINEERED FEATURES")
print("=" * 70)


# Family Size vs Survival

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="FamilySize",
    hue="Survived"
)

plt.title("Survival by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Number of Passengers")
plt.show()


# Alone vs Survival

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="IsAlone",
    hue="Survived"
)

plt.title("Survival: Alone vs With Family")
plt.xlabel("Is Alone (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()


# Age Group vs Survival

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="AgeGroup",
    hue="Survived"
)

plt.title("Survival by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=30)
plt.show()


# ============================================================
# 15. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

correlation_matrix = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))


plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# 16. FINAL DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Column Names:")
print(df.columns.tolist())

print("\nRemaining Missing Values:")
print(df.isnull().sum())

print("\nRemaining Duplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 17. FINAL DATASET PREVIEW
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANED DATASET")
print("=" * 70)

print(df.head(10))


# ============================================================
# 18. SAVE CLEANED DATASET
# ============================================================

output_file = "titanic_cleaned_feature_engineered.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCleaned dataset saved as:")
print(output_file)

print("\nFinal dataset shape:", df.shape)

print("\nFeatures created:")
print("1. Has_Cabin")
print("2. FamilySize")
print("3. IsAlone")
print("4. FarePerPerson")
print("5. AgeGroup")
print("6. FamilyFare")

print("\nTechniques used:")
print("- Exploratory Data Analysis (EDA)")
print("- Median Imputation")
print("- Mode Imputation")
print("- IQR Outlier Detection")
print("- IQR Outlier Treatment")
print("- Feature Engineering")
print("- Correlation Analysis")
print("- Data Visualization")

print("\nThank you!")
