# ============================================================
# DECODELABS - DATA SCIENCE PROJECT 3
# UNSUPERVISED LEARNING - CUSTOMER SEGMENTATION
# BATCH 2026
#
# SELF-CONTAINED IDLE VERSION
# NO EXTERNAL DATASET REQUIRED
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

np.random.seed(42)

sns.set_style("whitegrid")

print("=" * 80)
print("DECODELABS - DATA SCIENCE PROJECT 3")
print("UNSUPERVISED LEARNING - CUSTOMER SEGMENTATION")
print("=" * 80)


# ============================================================
# 1. GENERATE SYNTHETIC CUSTOMER DATASET
# ============================================================

print("\nGenerating customer dataset...")

n_customers = 1200

# Customer ID
customer_id = np.arange(10001, 10001 + n_customers)

# Age
age = np.random.randint(20, 71, n_customers)

# Income
income = np.random.normal(
    60000,
    18000,
    n_customers
)

income = np.clip(
    income,
    20000,
    150000
)

# Years as customer
customer_years = np.random.randint(
    1,
    16,
    n_customers
)

# Website visits
web_visits = np.random.poisson(
    6,
    n_customers
) + 1

# Purchases
web_purchases = np.random.poisson(
    5,
    n_customers
)

store_purchases = np.random.poisson(
    7,
    n_customers
)

catalog_purchases = np.random.poisson(
    3,
    n_customers
)

# Product spending
wine_spending = np.random.gamma(
    3,
    45,
    n_customers
)

fruit_spending = np.random.gamma(
    2,
    20,
    n_customers
)

meat_spending = np.random.gamma(
    3,
    60,
    n_customers
)

fish_spending = np.random.gamma(
    2,
    25,
    n_customers
)

sweet_spending = np.random.gamma(
    2,
    20,
    n_customers
)

gold_spending = np.random.gamma(
    2,
    30,
    n_customers
)

# Household information
children = np.random.randint(
    0,
    4,
    n_customers
)

teenagers = np.random.randint(
    0,
    3,
    n_customers
)

# Marketing campaign responses
campaign1 = np.random.binomial(
    1,
    0.18,
    n_customers
)

campaign2 = np.random.binomial(
    1,
    0.15,
    n_customers
)

campaign3 = np.random.binomial(
    1,
    0.12,
    n_customers
)

campaign4 = np.random.binomial(
    1,
    0.10,
    n_customers
)

campaign5 = np.random.binomial(
    1,
    0.08,
    n_customers
)

# Website and catalog activity
pages_viewed = np.random.poisson(
    10,
    n_customers
) + 1

time_on_website = np.random.normal(
    8,
    3,
    n_customers
)

time_on_website = np.clip(
    time_on_website,
    1,
    30
)

# Discounts
discount_purchases = np.random.poisson(
    2,
    n_customers
)

# Complaints
complaints = np.random.poisson(
    0.5,
    n_customers
)

# Satisfaction
satisfaction = np.random.normal(
    7.5,
    1.5,
    n_customers
)

satisfaction = np.clip(
    satisfaction,
    1,
    10
)

# Loyalty points
loyalty_points = np.random.normal(
    500,
    250,
    n_customers
)

loyalty_points = np.clip(
    loyalty_points,
    0,
    2000
)

# Email engagement
email_opens = np.random.poisson(
    8,
    n_customers
)

email_clicks = np.random.poisson(
    3,
    n_customers
)

# Mobile usage
mobile_sessions = np.random.poisson(
    10,
    n_customers
)

# Returns
returns = np.random.poisson(
    1,
    n_customers
)

# Delivery rating
delivery_rating = np.random.normal(
    4.0,
    0.7,
    n_customers
)

delivery_rating = np.clip(
    delivery_rating,
    1,
    5
)

# Gender
gender = np.random.choice(
    ["Male", "Female"],
    n_customers
)

# Education
education = np.random.choice(
    [
        "Graduate",
        "Postgraduate",
        "High School",
        "PhD"
    ],
    n_customers,
    p=[
        0.40,
        0.25,
        0.25,
        0.10
    ]
)

# Marital status
marital_status = np.random.choice(
    [
        "Single",
        "Married",
        "Divorced"
    ],
    n_customers,
    p=[
        0.35,
        0.50,
        0.15
    ]
)


# ============================================================
# 2. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({

    "Customer_ID": customer_id,

    "Age": age,

    "Income": income,

    "Customer_Years": customer_years,

    "Web_Visits": web_visits,

    "Web_Purchases": web_purchases,

    "Store_Purchases": store_purchases,

    "Catalog_Purchases": catalog_purchases,

    "Wine_Spending": wine_spending,

    "Fruit_Spending": fruit_spending,

    "Meat_Spending": meat_spending,

    "Fish_Spending": fish_spending,

    "Sweet_Spending": sweet_spending,

    "Gold_Spending": gold_spending,

    "Children": children,

    "Teenagers": teenagers,

    "Campaign_1": campaign1,

    "Campaign_2": campaign2,

    "Campaign_3": campaign3,

    "Campaign_4": campaign4,

    "Campaign_5": campaign5,

    "Pages_Viewed": pages_viewed,

    "Time_On_Website": time_on_website,

    "Discount_Purchases": discount_purchases,

    "Complaints": complaints,

    "Satisfaction": satisfaction,

    "Loyalty_Points": loyalty_points,

    "Email_Opens": email_opens,

    "Email_Clicks": email_clicks,

    "Mobile_Sessions": mobile_sessions,

    "Returns": returns,

    "Delivery_Rating": delivery_rating,

    "Gender": gender,

    "Education": education,

    "Marital_Status": marital_status
})


print("\nDataset generated successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())


# ============================================================
# 3. INTRODUCE SOME MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("CREATING REALISTIC MISSING VALUES")
print("=" * 80)

missing_columns = [
    "Income",
    "Satisfaction",
    "Loyalty_Points",
    "Time_On_Website"
]

for column in missing_columns:

    random_indices = np.random.choice(
        df.index,
        size=20,
        replace=False
    )

    df.loc[
        random_indices,
        column
    ] = np.nan


print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. BASIC EDA
# ============================================================

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\nDataset information:")

print(df.info())

print("\nStatistical summary:")

print(
    df.describe(
        include="all"
    )
)


# ============================================================
# 5. DUPLICATE CHECK
# ============================================================

duplicate_count = df.duplicated().sum()

print(
    "\nDuplicate rows:",
    duplicate_count
)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print(
        "Duplicates removed."
    )


# ============================================================
# 6. MISSING VALUE IMPUTATION
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUE IMPUTATION")
print("=" * 80)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    if df[column].isnull().sum() > 0:

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )

        print(
            column,
            "filled using median:",
            round(median_value, 2)
        )


# Categorical missing values

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()[0]

        df[column] = df[column].fillna(
            mode_value
        )


print(
    "\nTotal missing values after treatment:",
    df.isnull().sum().sum()
)


# ============================================================
# 7. MISSING VALUE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(14, 6)
)

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title(
    "Missing Values Heatmap"
)

plt.tight_layout()

plt.show()


# ============================================================
# 8. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)


# Feature 1
df["Total_Spending"] = (
    df["Wine_Spending"]
    +
    df["Fruit_Spending"]
    +
    df["Meat_Spending"]
    +
    df["Fish_Spending"]
    +
    df["Sweet_Spending"]
    +
    df["Gold_Spending"]
)

print(
    "1. Total_Spending created"
)


# Feature 2
df["Total_Purchases"] = (
    df["Web_Purchases"]
    +
    df["Store_Purchases"]
    +
    df["Catalog_Purchases"]
)

print(
    "2. Total_Purchases created"
)


# Feature 3
df["Total_Family_Members"] = (
    df["Children"]
    +
    df["Teenagers"]
    + 1
)

print(
    "3. Total_Family_Members created"
)


# Feature 4
df["Total_Campaign_Acceptance"] = (
    df["Campaign_1"]
    +
    df["Campaign_2"]
    +
    df["Campaign_3"]
    +
    df["Campaign_4"]
    +
    df["Campaign_5"]
)

print(
    "4. Total_Campaign_Acceptance created"
)


# Feature 5
df["Digital_Engagement"] = (
    df["Web_Visits"]
    +
    df["Pages_Viewed"]
    +
    df["Email_Opens"]
    +
    df["Email_Clicks"]
    +
    df["Mobile_Sessions"]
)

print(
    "5. Digital_Engagement created"
)


# Feature 6
df["Average_Spending_Per_Purchase"] = (
    df["Total_Spending"]
    /
    (df["Total_Purchases"] + 1)
)

print(
    "6. Average_Spending_Per_Purchase created"
)


# Feature 7
df["Customer_Value_Score"] = (
    df["Total_Spending"]
    *
    (df["Customer_Years"] + 1)
)

print(
    "7. Customer_Value_Score created"
)


# Feature 8
df["Discount_Rate"] = (
    df["Discount_Purchases"]
    /
    (df["Total_Purchases"] + 1)
)

print(
    "8. Discount_Rate created"
)


# ============================================================
# 9. OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 80)
print("OUTLIER DETECTION USING IQR")
print("=" * 80)

outlier_columns = [
    "Income",
    "Total_Spending",
    "Total_Purchases",
    "Customer_Value_Score"
]

outlier_summary = {}

for column in outlier_columns:

    Q1 = df[column].quantile(
        0.25
    )

    Q3 = df[column].quantile(
        0.75
    )

    IQR = Q3 - Q1

    lower_bound = (
        Q1 - 1.5 * IQR
    )

    upper_bound = (
        Q3 + 1.5 * IQR
    )

    outliers = (
        (df[column] < lower_bound)
        |
        (df[column] > upper_bound)
    )

    outlier_count = outliers.sum()

    outlier_summary[column] = outlier_count

    print(
        column,
        "outliers:",
        outlier_count
    )

    # Winsorization / clipping
    df[column] = df[column].clip(
        lower_bound,
        upper_bound
    )


# ============================================================
# 10. DATA VISUALIZATION
# ============================================================

plt.figure(
    figsize=(12, 6)
)

sns.histplot(
    df["Total_Spending"],
    bins=30,
    kde=True,
    color="blue"
)

plt.title(
    "Distribution of Total Customer Spending"
)

plt.xlabel(
    "Total Spending"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.show()


# ============================================================
# 11. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

print("\n" + "=" * 80)
print("PREPARING DATA FOR MACHINE LEARNING")
print("=" * 80)


# Remove ID because it has no predictive meaning.

model_df = df.drop(
    columns=[
        "Customer_ID"
    ]
)


# One-hot encode categorical variables.

categorical_columns = model_df.select_dtypes(
    include="object"
).columns.tolist()

print(
    "\nCategorical columns:"
)

print(
    categorical_columns
)


model_df = pd.get_dummies(
    model_df,
    columns=categorical_columns,
    drop_first=True
)


# Convert boolean to integer.

for column in model_df.columns:

    if model_df[column].dtype == bool:

        model_df[column] = (
            model_df[column].astype(int)
        )


# Replace infinite values.

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)

model_df = model_df.fillna(
    model_df.median(numeric_only=True)
)

model_df = model_df.fillna(0)


print(
    "\nFinal number of features before PCA:",
    model_df.shape[1]
)


# ============================================================
# 12. STANDARDIZATION
# ============================================================

print("\n" + "=" * 80)
print("STANDARDIZATION")
print("=" * 80)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    model_df
)

print(
    "Standardization completed."
)


# ============================================================
# 13. PCA ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("PCA DIMENSIONALITY REDUCTION")
print("=" * 80)

pca_full = PCA()

X_pca_full = pca_full.fit_transform(
    X_scaled
)

explained_variance = (
    pca_full.explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)


variance_df = pd.DataFrame({

    "Component":
        range(
            1,
            len(explained_variance) + 1
        ),

    "Explained_Variance":
        explained_variance,

    "Cumulative_Variance":
        cumulative_variance
})


print(
    "\nPCA explained variance:"
)

print(
    variance_df.head(10).round(4)
)


# ============================================================
# 14. PCA VARIANCE PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    range(
        1,
        len(cumulative_variance) + 1
    ),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    0.80,
    color="red",
    linestyle="--",
    label="80%"
)

plt.axhline(
    0.90,
    color="green",
    linestyle="--",
    label="90%"
)

plt.xlabel(
    "Number of Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "PCA Explained Variance"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 15. REDUCE TO 3 PCA COMPONENTS
# ============================================================

pca = PCA(
    n_components=3
)

X_pca = pca.fit_transform(
    X_scaled
)

pca_df = pd.DataFrame(
    X_pca,
    columns=[
        "PC1",
        "PC2",
        "PC3"
    ]
)


total_variance = (
    pca.explained_variance_ratio_.sum()
    * 100
)

print(
    "\nOriginal number of features:",
    model_df.shape[1]
)

print(
    "Reduced number of dimensions:",
    X_pca.shape[1]
)

print(
    "Variance explained by 3 components:",
    round(
        total_variance,
        2
    ),
    "%"
)


# ============================================================
# 16. ELBOW METHOD
# ============================================================

print("\n" + "=" * 80)
print("ELBOW METHOD")
print("=" * 80)

k_range = range(
    2,
    11
)

inertia_values = []

for k in k_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(
        X_pca
    )

    inertia_values.append(
        kmeans.inertia_
    )

    print(
        "K =",
        k,
        "Inertia =",
        round(
            kmeans.inertia_,
            2
        )
    )


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    list(k_range),
    inertia_values,
    marker="o",
    linewidth=2
)

plt.xlabel(
    "Number of Clusters"
)

plt.ylabel(
    "Inertia"
)

plt.title(
    "Elbow Method"
)

plt.xticks(
    list(k_range)
)

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 17. SILHOUETTE SCORE
# ============================================================

print("\n" + "=" * 80)
print("SILHOUETTE SCORE")
print("=" * 80)

silhouette_scores = {}

for k in k_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_pca
    )

    score = silhouette_score(
        X_pca,
        labels
    )

    silhouette_scores[k] = score

    print(
        "K =",
        k,
        "Silhouette Score =",
        round(
            score,
            4
        )
    )


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    list(silhouette_scores.keys()),
    list(silhouette_scores.values()),
    marker="o",
    color="purple",
    linewidth=2
)

plt.xlabel(
    "Number of Clusters"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score Analysis"
)

plt.xticks(
    list(k_range)
)

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 18. SELECT BEST K
# ============================================================

optimal_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)

best_score = silhouette_scores[
    optimal_k
]

print("\n" + "=" * 80)
print("OPTIMAL K")
print("=" * 80)

print(
    "Optimal number of clusters:",
    optimal_k
)

print(
    "Best silhouette score:",
    round(
        best_score,
        4
    )
)


# ============================================================
# 19. FINAL K-MEANS MODEL
# ============================================================

print("\n" + "=" * 80)
print("FINAL K-MEANS CLUSTERING")
print("=" * 80)

final_kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=20
)

cluster_labels = final_kmeans.fit_predict(
    X_pca
)

pca_df["Cluster"] = cluster_labels

df["Cluster"] = cluster_labels


print(
    "\nCluster counts:"
)

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 20. 2D PCA CLUSTER VISUALIZATION
# ============================================================

plt.figure(
    figsize=(11, 7)
)

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set2",
    s=60,
    alpha=0.75
)

plt.title(
    "Customer Segmentation using PCA - 2D"
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.legend(
    title="Cluster"
)

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 21. 3D PCA CLUSTER VISUALIZATION
# ============================================================

fig = plt.figure(
    figsize=(12, 9)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

scatter = ax.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    pca_df["PC3"],
    c=pca_df["Cluster"],
    cmap="viridis",
    s=50,
    alpha=0.7
)

ax.set_title(
    "Customer Segmentation using PCA - 3D"
)

ax.set_xlabel(
    "PC1"
)

ax.set_ylabel(
    "PC2"
)

ax.set_zlabel(
    "PC3"
)

plt.colorbar(
    scatter,
    ax=ax,
    label="Cluster"
)

plt.show()


# ============================================================
# 22. CLUSTER PROFILE
# ============================================================

print("\n" + "=" * 80)
print("CLUSTER PROFILING")
print("=" * 80)

profile_columns = [

    "Age",

    "Income",

    "Customer_Years",

    "Total_Spending",

    "Total_Purchases",

    "Total_Family_Members",

    "Total_Campaign_Acceptance",

    "Digital_Engagement",

    "Average_Spending_Per_Purchase",

    "Customer_Value_Score",

    "Discount_Rate",

    "Satisfaction",

    "Loyalty_Points",

    "Complaints",

    "Returns"
]


cluster_profile = (
    df.groupby("Cluster")[
        profile_columns
    ]
    .mean()
)


print(
    "\nAverage characteristics of each cluster:"
)

print(
    cluster_profile.round(2)
)


# ============================================================
# 23. CREATE CUSTOMER PERSONAS
# ============================================================

print("\n" + "=" * 80)
print("CUSTOMER PERSONA CREATION")
print("=" * 80)


# Rank clusters by spending.

spending_rank = (
    cluster_profile[
        "Total_Spending"
    ]
    .rank(
        ascending=False
    )
)


# Rank clusters by purchases.

purchase_rank = (
    cluster_profile[
        "Total_Purchases"
    ]
    .rank(
        ascending=False
    )
)


# Rank clusters by income.

income_rank = (
    cluster_profile[
        "Income"
    ]
    .rank(
        ascending=False
    )
)


# Rank clusters by recency proxy:
# Customer_Years and engagement are used.

engagement_rank = (
    cluster_profile[
        "Digital_Engagement"
    ]
    .rank(
        ascending=False
    )
)


# Overall business value.

business_score = (
    spending_rank
    +
    purchase_rank
    +
    income_rank
    +
    engagement_rank
)


cluster_order = (
    business_score
    .sort_values(
        ascending=False
    )
    .index
    .tolist()
)


# Persona names.

persona_names = [

    "VIP High-Value Customers",

    "Loyal Regular Customers",

    "Potential Growth Customers",

    "Digital Active Customers",

    "Budget-Conscious Customers",

    "Occasional Customers",

    "Low-Value Customers",

    "Emerging Customers",

    "At-Risk Customers",

    "Dormant Customers"
]


persona_mapping = {}


for i, cluster in enumerate(
    cluster_order
):

    if i < len(persona_names):

        persona_mapping[
            cluster
        ] = persona_names[i]

    else:

        persona_mapping[
            cluster
        ] = (
            "Customer Segment "
            + str(i + 1)
        )


df["Persona"] = (
    df["Cluster"]
    .map(persona_mapping)
)


# ============================================================
# 24. PERSONA SUMMARY
# ============================================================

persona_summary = (
    df.groupby(
        [
            "Cluster",
            "Persona"
        ]
    )
    .agg(

        Customers=(
            "Customer_ID",
            "count"
        ),

        Average_Income=(
            "Income",
            "mean"
        ),

        Average_Spending=(
            "Total_Spending",
            "mean"
        ),

        Average_Purchases=(
            "Total_Purchases",
            "mean"
        ),

        Average_Engagement=(
            "Digital_Engagement",
            "mean"
        ),

        Average_Satisfaction=(
            "Satisfaction",
            "mean"
        )
    )
    .reset_index()
)


persona_summary["Percentage"] = (
    persona_summary["Customers"]
    /
    len(df)
    *
    100
)


print(
    "\nCustomer Personas:"
)

print(
    persona_summary.round(2)
)


# ============================================================
# 25. BUSINESS RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 80)
print("ACTIONABLE BUSINESS RECOMMENDATIONS")
print("=" * 80)


for _, row in persona_summary.iterrows():

    persona = row["Persona"]

    print(
        "\n--------------------------------------------"
    )

    print(
        "Persona:",
        persona
    )

    if "VIP" in persona:

        print(
            "Strategy: Provide premium loyalty "
            "benefits, exclusive products, early "
            "access and personalized offers."
        )

    elif "Loyal" in persona:

        print(
            "Strategy: Encourage repeat purchases "
            "through loyalty rewards and cross-selling."
        )

    elif "Growth" in persona:

        print(
            "Strategy: Use personalized recommendations "
            "and product bundles to increase spending."
        )

    elif "Digital" in persona:

        print(
            "Strategy: Focus on digital marketing, "
            "email campaigns and mobile promotions."
        )

    elif "Budget" in persona:

        print(
            "Strategy: Offer discounts, value packs "
            "and price-sensitive promotions."
        )

    elif "Occasional" in persona:

        print(
            "Strategy: Use reminders and targeted "
            "offers to increase purchase frequency."
        )

    elif "Low-Value" in persona:

        print(
            "Strategy: Use low-cost marketing campaigns "
            "and targeted discounts."
        )

    elif "Emerging" in persona:

        print(
            "Strategy: Use welcome offers and "
            "personalized recommendations."
        )

    elif "At-Risk" in persona:

        print(
            "Strategy: Run win-back campaigns and "
            "personalized incentives."
        )

    else:

        print(
            "Strategy: Use reactivation campaigns "
            "and personalized offers."
        )


# ============================================================
# 26. SPENDING VS PURCHASES
# ============================================================

plt.figure(
    figsize=(13, 7)
)

sns.scatterplot(
    data=df,
    x="Total_Spending",
    y="Total_Purchases",
    hue="Persona",
    palette="tab10",
    s=70,
    alpha=0.75
)

plt.title(
    "Customer Personas by Spending and Purchases"
)

plt.xlabel(
    "Total Spending"
)

plt.ylabel(
    "Total Purchases"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.show()


# ============================================================
# 27. PERSONA SIZE GRAPH
# ============================================================

plt.figure(
    figsize=(13, 7)
)

sns.barplot(
    data=persona_summary,
    x="Persona",
    y="Customers",
    hue="Persona",
    palette="Set2",
    legend=False
)

plt.title(
    "Number of Customers in Each Persona"
)

plt.xlabel(
    "Customer Persona"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.show()


# ============================================================
# 28. SAVE SYNTHETIC DATASET
# ============================================================

df.to_csv(
    "synthetic_customer_dataset.csv",
    index=False
)

print(
    "\nSaved: synthetic_customer_dataset.csv"
)


# ============================================================
# 29. SAVE CLUSTER RESULTS
# ============================================================

df.to_csv(
    "customer_segments.csv",
    index=False
)

print(
    "Saved: customer_segments.csv"
)


# ============================================================
# 30. SAVE PERSONA SUMMARY
# ============================================================

persona_summary.to_csv(
    "customer_persona_summary.csv",
    index=False
)

print(
    "Saved: customer_persona_summary.csv"
)


# ============================================================
# 31. SAVE PCA DATA
# ============================================================

pca_df.to_csv(
    "customer_pca_clusters.csv",
    index=False
)

print(
    "Saved: customer_pca_clusters.csv"
)


# ============================================================
# 32. SAVE CLUSTER PROFILE
# ============================================================

cluster_profile.to_csv(
    "cluster_profiles.csv"
)

print(
    "Saved: cluster_profiles.csv"
)


# ============================================================
# 33. SAVE PCA VARIANCE
# ============================================================

variance_df.to_csv(
    "pca_explained_variance.csv",
    index=False
)

print(
    "Saved: pca_explained_variance.csv"
)


# ============================================================
# 34. SAVE MACHINE LEARNING MODEL
# ============================================================

model_package = {

    "scaler": scaler,

    "pca": pca,

    "kmeans": final_kmeans,

    "feature_columns":
        model_df.columns.tolist(),

    "persona_mapping":
        persona_mapping
}


joblib.dump(
    model_package,
    "customer_segmentation_model.pkl"
)

print(
    "Saved: customer_segmentation_model.pkl"
)


# ============================================================
# 35. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PROJECT 3 COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    "\nDataset generated:",
    len(df),
    "customers"
)

print(
    "Features before PCA:",
    model_df.shape[1]
)

print(
    "PCA dimensions:",
    X_pca.shape[1]
)

print(
    "Variance explained by 3 PCA components:",
    round(
        total_variance,
        2
    ),
    "%"
)

print(
    "Optimal number of clusters:",
    optimal_k
)

print(
    "Best Silhouette Score:",
    round(
        best_score,
        4
    )
)


print("\nTechniques completed:")

print("1. Synthetic customer dataset generation")

print("2. Exploratory Data Analysis")

print("3. Missing value treatment")

print("4. Feature engineering")

print("5. Outlier detection using IQR")

print("6. Categorical encoding")

print("7. Feature scaling")

print("8. PCA dimensionality reduction")

print("9. Elbow Method")

print("10. Silhouette Score")

print("11. K-Means clustering")

print("12. 2D PCA visualization")

print("13. 3D PCA visualization")

print("14. Cluster profiling")

print("15. Customer persona creation")

print("16. Business recommendations")


print("\nOutput files created:")

print(
    "- synthetic_customer_dataset.csv"
)

print(
    "- customer_segments.csv"
)

print(
    "- customer_persona_summary.csv"
)

print(
    "- customer_pca_clusters.csv"
)

print(
    "- cluster_profiles.csv"
)

print(
    "- pca_explained_variance.csv"
)

print(
    "- customer_segmentation_model.pkl"
)


print("\n" + "=" * 80)
print("DECODELABS PROJECT 3 - COMPLETE")
print("=" * 80)

input(
    "\nPress ENTER to close..."
)
