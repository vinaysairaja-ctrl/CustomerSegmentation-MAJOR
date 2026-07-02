# ==========================================
# Customer Segmentation Dashboard
# ==========================================

# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.cluster import KMeans

# ------------------------------------------
# Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("Mall_Customers.csv")

# ------------------------------------------
# Sidebar
# ------------------------------------------

st.sidebar.title("📊 Dashboard Menu")

page = st.sidebar.radio(
    "",
    [
        "Home",
        "Dataset",
        "EDA",
        "Customer Segmentation",
        "Prediction"
    ]
)

# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.title("📊 Customer Segmentation Dashboard")

    st.write("---")

    st.header("📌 Project Overview")

    st.write("""
This dashboard groups customers based on their Annual Income and Spending Score
using the K-Means Clustering algorithm.

The project helps businesses understand different customer groups and make better
marketing decisions.
""")

    st.write("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Total Customers", len(df))
    col2.metric("👨 Male Customers", len(df[df["Gender"]=="Male"]))
    col3.metric("👩 Female Customers", len(df[df["Gender"]=="Female"]))

    st.write("")

    col1, col2, col3 = st.columns(3)

    col1.metric("🎯 Clusters", 5)
    col2.metric("💰 Highest Income", df["Annual Income (k$)"].max())
    col3.metric("⭐ Highest Spending", df["Spending Score (1-100)"].max())

    st.write("---")

    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# DATASET PAGE
# ==========================================

elif page == "Dataset":

    st.title("📋 Dataset Information")

    st.write("### First 10 Records")

    st.dataframe(df.head(10), use_container_width=True)
    st.caption("Showing the first 10 records of the dataset.")
    st.download_button(
    "📥 Download Dataset",
    data=df.to_csv(index=False),
    file_name="Mall_Customers.csv",
    mime="text/csv"
)

    st.write("---")

    st.write("### 📐 Dataset Shape")

    rows, cols = df.shape
    st.write(f"Rows: {rows}")
    st.write(f"Columns: {cols}")

    st.write("### 📊 Statistical Summary")

    st.dataframe(df.describe(), use_container_width=True)
    st.caption("Summary statistics for all numerical columns.")

    # ==========================================
# EDA PAGE
# ==========================================

elif page == "EDA":

    st.title("📊 Exploratory Data Analysis")

    st.write("Explore the customer dataset using different visualizations.")

    # ------------------------------
    # Gender Distribution
    # ------------------------------

    st.subheader("Gender Distribution")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x="Gender", ax=ax)
    ax.set_title("Gender Distribution")
    st.pyplot(fig)

    # ------------------------------
    # Age Distribution
    # ------------------------------

    st.subheader("Age Distribution")

    fig, ax = plt.subplots(figsize=(6,3))
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    # ------------------------------
    # Annual Income Distribution
    # ------------------------------

    st.subheader("Annual Income Distribution")

    fig, ax = plt.subplots(figsize=(6,3))
    sns.histplot(df["Annual Income (k$)"], bins=20, kde=True, ax=ax)
    ax.set_title("Annual Income Distribution")
    st.pyplot(fig)

    # ------------------------------
    # Spending Score Distribution
    # ------------------------------

    st.subheader("Spending Score Distribution")

    fig, ax = plt.subplots(figsize=(6,3))
    sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True, ax=ax)
    ax.set_title("Spending Score Distribution")
    st.pyplot(fig)

    # ------------------------------
    # Age vs Annual Income
    # ------------------------------

    st.subheader("Age vs Annual Income")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(
        data=df,
        x="Age",
        y="Annual Income (k$)",
        ax=ax
    )
    ax.set_title("Age vs Annual Income")
    st.pyplot(fig)

    # ------------------------------
    # Income vs Spending Score
    # ------------------------------

    st.subheader("Annual Income vs Spending Score")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        ax=ax
    )
    ax.set_title("Annual Income vs Spending Score")
    st.pyplot(fig)

    # ------------------------------
    # Correlation Heatmap
    # ------------------------------

    st.subheader("Correlation Heatmap")

    temp_df = df.copy()

    temp_df["Gender"] = temp_df["Gender"].map({
        "Male":1,
        "Female":0
    })

    fig, ax = plt.subplots(figsize=(6,4))

    sns.heatmap(
        temp_df.corr(numeric_only=True),
        annot=True,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)
    # ==========================================
# CUSTOMER SEGMENTATION PAGE
# ==========================================

elif page == "Customer Segmentation":

    st.title("🎯 Customer Segmentation")

    # Select Features
    x = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    # -----------------------------
    # Elbow Method
    # -----------------------------

    st.subheader("Elbow Method")

    wcss = []

    for i in range(1, 11):
        model = KMeans(n_clusters=i, random_state=42)
        model.fit(x)
        wcss.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(6,3))

    ax.plot(range(1,11), wcss, marker="o")
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")

    st.pyplot(fig)

    st.success("Optimal Number of Clusters = 5")

    # -----------------------------
    # K-Means Model
    # -----------------------------

    model = KMeans(n_clusters=5, random_state=42)

    y_pred = model.fit_predict(x)

    # -----------------------------
    # Cluster Visualization
    # -----------------------------

    st.subheader("Customer Segments")

    fig, ax = plt.subplots(figsize=(6,4))

    colors = ["red","green","blue","orange","purple"]

    for i in range(5):

        ax.scatter(
            x.iloc[y_pred==i,0],
            x.iloc[y_pred==i,1],
            s=70,
            c=colors[i],
            label=f"Cluster {i+1}"
        )

    ax.scatter(
        model.cluster_centers_[:,0],
        model.cluster_centers_[:,1],
        s=250,
        marker="*",
        c="black",
        label="Centroids"
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Segmentation")

    ax.legend()

    st.pyplot(fig)

    # -----------------------------
    # Cluster Centers
    # -----------------------------

    st.subheader("Cluster Centers")

    centers = pd.DataFrame(
        model.cluster_centers_,
        columns=[
            "Annual Income (k$)",
            "Spending Score"
        ]
    )

    st.dataframe(centers)

    st.info("""
Cluster centers represent the average Annual Income and Spending Score of customers in each cluster.
""")
    # ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "Prediction":

    st.title("🤖 Customer Cluster Prediction")

    st.write("Enter the customer's details below.")

    # Train Model
    x = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    model = KMeans(n_clusters=5, random_state=42)
    model.fit(x)

    # User Input
    income = st.number_input(
        "Annual Income (k$)",
        min_value=0,
        max_value=150,
        value=50
    )

    score = st.number_input(
        "Spending Score (1-100)",
        min_value=1,
        max_value=100,
        value=50
    )

    if st.button("Predict Customer Cluster"):

        new_customer = pd.DataFrame(
            [[income, score]],
            columns=[
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        )

        prediction = model.predict(new_customer)[0]

        st.success(f"Predicted Cluster: Cluster {prediction+1}")

        cluster_info = {
            0: "Average Income & Average Spending",
            1: "High Income & High Spending",
            2: "Low Income & High Spending",
            3: "High Income & Low Spending",
            4: "Low Income & Low Spending"
        }

        st.info(f"""
### Customer Type

{cluster_info[prediction]}

### Business Suggestion

Offer personalized marketing based on this customer's spending behaviour.
""")

    st.write("---")

    st.subheader("Cluster Description")

    st.markdown("""
- **Cluster 1:** Average Income & Average Spending
- **Cluster 2:** High Income & High Spending
- **Cluster 3:** Low Income & High Spending
- **Cluster 4:** High Income & Low Spending
- **Cluster 5:** Low Income & Low Spending
""")

    st.write("---")
    st.caption("👨‍💻 Developed by RAJA VINAY SAI")
    st.caption("📊 Customer Segmentation Dashboard using Python, K-Means & Streamlit")