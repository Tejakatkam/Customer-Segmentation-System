import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# App Title
st.title("DATA-DRIVEN CUSTOMER SEGMENTATION")

# Sidebar for Dashboard
st.sidebar.title("Dashboard")
pages = ["Upload Data", "Data Overview", "Visualizations", "Interactive Analysis", "K-Means Clustering", "PCA Visualization & Final Analysis"]
page = st.sidebar.radio("Go to", pages)

# Upload Data
if page == "Upload Data":
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            # Attempt to read CSV or Excel file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            
            st.success("File uploaded successfully!")
            st.write(df.head())
            
            # Save the dataset for other pages
            st.session_state["data"] = df

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Data Overview
if page == "Data Overview":
    st.header("Data Overview")
    if "data" in st.session_state:
        df = st.session_state["data"]

        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("Dataset Info")
        # Directly print the paragraph as a markdown
        paragraph = """
        The mall customer dataset contains information about customers’ demographics, including age, gender, and annual income, 
        as well as their spending score (ranging from 1 to 100). It provides valuable insights into customer behavior, enabling 
        segmentation based on purchasing patterns and demographic traits. This dataset is useful for understanding customer 
        preferences, identifying market segments, and tailoring marketing strategies.
        """
        st.markdown(paragraph)

        st.subheader("Summary Statistics")
        st.write(df.describe())
    else:
        st.warning("Please upload a dataset first.")
        

# Visualizations
if page == "Visualizations":
    st.header("Visualizations")
    if "data" in st.session_state:
        df = st.session_state["data"]

        st.subheader("Correlation Heatmap")
        if st.button("Generate Heatmap"):
            if "data" in st.session_state:
                df = st.session_state["data"]

                try:
                    # Select numeric columns only
                    numeric_columns = df.select_dtypes(include=np.number)
                    
                    if numeric_columns.empty:
                        st.warning("No numeric columns available for correlation heatmap.")
                    else:
                        # Drop missing values for correlation computation
                        df_cleaned = numeric_columns.dropna()

                        # Plot heatmap
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(df_cleaned.corr(), annot=True, cmap="coolwarm", ax=ax)
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error generating heatmap: {e}")
            else:
                st.warning("Please upload a dataset first.")


        st.subheader("Custom Chart")
        chart_type = st.selectbox("Select chart type", ["Bar Plot", "Line Plot", "Scatter Plot"])
        x_axis = st.selectbox("Select X-axis", df.columns)
        y_axis = st.selectbox("Select Y-axis", df.columns)

        if st.button("Generate Chart"):
            fig, ax = plt.subplots()
            if chart_type == "Bar Plot":
                sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax)
            elif chart_type == "Line Plot":
                sns.lineplot(x=x_axis, y=y_axis, data=df, ax=ax)
            elif chart_type == "Scatter Plot":
                sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Please upload a dataset first.")

# Interactive Analysis
if page == "Interactive Analysis":
    st.header("Interactive Analysis")
    if "data" in st.session_state:
        df = st.session_state["data"]

        st.subheader("Filter Data")
        columns = st.multiselect("Select columns to display", df.columns, default=df.columns)
        num_rows = st.slider("Select number of rows to display", min_value=5, max_value=len(df), value=10)
        st.write(df[columns].head(num_rows))
    else:
        st.warning("Please upload a dataset first.")

# Function to generate descriptive cluster interpretations
def interpret_clusters_descriptive(df, numeric_columns):
    grouped = df.groupby('Cluster')[numeric_columns].mean()
    cluster_descriptions = []

    for cluster in grouped.index:
        income = grouped.loc[cluster, 'Annual Income (k$)']
        spending = grouped.loc[cluster, 'Spending Score (1-100)']
        age = grouped.loc[cluster, 'Age']

        description = f"**Cluster {cluster}:** "

        # Income interpretation
        if income > 75:
            description += "This cluster represents customers with high income, "
        elif income > 50:
            description += "This cluster represents customers with moderate income, "
        else:
            description += "This cluster represents customers with low income, "

        # Spending score interpretation
        if spending > 70:
            description += "high spending capacity, "
        elif spending > 40:
            description += "average spending capacity, "
        else:
            description += "low spending capacity, "

        # Age interpretation
        if age > 45:
            description += "and they are generally older (above 45 years)."
        elif age > 30:
            description += "and they are middle-aged (30-45 years)."
        else:
            description += "and they are young (under 30 years)."

        cluster_descriptions.append(description)

    return "\n\n".join(cluster_descriptions)

# K-Means Clustering
if page == "K-Means Clustering":
    st.header("K-Means Clustering")
    if "data" in st.session_state:
        df = st.session_state["data"]

        # Select numeric columns dynamically
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_columns:
            st.error("No numeric columns available for clustering.")
        else:
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_columns])

            # Clustering
            num_clusters = st.slider("Select Number of Clusters", 2, 10, 4)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled_data)

            # Save cluster data to session state
            st.session_state["data"] = df

            # Show Silhouette Score
            silhouette = silhouette_score(scaled_data, df['Cluster'])
            st.write(f"Silhouette Score: {silhouette:.2f}")

            # Show cluster summary
            st.subheader("Cluster Averages")
            cluster_averages = df.groupby('Cluster')[numeric_columns].mean()
            st.dataframe(cluster_averages)
    else:
        st.warning("Please upload a dataset first.")

# PCA Visualization 
# PCA Visualization for Specific Features
if page == "PCA Visualization & Final Analysis":
    st.header("PCA Visualization")
    if "data" in st.session_state and "Cluster" in st.session_state["data"].columns:
        df = st.session_state["data"]

        # Select specific numeric columns for PCA
        specific_columns = ["Age", "Spending Score (1-100)", "Annual Income (k$)"]
        if all(col in df.columns for col in specific_columns):
            selected_data = df[specific_columns]

            # Standardize selected data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(selected_data)

            # PCA for dimensionality reduction
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)

            # Plot PCA results
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=df['Cluster'], cmap='viridis', s=50, alpha=0.7)
            ax.set_title("PCA Visualization Based on Age, Spending Score, and Annual Income")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig)

            # Dynamic descriptive analysis using the PCA graph and cluster averages
            st.subheader("Final Cluster Analysis")
            numeric_columns = ["Age", "Spending Score (1-100)", "Annual Income (k$)"]
            descriptive_analysis = interpret_clusters_descriptive(df, numeric_columns)
            st.markdown(descriptive_analysis)
        else:
            st.warning("The required columns for PCA are missing in the dataset. Ensure 'Age', 'Spending Score (1-100)', and 'Annual Income (k$)' are present.")
    else:
        st.warning("Please complete K-Means Clustering first.")

    
    
