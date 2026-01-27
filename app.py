import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.decomposition import PCA

st.set_page_config(page_title="Credit Card Customer Segmentation", layout="wide")

st.title("💳 Credit Card Customer Segmentation using K-Means")
st.write("This app segments customers based on their credit card usage behavior.")

# Load saved model and scaler
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# File uploader
uploaded_file = st.file_uploader("Upload Credit Card Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Data")
    st.write(df.head())

    # Drop ID column
    if "CUST_ID" in df.columns:
        df = df.drop("CUST_ID", axis=1)

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Scale using saved scaler
    scaled_data = scaler.transform(df)

    # Predict clusters
    clusters = kmeans.predict(scaled_data)
    df["Cluster"] = clusters

    st.subheader("🔢 Clustered Data")
    st.write(df.head())

    # PCA for visualization
    st.subheader("📊 Customer Segments Visualization (PCA)")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Customer Segments")
    st.pyplot(fig)

    # Cluster summary
    st.subheader("📋 Cluster Summary")
    st.write(df.groupby("Cluster").mean())

    # Download option
    st.download_button(
        label="⬇ Download Clustered Data",
        data=df.to_csv(index=False),
        file_name="clustered_customers.csv",
        mime="text/csv"
    )

