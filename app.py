import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.express as px

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

    # ------------------- 2D PCA -------------------
    st.subheader("📊 2D Customer Segments (PCA)")
    pca_2d = PCA(n_components=2)
    pca_data_2d = pca_2d.fit_transform(scaled_data)

    fig2d = px.scatter(
        x=pca_data_2d[:, 0],
        y=pca_data_2d[:, 1],
        color=df["Cluster"].astype(str),
        labels={"x": "PCA Component 1", "y": "PCA Component 2"},
        title="2D PCA Cluster Visualization"
    )
    st.plotly_chart(fig2d, use_container_width=True)

    # ------------------- 3D PCA -------------------
    st.subheader("🧭 3D Customer Segments (Interactive PCA)")
    pca_3d = PCA(n_components=3)
    pca_data_3d = pca_3d.fit_transform(scaled_data)

    fig3d = px.scatter_3d(
        x=pca_data_3d[:, 0],
        y=pca_data_3d[:, 1],
        z=pca_data_3d[:, 2],
        color=df["Cluster"].astype(str),
        labels={"x": "PCA 1", "y": "PCA 2", "z": "PCA 3"},
        title="3D PCA Cluster Visualization"
    )
    st.plotly_chart(fig3d, use_container_width=True)

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
