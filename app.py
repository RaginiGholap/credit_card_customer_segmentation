import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plot

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

    fig2d, ax2d = plt.subplots()
    ax2d.scatter(pca_data_2d[:, 0], pca_data_2d[:, 1], c=clusters)
    ax2d.set_xlabel("PCA Component 1")
    ax2d.set_ylabel("PCA Component 2")
    ax2d.set_title("2D PCA Cluster Visualization")
    st.pyplot(fig2d)

    # ------------------- 3D PCA -------------------
    st.subheader("🧭 3D Customer Segments (PCA)")
    pca_3d = PCA(n_components=3)
    pca_data_3d = pca_3d.fit_transform(scaled_data)

    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')

    ax3d.scatter(
        pca_data_3d[:, 0],
        pca_data_3d[:, 1],
        pca_data_3d[:, 2],
        c=clusters
    )

    ax3d.set_xlabel("PCA Component 1")
    ax3d.set_ylabel("PCA Component 2")
    ax3d.set_zlabel("PCA Component 3")
    ax3d.set_title("3D PCA Cluster Visualization")

    st.pyplot(fig3d)

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
