import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st

st.title("KMeans Clustering")

df = pd.read_csv("data/mall_customers.csv")

# Create a feature matrix from our 2 key variables
# Annual Income & Spending Score
X = df.iloc[:, [3, 4]].values

# Elbow Method
# Assess how within cluster variability shifts as number of clusters increases
# Within-Cluster Sum of Squares
# Goal to find the point where an increase in the number of clusters does not result in significant improvement in variability
wcss = []
n_clu = 10

for i in range(1, n_clu + 1):
    kmeans = KMeans(
        n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig = px.line(wcss)

st.header("Elbow Method")
st.plotly_chart(fig)

# Fit predict
# Fit = identify the centroids of the clusters
# Predict = assigns each data point in X to one of the 5 clusters based on proximity to the centroids
kmeans = KMeans(
    n_clusters=5, init="k-means++", max_iter=100, n_init=10, random_state=42
)
y_kmeans = kmeans.fit_predict(X)

# Shove the new cluster column back onto the dataset
st.header("Dataframe with clusters")
df["cluster"] = y_kmeans

st.dataframe(df)

# Define custom colors for each cluster
colors = ["#FF7F50", "#1F77B4", "#2CA02C", "#D62728", "#9467BD"]

# Display the scatter plot with different colors for each cluster using Plotly Express
fig = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="cluster",
    color_discrete_sequence=colors,
)

st.header("KMeans Clustering")
# Show the plot
st.plotly_chart(fig)
