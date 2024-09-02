from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Helps to find the best number of cluster in the dataset, given a certain label
def find_clusters(df, label):
    X = df[[label]]

    silhouette_scores = []
    inertia_values = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)

        silhouette_avg = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

        inertia_values.append(kmeans.inertia_)

    plot_elbow_method(range(2, 11), inertia_values, silhouette_scores, label)

# Plots the silhouette index and inertia in order to use the elbow method
def plot_elbow_method(k_values, inertia_values, silhouette_scores, label):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Number of clusters (K)')
    ax1.set_ylabel('Inertia score', color='tab:blue')
    ax1.plot(k_values, inertia_values, marker='o', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Silhouette score', color='tab:red')
    ax2.plot(k_values, silhouette_scores, marker='s', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0,1)

    plt.title(f'Elbow Method for {label}')
    plt.show()

# Plots the hostogrmas to show the distributions
def plot_distribution_cluster(df, label, k):
    X = df[[label]]

    kmeans = KMeans(n_clusters=k, random_state=42)
    df[f'cluster_{label}'] = kmeans.fit_predict(X)

    plot_cluster_distribution(df, label)

def plot_cluster_distribution(df, label):
    sns.histplot(data=df, x=label, hue=f'cluster_{label}', multiple='stack', bins=20)
    plt.title('Histogram of Cluster')
    plt.xlabel(label)
    plt.ylabel('Count')
    plt.show()

# Example usage:
# find_clusters(your_df, 'your_label')
# plot_distribution_cluster(your_df, 'your_label', k)
