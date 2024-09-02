import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.estimators import PC
import networkx as nx

def plot_attribute_distributions(df, figsize=(20, 100), bins=20, row_spacing=0.7):
    """
    Plots distributions of attributes in a DataFrame.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - figsize (tuple, optional): Figure size of the plot. Default is (20, 100).
    - bins (int, optional): Number of bins for histogram plots. Default is 20.
    - row_spacing (float, optional): Spacing between rows of subplots. Default is 0.7.

    Returns:
    - None

    Note:
    - For numeric attributes, a boxplot and histogram are plotted.
    - For non-numeric attributes, a countplot is plotted.
    """

    num_attributes = len(df.columns)
    num_plots = num_attributes * 2
    
    plt.figure(figsize=figsize)
    
    for i, attribute in enumerate(df.columns):
        plt.subplot(num_plots, 2, 2*i+1)
        
        if pd.api.types.is_numeric_dtype(df[attribute]):
            sns.boxplot(data=df, x=attribute, showfliers=True, showmeans=True)
            plt.title(f'{attribute} boxplot')
        else:
            sns.countplot(data=df, x=attribute)
            plt.title(f'{attribute} countplot')

        plt.subplot(num_plots, 2, 2*i+2)
        
        if pd.api.types.is_numeric_dtype(df[attribute]):
            sns.histplot(df[attribute], bins=bins, kde=True, color='skyblue')
            plt.title(f'{attribute} histogram')
        else:
            sns.countplot(data=df, x=attribute)
            plt.title(f'{attribute} countplot')
    
    plt.subplots_adjust(hspace=row_spacing)
    plt.show()

def find_clusters(df, label):
    """
    Finds the optimal number of clusters using the Elbow Method and plots the results.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - label (str): The label of the attribute used for clustering.

    Returns:
    - None

    Note:
    - This function applies the KMeans clustering algorithm to the data in 'df' using the
      specified attribute 'label' and evaluates the silhouette scores and inertia values
      for different numbers of clusters (ranging from 2 to 10). It then plots the Elbow Method
      graph to assist in determining the optimal number of clusters.
    """

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

def plot_elbow_method(k_values, inertia_values, silhouette_scores, label):
    """
    Plots the Elbow Method graph for evaluating the optimal number of clusters.

    Parameters:
    - k_values (list): List of K values (number of clusters).
    - inertia_values (list): List of inertia scores corresponding to each K value.
    - silhouette_scores (list): List of silhouette scores corresponding to each K value.
    - label (str): Label describing the dataset or clustering algorithm.

    Returns:
    - None

    Note:
    - The Elbow Method helps in determining the optimal number of clusters by
      observing the rate of decrease of the inertia score and the silhouette score.
    """

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

def plot_distribution_cluster(df, label, k):
    """
    Performs KMeans clustering on the data and plots the distribution of clusters.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - label (str): The label of the attribute used for clustering.
    - k (int): The number of clusters to be formed.

    Returns:
    - result (array-like): Array containing cluster labels assigned to each data point.
    """

    X = df[[label]]

    kmeans = KMeans(n_clusters=k, random_state=42)
    df_copy = df.copy()
    result = kmeans.fit_predict(X)
    df_copy[f'cluster_{label}'] = result

    plot_cluster_distribution(df_copy, label)
    return result

def plot_cluster_distribution(df, label):
    """
    Plots the distribution of clusters based on the specified label.

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - label (str): The label of the attribute for which the distribution of clusters is plotted.

    Returns:
    - None

    Note:
    - This function plots a histogram showing the distribution of clusters based on the specified label.
      Each cluster is represented by a different color.
    """

    sns.histplot(data=df, x=label, hue=f'cluster_{label}', multiple='stack')
    plt.title('Histogram of Cluster')
    plt.xlabel(label)
    plt.ylabel('Count')
    plt.show()

def print_full(cpd):
    """
    Prints the Conditional Probability Distribution (CPD) without truncation.

    Parameters:
    - cpd (TabularCPD): The Conditional Probability Distribution object.

    Returns:
    - None

    Note:
    - This function temporarily overrides the truncation of the CPD table to print the entire table
      without any truncation or ellipsis.
    """

    # Backup the original truncation method
    backup = TabularCPD._truncate_strtable

    # Override the truncation method to disable truncation
    TabularCPD._truncate_strtable = lambda self, x: x

    # Print the CPD
    print(cpd)

    # Restore the original truncation method
    TabularCPD._truncate_strtable = backup

def pick_biggest_acyclic(estimator, variant, show_progress, max_cond_vars):
    max_cond_vars_count = max_cond_vars
    best_structure = None

    while max_cond_vars_count > 1 :
        try:
            print("\n\tTrying max_cond_vars =", max_cond_vars_count)
            # Try to estimate the structure with the current max_cond_vars
            estimated_edges = estimator.estimate(variant='parallel', show_progress=False, max_cond_vars=max_cond_vars_count)
            
            # If the estimated structure is acyclic, update the best_structure
            if not any(nx.cycle_basis(nx.DiGraph(estimated_edges))):
                return estimated_edges
            else:
                max_cond_vars_count -= 1
        except Exception as e:
            print(f"Error: {e}")
            max_cond_vars_count -= 1

    return best_structure