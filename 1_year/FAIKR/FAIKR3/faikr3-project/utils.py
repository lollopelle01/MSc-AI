import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork
import time
from pgmpy.inference import VariableElimination, ApproxInference
from pgmpy.sampling import GibbsSampling
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from sklearn.model_selection import ParameterGrid
from statistics import mean
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from functools import reduce

def plot_dataframe_columns(df, figsize_base=(15, 5), bins=10, show_all_xticks=False, rwidth=0.6):
    df_notna = df.dropna()
    num_columns = len(df_notna.columns)
    
    if num_columns == 0:
        print("No columns to plot.")
        return
    
    # Calculate rows and columns for subplots
    n_cols = int(np.ceil(np.sqrt(num_columns)))
    n_rows = int(np.ceil(num_columns / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figsize_base[0], figsize_base[1] * n_rows))
    axes = np.array(axes).flatten()
    
    # Plot each column
    for i, attr in enumerate(df_notna.columns):
        ax = axes[i]
        if np.issubdtype(df_notna[attr].dtype, np.number):
            unique_values = np.sort(df_notna[attr].unique())
            if not isinstance(bins, int) or (show_all_xticks and (len(unique_values) <= bins)):
                # Ensure bins align with unique values
                bins = np.append(unique_values, unique_values[-1] + 1) - 0.5
            
            df_notna[attr].plot(kind='hist', ax=ax, title=attr, bins=bins, edgecolor='black', rwidth=rwidth, align='mid')
            ax.set_ylabel('Frequency')
            
            if show_all_xticks:
                ax.set_xticks(unique_values)
                ax.set_xticklabels(unique_values.astype(int), rotation=0, ha='center')
                ax.set_xlim([unique_values.min() - 1, unique_values.max() + 1])
        else:
            df_notna[attr].value_counts().plot(kind='bar', ax=ax, title=attr, edgecolor='black')
            ax.set_ylabel('Count')
            ax.set_xlabel('')
            if show_all_xticks:
                ax.set_xticks(range(len(df_notna[attr].value_counts())))
    
    # Hide unused subplots
    for j in range(num_columns, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# We reconstruct the used encodings for acronyms used
def decode_degree(degree):
    degree_mapping = {
        "B.Pharm": "Bachelor of Pharmacy",
        "BSc": "Bachelor of Science",
        "BA": "Bachelor of Arts",
        "BCA": "Bachelor of Computer Applications",
        "M.Tech": "Master of Technology",
        "PhD": "Doctor of Philosophy",
        "Class 12": "High School Completion (12th Grade)",
        "B.Ed": "Bachelor of Education",
        "LLB": "Bachelor of Laws",
        "BE": "Bachelor of Engineering",
        "M.Ed": "Master of Education",
        "MSc": "Master of Science",
        "BHM": "Bachelor of Hotel Management",
        "M.Pharm": "Master of Pharmacy",
        "MCA": "Master of Computer Applications",
        "MA": "Master of Arts",
        "B.Com": "Bachelor of Commerce",
        "MD": "Doctor of Medicine",
        "MBA": "Master of Business Administration",
        "MBBS": "Bachelor of Medicine, Bachelor of Surgery",
        "M.Com": "Master of Commerce",
        "B.Arch": "Bachelor of Architecture",
        "LLM": "Master of Laws",
        "B.Tech": "Bachelor of Technology",
        "BBA": "Bachelor of Business Administration",
        "ME": "Master of Engineering",
        "MHM": "Master of Hotel Management",
        "Others": "Other Qualifications"
    }
    
    return degree_mapping.get(degree, "Unknown Degree")

# To follow the notation of Bayesian Networks
def to_camel_case(s: str) -> str:
    return ''.join(word.capitalize() for word in s.split())

def hierarchical_layout(G, vertical_spacing=1.5, horizontal_spacing=2.0):
    """
    Generates a hierarchical layout for a graph, distributing nodes into levels.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Hierarchical layout works best with a Directed Acyclic Graph (DAG).")

    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not roots:
        shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
        avg_distance = {node: sum(d.values()) / len(d) for node, d in shortest_paths.items()}
        roots = [min(avg_distance, key=avg_distance.get)]

    levels = {}
    for root in roots:
        for node, level in nx.single_source_shortest_path_length(G, root).items():
            levels[node] = min(levels.get(node, float("inf")), level)

    level_dict = {}
    for node, level in levels.items():
        level_dict.setdefault(level, []).append(node)

    pos = {}
    for level, nodes in level_dict.items():
        x_positions = np.linspace(-len(nodes) / 2, len(nodes) / 2, len(nodes))
        for x, node in zip(x_positions, nodes):
            pos[node] = (x * horizontal_spacing, -level * vertical_spacing)

    return pos

def draw_graph(G, pos, node_color="lightblue", edge_color="gray", node_size=1000, title=None):
    """
    Draws the graph with the given layout, avoiding curved edges among same-layer nodes,
    ensuring no duplicate edges are drawn, and placing arrows at the end of edges.
    Allows setting a title that is clearly visible above the graph.
    """
    plt.figure(figsize=(15, 8))
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    nx.draw(G, pos, edgelist=[], with_labels=False, node_color=node_color, edge_color=edge_color, node_size=node_size)
    
    # Track edges already drawn to avoid duplicates
    drawn_edges = set()
    
    for edge in G.edges():
        start, end = edge
        if (start, end) in drawn_edges or (end, start) in drawn_edges:
            continue
        drawn_edges.add((start, end))
        
        x1, y1 = pos[start]
        x2, y2 = pos[end]
        
        # Check if nodes are at the same level (same y position)
        curvature = 0.3 if y1 == y2 else 0.0
        plt.annotate("",
                     xy=(x2, y2), xycoords='data',
                     xytext=(x1, y1), textcoords='data',
                     arrowprops=dict(arrowstyle="->",
                                     color=edge_color,
                                     lw=1.5,
                                     connectionstyle=f"arc3,rad={curvature}"))
    
    # Improved positioning of node labels
    for node, (x, y) in pos.items():
        offset_x = 0.1 if x > 0 else -0.1
        offset_y = 0.15
        plt.text(x + offset_x, y + offset_y, str(node), fontsize=10, ha='center',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    
    plt.show()

def print_full(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd)
    TabularCPD._truncate_strtable = backup
    
def count_bn_parameters(model: BayesianNetwork) -> int:
    """
    Compute the total number of free parameters in a discrete Bayesian Network.
    
    For each node X with r states and parents with product of cardinalities Q,
    the number of free parameters is (r - 1) * Q.
    
    Parameters:
        model: BayesianNetwork
            The Bayesian network model with CPDs defined.
    
    Returns:
        total_parameters: int
            The total number of free parameters in the network.
    """
    total_parameters = 0
    for cpd in model.get_cpds():
        r = cpd.cardinality[0]  # Cardinality of the variable
        Q = np.prod(cpd.cardinality[1:]) if len(cpd.cardinality) > 1 else 1  # Product of parent cardinalities
        free_params = (r - 1) * Q
        total_parameters += free_params
        
    return total_parameters

def log_like_score(model, df, show_progress=True):
    inference = VariableElimination(model)
    log_likelihood = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing Log-Likelihood", ncols=80, disable=not show_progress):
        for col in df.columns:
            evidence = {c: row[c] for c in df.columns if c != col}
            result = inference.query(variables=[col], evidence=evidence)
            
            # This is because "Cgpa" has no values with label 1 so we have to shift indexes
            # We hardcoded this as it's an anomaly of the dataset not to have the 1 label and 
            # we consider still right the binning we used 
            observed_val = int(row[col])
            if (col == 'Cgpa') and observed_val > 0:
                log_likelihood += np.log(result.values[int(observed_val - 1)])
            else:
                log_likelihood += np.log(result.values[int(observed_val)])
                
    return log_likelihood

def compute_bic(model, data, log_like_precomputed=None):
    """
    Computes the Bayesian Information Criterion (BIC) score for a given Bayesian Network model and dataset.
    
    Parameters:
        model (BayesianNetwork): The Bayesian Network model.
        data (pd.DataFrame): A DataFrame where each row represents an observation with observed variable values.
    
    Returns:
        float: The BIC score of the model given the dataset.
    """
    if log_like_precomputed is not None:
        log_likelihood = log_like_precomputed
    else:
        log_likelihood = log_like_score(model, data, show_progress=False)
    num_params = count_bn_parameters(model) # Total number of parameters
    num_samples = len(data)  # Number of data points
    
    bic = log_likelihood - (num_params * np.log(num_samples)) / 2
    return bic

def compute_accuracy(model, data, target_col, train_size=0.8, seed=42, fitting_params=None, show_progress=True):
    """
    Trains a Bayesian Network on a dataset and computes the accuracy for predicting a given attribute.

    :param model: BayesianNetwork (without CPDs) to be trained.
    :param data: Pandas DataFrame containing the dataset.
    :param target_col: Name of the column to be predicted.
    :param train_size: Percentage of the dataset used for training (default: 0.8).
    :param seed: Seed for train-test splitting (default: 42).
    :param fitting_params: Dictionary containing the estimator and its parameters.
    :param show_progress: Whether to display progress bars (default: True).
    :return: Prediction accuracy (float).
    """
    if fitting_params is None:
        raise ValueError("You must provide a fitting_params dictionary with an estimator and parameters.")

    # Copy the model structure (without CPDs)
    model_copy = BayesianNetwork(model.edges(), latents=list(model.latents))

    # Split dataset into train and test sets
    if show_progress:
        print("Splitting dataset...")
    train_data, test_data = train_test_split(data, train_size=train_size, random_state=seed)

    # Train the model with the given estimator and parameters
    if show_progress:
        print("Training the model...")
    estimator = fitting_params["estimator"](model_copy, train_data)
    estimated_params = estimator.get_parameters(**fitting_params["params"])

    # Assign learned CPDs to the model
    model_copy.cpds = estimated_params

    # Perform inference on the trained model
    inference = VariableElimination(model_copy)
    correct_predictions = 0
    total_predictions = len(test_data)

    # Compute accuracy with progress bar
    actual_total_predictions = total_predictions
    for _, row in tqdm(test_data.iterrows(), total=total_predictions, desc="Computing Accuracy", disable=not show_progress):
        evidence = row.drop(labels=[target_col]).to_dict()  # Use all other attributes as evidence
        real_value = row[target_col]

        try:
            predicted_value = inference.map_query([target_col], evidence=evidence, show_progress=False)[target_col]
            if predicted_value == real_value:
                correct_predictions += 1
        except:
            actual_total_predictions -= 1
            pass  # Ignore row if inference fails
        
    if show_progress and (actual_total_predictions == total_predictions) : 
        print("All inference succeeded!")

    return correct_predictions / actual_total_predictions if actual_total_predictions > 0 else 0

def query(model, query_vars, evidence=None, num_samples=1000):
    """
    Executes a query on a Bayesian network using exact inference and multiple approximate inference methods,
    considering the possibility of latent nodes.
    Tracks execution time for each method.
    
    :param model: BayesianModel from pgmpy
    :param query_vars: List of variables to query
    :param evidence: Dictionary of evidence variables and their values
    :param num_samples: Number of samples for approximate inference
    :return: Dictionary containing results and execution times
    """
    results = {}
    
    # Exact Inference
    exact_infer = VariableElimination(model)
    start_time = time.time()
    exact_result = exact_infer.query(
        variables=query_vars, 
        evidence=evidence,
        show_progress=False
    )
    exact_time = time.time() - start_time
    results['exact'] = {'result': exact_result, 'time': exact_time}
    
    ## Approximate Inference Methods
    #approx_infer = ApproxInference(model)
    #start_time = time.time()
    #approx_result = approx_infer.query(
    #    variables=query_vars,
    #    evidence=evidence,
    #    n_samples=num_samples,
    #    show_progress=False
    #    )
    #approx_time = time.time() - start_time
    #results['approx'] = {'result': approx_result, 'time': approx_time}

    return results

def get_nested_value(A, indices):
    '''
    Retrieve the result of a multidimensional array where each index is about a dimension of the array.
    '''
    try:
        for idx in indices:
            A = A[idx]
        return A
    except IndexError:
        print(f"Error: Index {indices} is out of bounds for array with shape {A.shape}")
        return None

def general_query(
    model,
    query_vars, 
    attribute_evidence, 
    index_retrieve_values, 
    show_progress=False
):
    """
    Generalized querying function to handle multiple query variables and index retrieval.
    """

    # Generate all possible combinations of evidence
    evidences = list(ParameterGrid({attr: model.get_cpds(attr).state_names[attr] for attr in attribute_evidence}))

    # Create a DataFrame to store results
    results_df = pd.DataFrame()

    # Collect results for each evidence
    evidence_iter = tqdm(evidences, desc=f"\tProcessing evidences", disable=not show_progress)

    for evidence in evidence_iter:
        result = query(model, query_vars, evidence=evidence)

        # Prepare a dictionary to store the results for each query variable and index
        result_dict = {'Evidence': str(evidence)}
            
        exact_score = get_nested_value(result["exact"]["result"].values, index_retrieve_values)
        exact_time = result["exact"]["time"]
#
        # Store results with a clear key format
        result_str = ", ".join(f"{var}={idx}" for var, idx in zip(query_vars, index_retrieve_values))
        result_dict[f'Exact_Score_{result_str}'] = exact_score
        result_dict[f'Exact_Time_{result_str}'] = exact_time

        result_df = pd.DataFrame([result_dict])
        results_df = pd.concat([results_df, result_df], ignore_index=True)

    return results_df