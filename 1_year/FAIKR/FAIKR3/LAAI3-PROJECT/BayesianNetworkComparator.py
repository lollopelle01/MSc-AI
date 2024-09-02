import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import find_clusters, plot_distribution_cluster, print_full, pick_biggest_acyclic

from pgmpy.models import BayesianNetwork, NaiveBayes
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, TreeSearch, BDeuScore, K2Score, TreeSearch, BicScore, ParameterEstimator, BayesianEstimator, ExpectationMaximization, BDsScore, AICScore, IVEstimator, PC, MmhcEstimator, ExhaustiveSearch

from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.sampling import BayesianModelSampling, GibbsSampling
from pgmpy.factors.discrete import State
import networkx as nx
import itertools

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pgmpy.inference.ExactInference import BeliefPropagation
from pgmpy.inference.CausalInference import CausalInference

import time

class BayesianNetworkComparator:
    def __init__(self, dataset):
        self.df = dataset
        self.structures = {}
        self.models = {}

    def generate_structures(self, target_label, custom=None):
        """
        Generate different structure configurations of Bayesian network models, given the dataset.
        
        Parameters:
        - df (pd.DataFrame): The dataset containing the data for analysis.

        The following structure configurations are generated:
        - Custom Model: A custom-defined Bayesian network model provided by the user.
        - Tree Model (Chow-Liu): A tree-structured Bayesian network model learned using the Chow-Liu algorithm.
        - Tree Model (TAN): A tree-structured Bayesian network model learned using the Tree Augmented Naive Bayes (TAN) algorithm.
        - HillClimb Models: Bayesian network models learned using the HillClimb search algorithm with various scoring methods such as BDeu, BDS, BIC, and AIC.
        - Naive Bayes Model: A Naive Bayes classifier for discrete features.
        """

        df = self.df

        ### Custom model ################################################

        if custom :
            print("generating Custom stucture in ", end='')
            start_time = time.time()
            self.structures['Custom'] = BayesianNetwork(custom)
            print((time.time() - start_time)*1000, "ms")

        ### Tree model ##################################################
        ts = TreeSearch(data=df)

        print("generating Tree (chow-liu) in ", end='')
        start_time = time.time()
        tree_model_chow_liu = ts.estimate(show_progress=False, estimator_type='chow-liu')
        self.structures['Tree (chow-liu)'] = BayesianNetwork(tree_model_chow_liu.edges())
        print((time.time() - start_time)*1000, "ms")

        print("generating Tree (tan) in ", end='')
        start_time = time.time()
        tree_model_tan = ts.estimate(show_progress=False, estimator_type='tan', class_node=target_label)
        self.structures['Tree (tan)'] = BayesianNetwork(tree_model_tan.edges())
        print((time.time() - start_time)*1000, "ms")

        ### Hillclimb model #############################################
        hc = HillClimbSearch(data=df)
        scores = ['k2score', 'bdeuscore', 'bdsscore', 'bicscore', 'aicscore'] # k2score is too complex

        for score in scores :
            print(f"generating HillClimb ({score}) in ", end='')
            start_time = time.time()
            self.structures['HillClimb (' + score + ')'] = BayesianNetwork(hc.estimate(scoring_method=score, show_progress=False).edges())
            print((time.time() - start_time)*1000, "ms")

        ## Naive Bayes model
        print(f"generating NaiveBayes in ", end='')
        start_time = time.time()
        self.structures['NaiveBayes'] = BayesianNetwork(NaiveBayes(feature_vars=df.columns.drop(target_label),dependent_var=target_label))
        print((time.time() - start_time)*1000, "ms")

        ### PC (not ready)      #TODO : capire se tenerlo o meno e nel caso come motivarlo 
        # pc = PC(data=df)
        # print("generating PC ", end='')
        # start_time = time.time()
        # correct_structure = pick_biggest_acyclic(estimator=pc, variant='parallel', show_progress=False, max_cond_vars=df.shape[1])
        # if correct_structure :
        #     self.structures['PC'] = BayesianNetwork(correct_structure.edges())
        #     print("in ", (time.time() - start_time)*1000, "ms")
        # else : 
        #     print("... FAILED => discard structure")

        scores = {
            'k2score' : K2Score(data=df),
            'bdeuscore' : BDeuScore(data=df),
            'bicscore' : BicScore(data=df), 
            'aicscore' : AICScore(data=df)
        }

        for score_name, score in scores.items() :
            if df.shape[1] <= 6 : # pgmpy:Generating all DAGs of n nodes likely not feasible for n>6!
                ### Exhaustive Search
                ex = ExhaustiveSearch(data=df)
                print(f"generating Exaustive ({score_name}) in ", end='')
                start_time = time.time()
                self.structures['Exaustive (' + score_name + ')'] = BayesianNetwork(ExhaustiveSearch(data=df, scoring_method=score).estimate().edges())
                print((time.time() - start_time)*1000, "ms")
            else :
                print("Too many attributes for Exaustive Search (n>6)")
            
            ### Mmhc Estimator
            mmhc = MmhcEstimator(data=df)
            print(f"generating MMHC ({score_name}) in ", end='')
            start_time = time.time()
            self.structures['MMHC (' + score_name + ')'] = BayesianNetwork(mmhc.estimate(scoring_method=score).edges())
            print((time.time() - start_time)*1000, "ms")

    def display_structures(self):
        """
        Display the graphical representation of generated Bayesian network structures.
        """

        num_structures = len(self.structures)
        rows = (num_structures + 2) // 3  # Calculate number of rows based on the number of structures
        cols = min(num_structures, 3)  # Maximum of 3 columns

        layout = nx.drawing.layout.circular_layout(self.structures[list(self.structures.keys())[0]])

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 10, rows * 10))  # Adjust figsize based on rows and columns

        for i, (m, ax) in enumerate(zip(self.structures.keys(), axes.flatten())):
            # Ensure all nodes in the model are present in the layout
            nx.draw_networkx(self.structures[m], pos=layout, ax=ax, node_size=5000, with_labels=False)

            # Draw node labels with line breaks
            labels = {k: '\n'.join(k.split()) for k in self.structures[m].nodes()}  # Replace spaces with line breaks
            nx.draw_networkx_labels(self.structures[m], pos=layout, labels=labels, ax=ax, font_size=10)

            ax.set_title(f'{m} model')

        plt.tight_layout()
        plt.show()

    def evaluate_parameters(self):
        """
        Evaluate parameters for the Bayesian network models using different parameter estimation techniques.

        This method evaluates parameters for the Bayesian network models stored in the 'structures' attribute
        using different parameter estimation techniques, such as Maximum Likelihood Estimation (MLE) and Expectation-Maximization (EM).

        Parameters:
        - df (pd.DataFrame): The dataset containing the data for parameter estimation.

        The method iterates over each Bayesian network model in the 'structures' attribute and applies the specified
        parameter estimation techniques to obtain the parameters for the models. It then creates new models with
        the evaluated parameters and stores them in the 'models' attribute of the class instance.

        The 'models' attribute is a dictionary where keys represent the model names, and values represent dictionaries
        containing the evaluated models with corresponding parameter estimation techniques as keys.

        For example, if the evaluated model for a Bayesian network named 'Custom' using Maximum Likelihood Estimation
        is stored, it can be accessed as self.models['Custom']['[MaximumLikelihoodEstimator]'].
        """

        df = self.df
        
        estimators = [MaximumLikelihoodEstimator, ExpectationMaximization]
        models = {}

        for model_name, model in self.structures.items():
            print(f"Training {model_name}: " + 100*'-')
            for estimator_class in estimators:
                print(f"\tfor {estimator_class.__name__} ... ", end='')
                estimator = estimator_class(model, df)
                parameters = estimator.get_parameters()

                # Create a new model and add CPDs
                evaluated_model = model.copy()
                evaluated_model.add_cpds(*parameters)

                models[f"{model_name}"] = { "[{estimator_class.__name__}]" : evaluated_model }
                print("finished")

        return models

    def structure_test(self, df):
        """
        Test the structure of Bayesian network models using various evaluation metrics.

        This method evaluates the structure of Bayesian network models stored in the 'structures' attribute
        using different evaluation metrics such as K2 score, BDeu score, BDS score, BIC score, and log-likelihood score.

        Parameters:
        - df (pd.DataFrame): The dataset containing the data for structure evaluation.

        Returns:
        - results (pd.DataFrame): A DataFrame containing the evaluation results for each Bayesian network model.
        The DataFrame has the model names as rows and the evaluation metrics (K2, BDeu, BDS, BIC, log-likelihood) as columns.

        The method iterates over each Bayesian network model in the 'structures' attribute and computes the structure
        scores using the specified evaluation metrics. It then stores the evaluation results in a DataFrame where rows
        represent model names and columns represent evaluation metrics.
        """

        results = pd.DataFrame(index=list(self.structures.keys()), columns=['k2', 'bdeu', 'bds', 'bic', 'log_likelihood'])

        from pgmpy.metrics import log_likelihood_score, structure_score

        structure_mthds = ['k2', 'bdeu', 'bds', 'bic']
        # correlation_mthds = ['chi_square', 'g_sq', 'log_likelihood', 'freeman_tuckey', 'modified_log_likelihood', 'neyman', 'cressie_read']

        for model_name, model in self.structures.items() :
            for strct_mthd in structure_mthds:
                results.loc[model_name, strct_mthd] = structure_score(model, data=df, scoring_method=strct_mthd)
            results.loc[model_name, 'log_likelihood'] = log_likelihood_score(model, data=df)
        
        return results

    def split_dataset_for_bayesian_network(df, target_label, train_size):
        """
        Split the dataset into a combined X_train and y_train, and X_test, y_test.

        Parameters:
        - df: DataFrame, the dataset.
        - target_label: str, the label of the target variable.
        - train_size: float, the proportion of the dataset to include in the train split.

        Returns:
        - X_train_combined: DataFrame, the combined training features and target variable.
        - X_test: DataFrame, the testing features.
        - y_test: Series, the testing target variable.
        """
        # Extract features (X) and target variable (y)
        X = df.drop(columns=[target_label])
        y = df[target_label]

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

        # Combine X_train and y_train
        X_train_combined = X_train.copy()
        X_train_combined[target_label] = y_train

        return X_train_combined, X_test, y_test
    
    def test(X_test, y_test, model, target_label, exact=True):
        """
        Evaluate a Bayesian Network model on the test set and calculate various performance metrics.

        Parameters:
        - X_test (pd.DataFrame): The testing features DataFrame.
        - y_test (pd.Series): The true testing target variable Series.
        - model (pgmpy.models.BayesianNetwork): The trained Bayesian Network model.
        - target_label (str): The label of the target variable.
        - exact (bool, optional): Whether to perform exact inference. Defaults to True.

        Returns:
        - result (pd.DataFrame): A DataFrame containing performance metrics for each type of inference.
        The DataFrame has the following columns:
        - 'Accuracy': Accuracy of the model for each type of inference.
        - 'Precision': Precision of the model for each type of inference.
        - 'Recall': Recall of the model for each type of inference.
        - 'F1': F1-score of the model for each type of inference.
        The index of the DataFrame represents different types of inference, such as variable elimination,
        belief propagation, and causal inference.
        """

        # Get predictions for each instance in X_test
        predictions = {
            'variable_elimination' : [],
            'belief_propagation' : [],
            'causal_inference' : [],
            # 'gibbs_sampling' : [],
        }

        # Inferences
        variable_elimination = VariableElimination(model)
        belief_propagation = BeliefPropagation(model)
        causal_inference = CausalInference(model)
        
        if exact :
            for i,(_ ,instance) in enumerate(X_test.iterrows()):
                predictions['variable_elimination'].append(variable_elimination.map_query(variables=[target_label], evidence=dict(instance), show_progress=False)[target_label])
                predictions['belief_propagation'].append(belief_propagation.map_query(variables=[target_label], evidence=dict(instance), show_progress=False)[target_label])
                predictions['causal_inference'].append(np.argmax(causal_inference.query(variables=[target_label], evidence=dict(instance), show_progress=False).values))

                print(f"Done: {i}/{len(X_test)}")
        else :
            # TODO : implement approximate inference
            pass

        # Initialize empty lists to store metric values
        accuracy = []
        precision = []
        recall = []
        f1 = []

        # Calculate performance metrics
        for infer_type in predictions.keys():
            accuracy.append(accuracy_score(y_test, predictions[infer_type]))
            precision.append(precision_score(y_test, predictions[infer_type], average='weighted'))
            recall.append(recall_score(y_test, predictions[infer_type], average='weighted'))
            f1.append(f1_score(y_test, predictions[infer_type], average='weighted'))

        # Create a dictionary containing the metric values
        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }

        # Create a DataFrame from the dictionary
        result = pd.DataFrame(metrics_dict)

        # Set the index of the DataFrame to be the inference types
        result.index = predictions.keys() 

        return result
    
    # Funzione per confrontare tutte le strutture e tutti i parametri 
    def compare_models(self, df, target_label):
        train_sizes = range(0.1, 1, 0.1) # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results_per_training = {} 

        for train_size in train_sizes :
            # Extract the sets
            X_train, X_test, y_test = self.split_dataset_for_bayesian_network(df, target_label, train_size=train_size)

            # Copy the structure of the models obtained before so that we don't change it
            # models_structure = self.models.copy()

            # Train the models on train test
            models_evaluated = self.evaluate_parameters(X_train)
            
            # Make predictions on test set
            measures = ['accuracy', 'precision', 'recall', 'f1']
            results = pd.DataFrame(index=list(models_evaluated.keys()), columns=measures)
            for model_name, model in models_evaluated.items():
                # evaluate predictions
                results.loc[model_name] = self.test(X_test, y_test, model)
            results_per_training[train_size] = results