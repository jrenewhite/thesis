import logging
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations
import random

class ModelInitializer:
    @staticmethod
    def initialize_classifiers(base_models):
        """
        Initialize classifiers based on the provided list of base model names.

        Args:
            base_models (list): List of strings specifying base model names.

        Returns:
            dict: Dictionary containing initialized classifier objects.
        """
        logging.info("Initializing classifiers for base models: %s", base_models)

        # Available base models configured for multi-class classification and soft voting compatibility
        available_models = {
            'rf': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),  # Supports predict_proba
            'lr': LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),  # Supports predict_proba
            'nb': GaussianNB(),  # Supports predict_proba
            'knn': KNeighborsClassifier(n_neighbors=5),  # Supports predict_proba
            'dt': DecisionTreeClassifier(random_state=42),  # Supports predict_proba
            'et': ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=1),  # Supports predict_proba
            'bnb': BernoulliNB()  # Supports predict_proba
        }

        # Validate base model names
        if len(set(base_models)) != len(base_models):
            logging.error("Base models contain duplicate entries.")
            raise ValueError("Base models contain duplicate entries.")
        if not all(model in available_models for model in base_models):
            logging.error("Invalid base model name(s) specified.")
            raise ValueError("Invalid base model name(s) specified.")

        # Initialize and return classifiers
        classifiers = {model: available_models[model] for model in base_models}
        logging.info("Classifiers initialized: %s", classifiers)
        return classifiers

    @staticmethod
    def create_ensembles(classifiers, num_base_models_in_each_ensemble, num_ensembles):
        """
        Create ensembles using a combination of classifiers.

        Args:
            classifiers (dict): Dictionary containing initialized classifier objects.
            num_base_models_in_each_ensemble (int): Number of base models in each ensemble.
            num_ensembles (int): Number of ensembles to create.

        Returns:
            list: List of ensemble classifier objects.
        """
        logging.info("Creating ensembles with %d base models per ensemble and %d ensembles", 
                     num_base_models_in_each_ensemble, num_ensembles)

        # Validate number of base models in each ensemble
        if num_base_models_in_each_ensemble > len(classifiers):
            logging.error("Number of base models in each ensemble cannot exceed the total number of classifiers.")
            raise ValueError("Number of base models in each ensemble cannot exceed the total number of classifiers.")

        # Generate combinations of base models
        base_model_combinations = list(combinations(classifiers.keys(), num_base_models_in_each_ensemble))

        # Validate number of ensembles
        if num_ensembles > len(base_model_combinations):
            logging.error("Number of ensembles cannot exceed the total number of possible base model combinations.")
            raise ValueError("Number of ensembles cannot exceed the total number of possible base model combinations.")

        # Select random combinations for ensembles
        selected_combinations = random.sample(base_model_combinations, num_ensembles)

        # Create ensembles using selected combinations
        ensembles = []
        for combination in selected_combinations:
            logging.debug("Creating ensemble with combination: %s", combination)
            estimators = [(name, classifiers[name]) for name in combination]
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensembles.append(ensemble)

        logging.info("Ensembles created successfully with %d ensembles", len(ensembles))
        return ensembles
