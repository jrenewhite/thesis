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
    def initialize_classifiers(base_models, model_params=None):
        """
        Initialize classifiers based on provided model names and optional hyperparameters.
        Defaults to standard parameters if none are provided in model_params.

        Args:
            base_models (list): List of strings specifying base model names.
            model_params (dict): Optional dictionary of model-specific parameters.

        Returns:
            dict: Dictionary containing initialized classifier objects.
        """
        logging.info("Initializing classifiers for base models: %s", base_models)

        # Define models with default parameters for multi-class and soft voting compatibility
        default_models = {
            'rf': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'lr': LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
            'nb': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'dt': DecisionTreeClassifier(random_state=42),
            'et': ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=1),
            'bnb': BernoulliNB()
        }

        classifiers = {}
        for model_name in base_models:
            if model_name in default_models:
                # Retrieve parameters from model_params if available, or use default
                params = model_params.get(model_name, {}) if model_params else {}
                try:
                    # Merge default parameters with any provided in model_params
                    classifiers[model_name] = type(default_models[model_name])(
                        **{**default_models[model_name].get_params(), **params}
                    )
                    logging.info("Initialized model '%s' with parameters: %s", model_name, params)
                except Exception as e:
                    logging.error("Error initializing model '%s' with parameters %s: %s", model_name, params, e)
                    raise
            else:
                logging.error("Model '%s' is not recognized in available models.", model_name)
                raise ValueError(f"Model '{model_name}' is not recognized.")

        logging.info("Classifiers initialized successfully: %s", list(classifiers.keys()))
        return classifiers

    @staticmethod
    def create_ensembles(classifiers, num_base_models_in_each_ensemble, num_ensembles):
        """
        Create ensembles from combinations of classifiers for ensemble learning.

        Args:
            classifiers (dict): Dictionary containing initialized classifier objects.
            num_base_models_in_each_ensemble (int): Number of base models in each ensemble.
            num_ensembles (int): Number of ensembles to create.

        Returns:
            list: List of ensemble classifier objects.
        """
        logging.info("Creating ensembles with %d base models per ensemble and %d total ensembles",
                     num_base_models_in_each_ensemble, num_ensembles)

        # Check that each ensemble has fewer models than total classifiers available
        if num_base_models_in_each_ensemble > len(classifiers):
            logging.error("Number of base models per ensemble exceeds the total classifiers available.")
            raise ValueError("Cannot create an ensemble with more base models than available classifiers.")

        # Generate all possible combinations of models for ensembles
        base_model_combinations = list(combinations(classifiers.keys(), num_base_models_in_each_ensemble))

        # Validate number of ensembles
        if num_ensembles > len(base_model_combinations):
            logging.error("Requested number of ensembles exceeds possible combinations.")
            raise ValueError("Cannot create more ensembles than the number of possible model combinations.")

        # Select random combinations to form each ensemble
        selected_combinations = random.sample(base_model_combinations, num_ensembles)

        # Create VotingClassifier ensembles with selected combinations
        ensembles = []
        for combination in selected_combinations:
            logging.debug("Creating ensemble with combination: %s", combination)
            estimators = [(name, classifiers[name]) for name in combination]
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensembles.append(ensemble)
            logging.info("Ensemble created with models: %s", [name for name, _ in estimators])

        logging.info("Total ensembles created: %d", len(ensembles))
        return ensembles
