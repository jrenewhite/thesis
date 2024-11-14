import logging
import os
import warnings
from data_loader import DataLoader
from model_initializer import ModelInitializer
from sklearn.exceptions import DataConversionWarning
import joblib  # For saving and loading pre-trained models
import pandas as pd

class Trainer:
    def __init__(self, config, syst_config):
        """
        Initializes the Trainer object with the given configurations.

        Args:
            config (dict): Dictionary containing training configurations.
            syst_config (dict): Dictionary containing system configurations.
        """
        self.config = config
        self.syst_config = syst_config
        self.logger = logging.getLogger(__name__)
        self.logger.info("Trainer initialized with configuration and system configuration")

        # Set defaults for large dataset handling if not provided in the config
        self.large_dataset_threshold = self.config.get('large_dataset_threshold', 50000)
        self.sampling_ratio = self.config.get('sampling_ratio', 1.0)

        # Ensemble save/load path, only if specified in config
        self.ensemble_save_path = self.config.get('pretrained_ensemble_path')
        if self.ensemble_save_path and os.path.exists(self.ensemble_save_path):
            self.logger.info(f"Pre-trained ensemble directory set to: {self.ensemble_save_path}")
        else:
            self.logger.info("No pre-trained ensemble path provided; proceeding with new training.")

        # Check for flag to ignore pre-trained ensembles, defaulting to False
        self.ignore_pretrained = self.config.get('ignore_pretrained', False)
        if self.ignore_pretrained:
            self.logger.info("Ignore pre-trained flag is set; training will not use pre-trained ensembles.")
        else:
            self.logger.info("Using available pre-trained ensembles unless specified otherwise.")

    def load_or_train_ensemble(self, ensemble_name, ensemble):
        """
        Load an ensemble from disk if it exists, otherwise train and save it.

        Args:
            ensemble_name (str): The name of the ensemble.
            ensemble: The ensemble model object.

        Returns:
            Trained ensemble model.
        """
        if self.ensemble_save_path:
            ensemble_path = os.path.join(self.ensemble_save_path, f"{ensemble_name}.joblib")

            # Load pre-trained ensemble if not ignoring and file exists
            if not self.ignore_pretrained and os.path.exists(ensemble_path):
                self.logger.info(f"Loading pre-trained ensemble '{ensemble_name}' from {ensemble_path}")
                return joblib.load(ensemble_path)
            else:
                self.logger.info(f"Pre-trained ensemble '{ensemble_name}' not found or ignored, training new ensemble")

        # Train the ensemble and save it if save path is available
        self.logger.info(f"Training ensemble '{ensemble_name}'")
        ensemble.fit(self.X_train, self.y_train.values.ravel())
        
        # Save the ensemble if a save path is specified
        if self.ensemble_save_path:
            os.makedirs(self.ensemble_save_path, exist_ok=True)
            joblib.dump(ensemble, os.path.join(self.ensemble_save_path, f"{ensemble_name}.joblib"))
            self.logger.info(f"Ensemble '{ensemble_name}' saved to {self.ensemble_save_path}")

        return ensemble

    def train(self):
        """
        Trains classifiers and creates ensembles based on the provided configurations.

        Returns:
            X_train (pd.DataFrame): Features of the training set.
            X_test (pd.DataFrame): Features of the testing set.
            y_train (pd.Series): Target labels of the training set.
            y_test (pd.Series): Target labels of the testing set.
            ensembles (list): List of trained ensemble models.
        """
        self.logger.info("Starting training process")

        # Capture specific warnings from scikit-learn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DataConversionWarning)

            # Load dataset
            if self.config['is_split']:
                self.logger.info("Loading dataset with split configuration")
                if not isinstance(self.config['dataset_path'], list) or len(self.config['dataset_path']) != 2:
                    self.logger.error("When 'is_split' is True, 'dataset_path' should be a list of length 2.")
                    raise ValueError("When 'is_split' is True, 'dataset_path' should be a list of length 2.")

                X, y = DataLoader.load_dataset(
                    dataset_path=self.config['dataset_path'][0],
                    feature_names=self.config['feature_names'],
                    target_name=self.config['target_col'],
                    delimiter=self.syst_config['delimeter'],
                    use_feature_range=self.config['use_feature_range'],
                    feature_range=self.config['feature_range'],
                    header_present=self.syst_config['header_present'],
                    feature_mapping_path=self.config['feature_mapping_path'],
                    large_dataset_threshold=self.large_dataset_threshold,
                    sampling_ratio=self.sampling_ratio
                )
                self.X_train, self.X_test, self.y_train, self.y_test = DataLoader.split_dataset(X, y, self.config['split_ratio'])
                self.logger.info("Dataset loaded and split successfully")
            else:
                self.logger.info("Loading dataset without split configuration")
                X, y = DataLoader.load_dataset(
                    dataset_path=self.config['dataset_path'],
                    feature_names=self.config['feature_names'],
                    target_name=self.config['target_col'],
                    delimiter=self.syst_config['delimeter'],
                    use_feature_range=self.config['use_feature_range'],
                    feature_range=self.config['feature_range'],
                    header_present=self.syst_config['header_present'],
                    feature_mapping_path=self.config['feature_mapping_path'],
                    large_dataset_threshold=self.large_dataset_threshold,
                    sampling_ratio=self.sampling_ratio
                )
                self.X_train, self.X_test, self.y_train, self.y_test = DataLoader.split_dataset(X, y, self.config['split_ratio'])
                self.logger.info("Dataset loaded and split successfully")

            # Log any warnings captured during dataset loading
            for warning in w:
                self.logger.warning(f"Warning captured during dataset loading: {warning.message}")

        # Initialize classifiers
        self.logger.info("Initializing classifiers")
        classifiers = ModelInitializer.initialize_classifiers(self.config['base_models'], self.config.get('model_params', None))
        self.logger.debug("Classifiers initialized: %s", classifiers)

        # Create ensembles
        self.logger.info("Creating ensembles")
        ensembles = ModelInitializer.create_ensembles(
            classifiers,
            self.config['num_base_models_in_each_ensemble'],
            self.config['num_ensembles']
        )
        self.logger.info("Finished creating ensembles")

        # Train or load each ensemble and store it
        trained_ensembles = []
        for ensemble in ensembles:
            # Generate ensemble name based on classifiers in ensemble
            ensemble_name = "_".join([name for name, _ in ensemble.named_estimators.items()])
            trained_ensemble = self.load_or_train_ensemble(ensemble_name, ensemble)
            trained_ensembles.append(trained_ensemble)

        self.logger.info("Training completed")
        return self.X_train, self.X_test, self.y_train, self.y_test, trained_ensembles
