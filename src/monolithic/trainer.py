import logging
import warnings
from data_loader import DataLoader
from model_initializer import ModelInitializer
from sklearn.exceptions import DataConversionWarning
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

    def train(self):
        """
        Trains classifiers and creates ensembles based on the provided configurations.

        Returns:
            X_train (pd.DataFrame): Features of the training set.
            X_test (pd.DataFrame): Features of the testing set.
            y_train (pd.Series): Target labels of the training set.
            y_test (pd.Series): Target labels of the testing set.
            ensembles (list): List of ensemble models.
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
                    feature_mapping_path=self.config['feature_mapping_path']
                )
                X_train, X_test, y_train, y_test = DataLoader.split_dataset(X, y, self.config['split_ratio'])
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
                    feature_mapping_path=self.config['feature_mapping_path']
                )
                X_train, X_test, y_train, y_test = DataLoader.split_dataset(X, y, self.config['split_ratio'])
                self.logger.info("Dataset loaded and split successfully")

            # Log any warnings captured during dataset loading
            for warning in w:
                self.logger.warning(f"Warning captured during dataset loading: {warning.message}")

        # Initialize classifiers
        self.logger.info("Initializing classifiers")
        classifiers = ModelInitializer.initialize_classifiers(self.config['base_models'])
        self.logger.debug("Classifiers initialized: %s", classifiers)

        # Create ensembles
        self.logger.info("Creating ensembles")
        ensembles = ModelInitializer.create_ensembles(
            classifiers,
            self.config['num_base_models_in_each_ensemble'],
            self.config['num_ensembles']
        )
        self.logger.info("Finished creating ensembles")

        # Train ensembles with additional warning capture
        for ensemble in ensembles:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", DataConversionWarning)
                ensemble.fit(X_train, y_train.values.ravel())  # Convert y_train to numpy and ravel it

                # Log warnings encountered during ensemble training
                for warning in w:
                    self.logger.warning(f"Warning during ensemble training ({ensemble}): {warning.message}")

        self.logger.info("Training completed")
        return X_train, X_test, y_train, y_test, ensembles
