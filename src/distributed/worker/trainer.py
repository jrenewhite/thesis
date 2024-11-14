import logging
import os
import warnings
from data_loader import DataLoader
from model_initializer import ModelInitializer
from sklearn.exceptions import DataConversionWarning
import joblib  # For saving and loading pre-trained models
import pandas as pd
from sklearn.model_selection import KFold

class Trainer:
    def __init__(self, config, syst_config, k_folds=5, fold_index=0):
        """
        Initializes the Trainer object with the given configurations, K-Fold parameters.

        Args:
            config (dict): Dictionary containing training configurations.
            syst_config (dict): Dictionary containing system configurations.
            k_folds (int): Number of folds for K-Fold cross-validation.
            fold_index (int): Index of the fold this Trainer instance should train on.
        """
        self.config = config
        self.syst_config = syst_config
        self.k_folds = k_folds
        self.fold_index = fold_index
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Trainer initialized for fold {self.fold_index} with K-Folds={self.k_folds}")

        self.large_dataset_threshold = self.config.get('large_dataset_threshold', 50000)
        self.sampling_ratio = self.config.get('sampling_ratio', 1.0)

        self.ensemble_save_path = self.config.get('pretrained_ensemble_path')
        if self.ensemble_save_path and os.path.exists(self.ensemble_save_path):
            self.logger.info(f"Pre-trained ensemble directory set to: {self.ensemble_save_path}")
        else:
            self.logger.info("No pre-trained ensemble path provided; proceeding with new training.")

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
            ensemble_path = os.path.join(self.ensemble_save_path, f"{ensemble_name}_fold{self.fold_index}.joblib")

            if not self.ignore_pretrained and os.path.exists(ensemble_path):
                self.logger.info(f"Loading pre-trained ensemble '{ensemble_name}' for fold {self.fold_index} from {ensemble_path}")
                return joblib.load(ensemble_path)
            else:
                self.logger.info(f"Pre-trained ensemble '{ensemble_name}' not found or ignored, training new ensemble")

        self.logger.info(f"Training ensemble '{ensemble_name}' for fold {self.fold_index}")
        ensemble.fit(self.X_train, self.y_train.values.ravel())
        
        if self.ensemble_save_path:
            os.makedirs(self.ensemble_save_path, exist_ok=True)
            joblib.dump(ensemble, ensemble_path)
            self.logger.info(f"Ensemble '{ensemble_name}' for fold {self.fold_index} saved to {self.ensemble_save_path}")

        return ensemble

    def train_k_fold(self):
        """
        Trains classifiers and creates ensembles using a specific fold of the K-Fold cross-validation.

        Returns:
            X_train (pd.DataFrame): Features of the training set.
            X_test (pd.DataFrame): Features of the testing set.
            y_train (pd.Series): Target labels of the training set.
            y_test (pd.Series): Target labels of the testing set.
            ensembles (list): List of trained ensemble models.
        """
        self.logger.info(f"Starting K-Fold training for fold {self.fold_index}")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DataConversionWarning)

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

            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
            folds = list(kf.split(X))

            if self.fold_index >= len(folds):
                self.logger.error("Fold index exceeds number of available folds.")
                raise ValueError("Fold index exceeds number of available folds.")

            train_indices, test_indices = folds[self.fold_index]
            self.X_train, self.X_test = X.iloc[train_indices], X.iloc[test_indices]
            self.y_train, self.y_test = y.iloc[train_indices], y.iloc[test_indices]
            self.logger.info(f"Fold {self.fold_index} - dataset split successfully")

            for warning in w:
                self.logger.warning(f"Warning captured during dataset loading: {warning.message}")

        self.logger.info("Initializing classifiers")
        classifiers = ModelInitializer.initialize_classifiers(self.config['base_models'], self.config.get('model_params', None))
        self.logger.debug("Classifiers initialized: %s", classifiers)

        self.logger.info("Creating ensembles")
        ensembles = ModelInitializer.create_ensembles(
            classifiers,
            self.config['num_base_models_in_each_ensemble'],
            self.config['num_ensembles']
        )
        self.logger.info("Finished creating ensembles")

        trained_ensembles = []
        for ensemble in ensembles:
            ensemble_name = "_".join([name for name, _ in ensemble.named_estimators.items()])
            trained_ensemble = self.load_or_train_ensemble(ensemble_name, ensemble)
            trained_ensembles.append(trained_ensemble)

        self.logger.info(f"Training completed for fold {self.fold_index}")
        return self.X_train, self.X_test, self.y_train, self.y_test, trained_ensembles
