import os
import json
import numpy as np
import time
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class MetricsEvaluator:
    @staticmethod
    def evaluate_performance_metrics(ensembles, X_train, X_test, y_train, y_test, results_folder, data_type):
        """
        Evaluate performance metrics for ensembles.

        Args:
            ensembles (list): List of ensemble models.
            X_train (array-like): Features of the training set.
            X_test (array-like): Features of the testing set.
            y_train (array-like): Target of the training set.
            y_test (array-like): Target of the testing set.
            results_folder (str): Path to the folder to save results.
            data_type (str): Type of data ('train' or 'test').

        Returns:
            list: List of tuples containing performance metrics for each ensemble.
        """
        logging.info("Starting evaluation of performance metrics for %s set", data_type)
        metrics = []
        ensemble_names = ['_'.join([name for name, _ in ensemble.named_estimators.items()]) for ensemble in ensembles]

        for i, ensemble in enumerate(ensembles, 1):
            logging.info("Evaluating ensemble %d: %s", i, ensemble_names[i - 1])

            # Measure training time (latency)
            start_time_train = time.time()
            ensemble.fit(X_train, y_train)
            end_time_train = time.time()
            train_latency = end_time_train - start_time_train
            logging.debug("Training latency for ensemble %d: %f seconds", i, train_latency)

            # Select the data type
            if data_type == 'train':
                X, y = X_train, y_train
            elif data_type == 'test':
                X, y = X_test, y_test
            else:
                logging.error("Invalid data type specified: %s", data_type)
                raise ValueError("Invalid data type. Use 'train' or 'test'.")

            # Measure testing time (latency)
            start_time_test = time.time()
            y_pred = ensemble.predict(X)
            end_time_test = time.time()
            test_latency = end_time_test - start_time_test
            logging.debug("Testing latency for ensemble %d: %f seconds", i, test_latency)

            # Calculate performance metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='macro')
            recall = recall_score(y, y_pred, average='macro')
            f1 = f1_score(y, y_pred, average='macro')
            confusion_mat = confusion_matrix(y, y_pred)
            logging.info("Metrics for ensemble %d - Accuracy: %f, Precision: %f, Recall: %f, F1 Score: %f", 
                         i, accuracy, precision, recall, f1)

            # Append metrics to list
            metrics.append((accuracy, precision, recall, f1, confusion_mat, train_latency, test_latency))

            # Save metrics to file
            ensemble_name = ensemble_names[i - 1]
            MetricsEvaluator.save_metrics(metrics[i-1], ensemble_name, data_type, results_folder)

        logging.info("Completed evaluation of performance metrics for %s set", data_type)
        return metrics

    @staticmethod
    def save_metrics(metrics, ensemble_name, data_type, results_folder):
        """
        Save performance metrics to a JSON file.

        Args:
            metrics (tuple): Tuple containing performance metrics.
            ensemble_name (str): Name of the ensemble.
            data_type (str): Type of data ('train' or 'test').
            results_folder (str): Path to the folder to save results.
        """
        metrics_file = os.path.join(results_folder, f'{ensemble_name}_{data_type}_metrics.json')
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

        # Convert NumPy arrays to lists
        metrics_dict = {
            'accuracy': metrics[0],
            'precision': metrics[1],
            'recall': metrics[2],
            'f1_score': metrics[3],
            'confusion_matrix': metrics[4].tolist(),
            'train_latency': metrics[5],
            'test_latency': metrics[6]
        }

        # Save metrics to JSON file
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logging.info("Metrics saved for ensemble %s (%s set) to %s", ensemble_name, data_type, metrics_file)
