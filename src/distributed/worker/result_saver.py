import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import logging

class ResultSaver:
    @staticmethod
    def save_results(train_df, test_df, ensemble_output_df, results_folder):
        """
        Save ensemble performance data and output to CSV files.

        Args:
            train_df (pd.DataFrame): DataFrame containing performance metrics for the training set.
            test_df (pd.DataFrame): DataFrame containing performance metrics for the testing set.
            ensemble_output_df (pd.DataFrame): DataFrame containing ensemble output data.
            results_folder (str): Path to the folder to save the results.
        """
        logging.info("Saving ensemble performance data to CSV files in %s", results_folder)
        os.makedirs(results_folder, exist_ok=True)
        
        train_path = os.path.join(results_folder, 'ensemble_performance_train.csv')
        test_path = os.path.join(results_folder, 'ensemble_performance_test.csv')
        output_path = os.path.join(results_folder, 'ensemble_output.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        ensemble_output_df.to_csv(output_path, index=False)
        
        logging.info("Train performance saved to %s", train_path)
        logging.info("Test performance saved to %s", test_path)
        logging.info("Ensemble output saved to %s", output_path)
    
    @staticmethod
    def save_confusion_matrices(ensemble_names, train_metrics, test_metrics, results_folder):
        """
        Save confusion matrices as PNG images for each ensemble model.

        Args:
            ensemble_names (list): List of ensemble model names.
            train_metrics (list): List of dictionaries containing training set metrics for each ensemble.
            test_metrics (list): List of dictionaries containing testing set metrics for each ensemble.
            results_folder (str): Path to the folder to save the confusion matrices.
        """
        logging.info("Saving confusion matrices for ensemble models in %s", results_folder)

        for ensemble_name in ensemble_names:
            train_metrics_file = os.path.join(results_folder, f'{ensemble_name}_train_metrics.json')
            test_metrics_file = os.path.join(results_folder, f'{ensemble_name}_test_metrics.json')
            
            if os.path.isfile(train_metrics_file) and os.path.isfile(test_metrics_file):
                with open(train_metrics_file, 'r') as train_file, open(test_metrics_file, 'r') as test_file:
                    train_metrics = json.load(train_file)
                    test_metrics = json.load(test_file)

                cm_train = train_metrics['confusion_matrix']
                cm_test = test_metrics['confusion_matrix']

                train_plot_path = os.path.join(results_folder, f'confusion_matrix_train_{ensemble_name.lower()}.png')
                test_plot_path = os.path.join(results_folder, f'confusion_matrix_test_{ensemble_name.lower()}.png')

                ResultSaver._save_confusion_matrix_plot(cm_train, train_plot_path, f"Confusion Matrix - Training Set - {ensemble_name}")
                ResultSaver._save_confusion_matrix_plot(cm_test, test_plot_path, f"Confusion Matrix - Test Set - {ensemble_name}")
                
                logging.info("Training confusion matrix saved to %s", train_plot_path)
                logging.info("Test confusion matrix saved to %s", test_plot_path)
            else:
                logging.warning("Metrics files for ensemble %s not found in %s", ensemble_name, results_folder)

    @staticmethod
    def _save_confusion_matrix_plot(confusion_matrix, save_path, title):
        """
        Save a confusion matrix plot as a PNG image.

        Args:
            confusion_matrix (array-like): Confusion matrix data.
            save_path (str): Path to save the plot.
            title (str): Title for the plot.
        """
        logging.info("Saving confusion matrix plot to %s", save_path)
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.array(confusion_matrix), annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig(save_path)
        plt.close()
        logging.info("Confusion matrix plot saved to %s", save_path)
