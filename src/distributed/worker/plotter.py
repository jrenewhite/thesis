import os
import json
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class Plotter:
    @staticmethod
    def create_latency_df(ensemble_names, results_folder):
        """
        Create a DataFrame containing latency data for each ensemble model.

        Args:
            ensemble_names (list): List of ensemble model names.
            results_folder (str): Path to the folder containing results.

        Returns:
            pd.DataFrame: DataFrame with latency data.
        """
        logging.info("Creating latency DataFrame for ensemble models")
        latency_data = {'Ensemble Model': [], 'Latency': [], 'Set': []}

        for ensemble_name in ensemble_names:
            # Load metrics JSON file for the ensemble and training set
            train_file_name = f"{ensemble_name}_train_metrics.json"
            train_file_path = os.path.join(results_folder, train_file_name)

            if os.path.isfile(train_file_path):
                with open(train_file_path, 'r') as file:
                    metrics = json.load(file)
                    train_latency = metrics.get('train_latency')
                    latency_data['Ensemble Model'].append(ensemble_name)
                    latency_data['Latency'].append(train_latency)
                    latency_data['Set'].append('Train')
                    logging.debug("Train latency for %s: %f seconds", ensemble_name, train_latency)
            else:
                logging.warning("Train metrics file for %s not found at %s", ensemble_name, train_file_path)

            # Load metrics JSON file for the ensemble and testing set
            test_file_name = f"{ensemble_name}_test_metrics.json"
            test_file_path = os.path.join(results_folder, test_file_name)

            if os.path.isfile(test_file_path):
                with open(test_file_path, 'r') as file:
                    metrics = json.load(file)
                    test_latency = metrics.get('test_latency')
                    latency_data['Ensemble Model'].append(ensemble_name)
                    latency_data['Latency'].append(test_latency)
                    latency_data['Set'].append('Test')
                    logging.debug("Test latency for %s: %f seconds", ensemble_name, test_latency)
            else:
                logging.warning("Test metrics file for %s not found at %s", ensemble_name, test_file_path)

        logging.info("Latency DataFrame created successfully")
        return pd.DataFrame(latency_data)

    
    @staticmethod
    def plot_latency(latency_df, results_folder):
        """
        Plot latency data for each ensemble model.

        Args:
            latency_df (pd.DataFrame): DataFrame containing latency data.
            results_folder (str): Path to the folder to save the plot.
        """
        logging.info("Plotting latency data for ensemble models")
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Ensemble Model', y='Latency', hue='Set', data=latency_df)
        plt.xlabel('Ensemble Model')
        plt.ylabel('Latency (seconds)')
        plt.title('Training and Testing Latency for Ensembles')
        plt.legend(title='Set')
        plt.tight_layout()
        plot_path = os.path.join(results_folder, 'latency_plot.png')
        plt.savefig(plot_path)
        logging.info("Latency plot saved to %s", plot_path)

    @staticmethod
    def create_sample_count_df(X_train, X_test, ensemble_names):
        """
        Create a DataFrame containing sample count data for each ensemble model.

        Args:
            X_train (array-like): Features of the training set.
            X_test (array-like): Features of the testing set.
            ensemble_names (list): List of ensemble model names.

        Returns:
            pd.DataFrame: DataFrame with sample count data.
        """
        logging.info("Creating sample count DataFrame for ensemble models")
        sample_count_data = {
            'Ensemble Model': ensemble_names * 2,
            'Number of Samples': [len(X_train)] * len(ensemble_names) + [len(X_test)] * len(ensemble_names),
            'Set': ['Train'] * len(ensemble_names) + ['Test'] * len(ensemble_names)
        }
        logging.info("Sample count DataFrame created successfully")
        return pd.DataFrame(sample_count_data)

    @staticmethod
    def plot_sample_count(sample_count_df, results_folder):
        """
        Plot sample count data for each ensemble model.

        Args:
            sample_count_df (pd.DataFrame): DataFrame containing sample count data.
            results_folder (str): Path to the folder to save the plot.
        """
        logging.info("Plotting sample count data for ensemble models")
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Ensemble Model', y='Number of Samples', hue='Set', data=sample_count_df)
        plt.xlabel('Ensemble Model')
        plt.ylabel('Number of Samples')
        plt.title('Number of Samples in Train and Test Sets for Ensembles')
        plt.legend(title='Set')
        plt.tight_layout()
        plot_path = os.path.join(results_folder, 'sample_count_plot.png')
        plt.savefig(plot_path)
        logging.info("Sample count plot saved to %s", plot_path)
