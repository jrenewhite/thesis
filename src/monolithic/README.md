# Ensemble Learning Project (Monolithic Solution)

This project demonstrates the implementation of ensemble learning models using Python. It includes the creation, evaluation, and visualization of ensemble models built with different base learners in a monolithic setup. The project also features logging capabilities for tracking processes and monitoring results.

## Introduction

Ensemble learning is a machine learning paradigm where multiple models (learners) are combined to solve a problem, often resulting in better performance than individual models. This project explores the use of ensemble techniques such as voting classifiers with different base estimators.

## Installation (Non-Docker Version)

1. Download the repository:

   ```sh
   git clone https://github.com/jrenewhite/thesis.git
   cd thesis
   ```

2. Create a virtual environment (optional but recommended):

   ```sh
   python -m venv env
   ```

3. Activate the virtual environment:
   - For Linux/macOS:

     ```sh
     source env/bin/activate
     ```

   - For Windows:

     ```sh
     .\env\Scripts\activate
     ```

4. Install the dependencies:

   ```sh
   pip install -r requirements.txt
   ```

5. Run the `main.py` script with the directory path as an argument:

   ```sh
   python src/main.py <config_directory>
   ```

   - Example:

     ```sh
     python src/main.py data/monolithic/configurations/wine/
     ```

## Installation and Usage (Docker Version)

1. Ensure you have Docker installed on your machine. If not, follow the instructions to install Docker from [Docker's official website](https://docs.docker.com/get-docker/).

2. Build the Docker image:

   ```sh
   docker build -t ensemble-monolithic .
   ```

3. Run the Docker container with a specific configuration directory and mount the necessary data and logs:

   ```sh
   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/logs:/app/logs" ensemble-monolithic data/monolithic/configurations/wine
   ```

   This command mounts the `data` and `logs` directories from your local machine to ensure that results and logs are accessible outside the container.

4. To process all configurations within the `data/monolithic/configurations` directory, simply run:

   ```sh
   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/logs:/app/logs" ensemble-monolithic
   ```

   This command will execute the `main.py` script for every configuration found in the subdirectories of `data/monolithic/configurations`.

## Usage

1. Prepare your configuration files:
   - `system.json`: Contains system-specific configurations such as dataset delimiter and header presence.
   - `job.json`: Contains job-specific configurations such as dataset path, feature columns, target column, and split ratio.

   Example `job.json`:

   ```json
   {
       "dataset_path": "data/datasets/wine/wine.data",
       "is_split": false,
       "base_models": ["rf", "svm", "nb", "lr"],
       "num_base_models_in_each_ensemble": 3,
       "num_ensembles": 2,
       "feature_names": [
           "feature1", "feature2", "feature3", "feature4", "feature5",
           "feature6", "feature7", "feature8", "feature9", "feature10",
           "feature11", "feature12", "feature13"
       ],
       "target_col": "target",
       "split_ratio": 0.2,
       "results_folder": "data/monolithic/results/wine",
       "use_feature_range": false,
       "feature_range": ["feature1", "feature13"],
       "feature_mapping_path": "data/monolithic/configurations/wine/feature_mapping.csv"
   }
   ```

   Example `system.json`:

   ```json
   {
       "delimeter": ",",
       "header_present": 0
   }
   ```

2. Organize your directories:
   - Place the `system.json` and `job.json` files in a directory, such as `data/monolithic/configurations/wine`. The main script will accept this directory as an input parameter.
   - You can also have multiple subdirectories within the `monolithic/configurations` directory, each containing `system.json` and `job.json` files.

3. Run the `main.py` script with the directory path as an argument, either directly or via Docker.

## Logging

The project includes logging to track processes and output key information:

- **Console Logging**: Displays logs in the terminal for real-time tracking.
- **File Logging**: Logs are saved to a `logs` directory, which can be mounted outside the Docker container for persistence.

Example log output:

```plaintext
INFO - Starting training for ensemble model
DEBUG - Training latency for ensemble 1: 3.456 seconds
WARNING - Metrics files for ensemble XYZ not found
INFO - Confusion matrix plot saved to data/monolithic/results/wine/confusion_matrix_test_XYZ.png
```

## File Structure

- **src/main.py**: The main script that orchestrates the entire process, from loading configurations to training models, evaluating them, and saving results.

- **src/trainer.py**: Contains the `Trainer` class, responsible for training machine learning models using the provided data.

- **src/metrics_evaluator.py**: Contains the `MetricsEvaluator` class, which evaluates the performance metrics of trained models.

- **src/evaluation.py**: Contains the `evaluate_and_save` function, which evaluates the trained models and saves the results, such as performance metrics and visualizations.

- **src/data_loader.py**: Contains the `DataLoader` class, responsible for loading and preprocessing datasets from various sources.

- **src/result_saver.py**: Contains the `ResultSaver` class, which saves the results of model evaluations, such as performance metrics and predictions.

- **src/plotter.py**: Contains the `Plotter` class, which provides utilities for creating visualizations to analyze the results of model evaluations.

- **data/monolithic/configurations**: Contains configuration files (`system.json`, `job.json`) for different datasets.

- **data/monolithic/results**: Contains the output results for different datasets, including performance metrics and plots.

- **logs**: Stores logs generated by the application for each execution, including both console and file logs when running via Docker.

- **README.md**: This file provides information about the project, including installation instructions, usage examples, and a description of the file structure.

- **requirements.txt**: Lists the required Python packages and their versions for running the project. It is used for dependency management and installation.

## Additional Notes

- Ensure that your `system.json` and `job.json` files are correctly structured according to the provided examples.
- The project is designed to be flexible, allowing you to easily extend it by adding new base models or ensemble techniques.
- Configuration files (`system.json` and `job.json`) might not be present in the repository, as they could be located outside the project directory on the user's machine.
- For production environments, ensure that the Docker container is set up with appropriate resource limits to handle the data sizes and computational load.
