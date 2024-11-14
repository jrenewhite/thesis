# Ensemble Learning Project (Monolithic Solution)

This project demonstrates ensemble learning models in a monolithic setup using various base learners. It includes training, evaluating, and visualizing ensemble models. The project also supports loading pre-trained models and includes logging to monitor processes and results.

## Introduction

Ensemble learning combines multiple models (learners) to solve problems, often improving performance. This project explores ensemble techniques such as voting classifiers with different base estimators.

## Installation (Non-Docker Version)

1. Clone the repository:

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

4. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

5. Run the `main.py` script with the path to a configuration directory:

   ```sh
   python src/main.py <config_directory>
   ```

   - Example:

     ```sh
     python src/main.py data/monolithic/configurations/wine/
     ```

## Installation and Usage (Docker Version)

1. Ensure Docker is installed on your machine. Instructions are available on [Docker's website](https://docs.docker.com/get-docker/).

2. Build the Docker image:

   ```sh
   docker build -t ensemble-monolithic .
   ```

3. Run the Docker container with a configuration directory and mount data and logs for external access:

   ```sh
   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/logs:/app/logs" ensemble-monolithic data/monolithic/configurations/wine
   ```

   This mounts the `data` and `logs` directories from your machine to the container, preserving results and logs outside the container.

4. To process all configurations within `data/monolithic/configurations`:

   ```sh
   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/logs:/app/logs" ensemble-monolithic
   ```

## Configuration Files

Each dataset has a corresponding `job.json` in `data/monolithic/configurations/[dataset_name]/`. System-specific settings are stored in `system.json`.

### Sample `job.json`

```json
{
    "dataset_path": "data/datasets/iris/iris.data",
    "is_split": false,
    "base_models": ["rf", "svm", "nb", "lr"],
    "num_base_models_in_each_ensemble": 3,
    "num_ensembles": 2,
    "feature_names": ["feature1", "feature2", "feature3", "feature4"],
    "target_col": "target",
    "split_ratio": 0.2,
    "results_folder": "data/monolithic/results/iris",
    "use_feature_range": false,
    "feature_range": ["feature1", "feature4"],
    "feature_mapping_path": "data/monolithic/configurations/iris/feature_mapping.csv",
    "pretrained_ensemble_path": "data/pretrained_ensembles/iris",
    "ignore_pretrained": false,
    "model_params": {
        "rf": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 4, "random_state": 42},
        "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": true, "random_state": 42},
        "lr": {"solver": "liblinear", "C": 0.5, "max_iter": 200}
    }
}
```

### Sample `system.json`

```json
{
    "delimiter": ",",
    "header_present": 0
}
```

### Configuration Fields

- **dataset_path**: Path to dataset files, either as a single path or a list for separate train/test sets.
- **is_split**: `true` if separate train/test sets are provided; otherwise, `false`.
- **base_models**: List of base models (`rf`, `svm`, `nb`, `lr`, etc.).
- **num_base_models_in_each_ensemble**: Number of base models per ensemble.
- **num_ensembles**: Total number of ensembles to create.
- **feature_names**: List of feature names.
- **target_col**: Name of the target column.
- **split_ratio**: Ratio for train/test split when `is_split` is `false`.
- **results_folder**: Directory to save results.
- **use_feature_range**: `true` if only specified features should be used.
- **feature_range**: List of features to use if `use_feature_range` is `true`.
- **feature_mapping_path**: Path to a CSV file mapping feature types if needed.
- **pretrained_ensemble_path**: Directory for pre-trained ensembles.
- **ignore_pretrained**: `true` if pre-trained models should be ignored, forcing a re-train.
- **model_params**: Dictionary of hyperparameters for specific models.

## Usage

1. **Prepare Configurations**: Ensure `job.json` and `system.json` are in each dataset’s configuration folder.
2. **Organize Directories**:
   - Place `job.json` and `system.json` in `data/monolithic/configurations/[dataset_name]`.
3. **Run `main.py`**: Either directly or with Docker, specify the configuration directory as a command-line argument.

## Logging

The project includes comprehensive logging:

- **Console Logging**: Displays logs in the terminal.
- **File Logging**: Saves logs in the `logs` directory. Mount this directory when running via Docker to access logs outside the container.

Example log output:

```plaintext
INFO - Starting training for ensemble model
DEBUG - Training latency for ensemble 1: 3.456 seconds
WARNING - Metrics files for ensemble XYZ not found
INFO - Confusion matrix plot saved to data/monolithic/results/wine/confusion_matrix_test_XYZ.png
```

## File Structure

- **src/main.py**: Main script to load configurations, train models, evaluate, and save results.
- **src/trainer.py**: The `Trainer` class, handling data loading, model initialization, and training.
- **src/evaluation.py**: The `evaluate_and_save` function to evaluate models and save results.
- **src/data_loader.py**: The `DataLoader` class for loading datasets.
- **src/result_saver.py**: The `ResultSaver` class to save evaluation results.
- **src/plotter.py**: The `Plotter` class for visualizing results.
- **data/monolithic/configurations**: Configuration files (`system.json` and `job.json`) for each dataset.
- **data/monolithic/results**: Directory for saved results.
- **logs**: Stores logs for each run.
- **requirements.txt**: Lists Python packages for the project.

## Running Pre-trained Ensembles

- **Pre-trained Ensembles**: If `pretrained_ensemble_path` is specified and the models exist, the application will load them unless `ignore_pretrained` is set to `true`.
- **Forcing Re-training**: Set `"ignore_pretrained": true` in `job.json` or omit the `pretrained_ensemble_path`.
