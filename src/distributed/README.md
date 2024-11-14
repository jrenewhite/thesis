# Distributed K-Fold Evaluation Framework

This project implements a distributed K-Fold machine learning evaluation framework using Docker, where a coordinator node assigns tasks (folds) to worker nodes. The project is designed for flexible configuration and scalable distributed evaluation of machine learning models.

## Project Structure

```plaintext
.
├── coordinator
│   ├── data_loader.py
│   ├── Dockerfile
│   ├── evaluation.py
│   ├── main.py
│   ├── metrics_evaluator.py
│   ├── model_initializer.py
│   ├── plotter.py
│   ├── requirements.txt
│   ├── result_saver.py
│   └── trainer.py
├── worker
│   ├── data_loader.py
│   ├── Dockerfile
│   ├── evaluation.py
│   ├── main.py
│   ├── metrics_evaluator.py
│   ├── model_initializer.py
│   ├── plotter.py
│   ├── requirements.txt
│   ├── result_saver.py
│   └── trainer.py
├── docker-compose.yml
├── docker-compose.override.wine.yml
└── data
    ├── distributed
    │   ├── configurations
    │   │   └── wine
    │   │       ├── job.json
    │   │       └── system.json
    │   └── datasets
    │       └── wine
    │           └── wine.data
    └── monolithic
```

- `coordinator/`: Contains the coordinator code and configuration.
- `worker/`: Contains the worker code responsible for processing assigned tasks (folds).
- `data/`: Stores configurations and datasets for different tasks and datasets.

## Requirements

The framework relies on Docker and Docker Compose. Make sure both are installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Setup and Configuration

### 1. Docker Compose Configuration

Each dataset has a separate Docker Compose override file to specify its configuration. For example, the `docker-compose.override.wine.yml` file configures the system for the "wine" dataset.

To configure additional datasets:

1. Create a `docker-compose.override.<dataset>.yml` file based on the structure in `docker-compose.override.wine.yml`.
2. Specify the dataset-specific configuration folder path for each dataset.

### 2. Dataset and Configuration

Each dataset configuration should have:

- `job.json`: Defines dataset paths, K-Fold setup, model parameters, and evaluation options.
- `system.json`: Contains system-wide configurations like delimiter, header settings, and feature mappings.

### Example Configuration (`job.json` for Wine Dataset)

```json
{
    "dataset_path": "data/datasets/wine/wine.data",
    "is_split": false,
    "base_models": ["rf", "svm", "lr", "et"],
    "num_base_models_in_each_ensemble": 3,
    "num_ensembles": 3,
    "feature_names": ["feature1", "feature2", ..., "feature13"],
    "target_col": "target",
    "split_ratio": 0.2,
    "k": 5,
    "results_folder": "data/distributed/results/wine",
    "use_feature_range": false,
    "feature_mapping_path": "data/distributed/configurations/wine/feature_mapping.csv",
    "pretrained_ensemble_path": "data/distributed/pretrained_ensembles/wine",
    "ignore_pretrained": false,
    "model_params": { ... }
}
```

### 3. Install Dependencies

The dependencies for each component are listed in `requirements.txt` in the `coordinator` and `worker` directories. These are automatically installed in the Docker containers.

## Running the Distributed Framework

### 1. Start the Coordinator and Workers

To start the distributed setup for a specific dataset (e.g., wine), use:

```bash
docker-compose -f docker-compose.yml -f docker-compose.override.wine.yml up --build
```

This command will:

- Build the Docker images for the coordinator and workers.
- Start the coordinator, which assigns folds to each worker.
- Process each fold and collect results in the specified `results_folder`.

### 2. Shutting Down the Framework

To gracefully stop the distributed framework:

```bash
docker-compose -f docker-compose.yml -f docker-compose.override.wine.yml down
```

## Logs and Results

- Logs for each component (coordinator and workers) are stored in `/app/logs/` within the container. You can customize this path via environment variables.
- Results are saved in the `results_folder` specified in each `job.json` configuration file.

## Customizing K-Fold Evaluation

To change the number of folds (e.g., 10), modify the `k` value in `job.json` for the specific dataset configuration.

## Additional Notes

- **Debugging**: Check logs in `coordinator` and `worker` containers if issues arise during communication or task processing.
- **Pre-trained Ensembles**: If `ignore_pretrained` is set to `false`, ensure pretrained models are saved in the specified path; otherwise, they will be retrained.

This setup provides a modular and scalable way to evaluate models across different datasets and configurations with K-Fold cross-validation in a distributed environment.
