# Ensemble Learning Project

This project implements ensemble learning models in Python, including training, evaluation, and visualization. It supports both **monolithic** and **distributed** configurations, allowing flexibility for different machine learning setups.

- **Monolithic Version**: Runs on a single machine, handling all processes from data loading to result visualization in a straightforward, standalone setup.
- **Distributed Version**: Designed for a distributed environment, where ensemble models are executed across several devices, either as separate physical machines or Docker containers within a network.

## Repository

To clone this project:

```sh
git clone https://github.com/jrenewhite/thesis.git
cd thesis
```

---

## Monolithic Version

### Introduction

The monolithic solution is designed to combine multiple machine learning models using ensemble learning techniques, such as voting classifiers with various base estimators. This setup is effective in a single-machine environment, often yielding better performance than individual models.

### Installation (Non-Docker Version)

1. **Set Up the Environment**:

   ```sh
   python -m venv env
   source env/bin/activate  # For Linux/macOS
   .\env\Scripts\activate   # For Windows
   ```

2. **Install Dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Main Script**:

   ```sh
   python src/main.py <config_directory>
   # Example:
   python src/main.py data/monolithic/configurations/wine/
   ```

### Installation and Usage (Docker Version)

1. **Build the Docker Image**:

   ```sh
   docker build -t ensemble-monolithic .
   ```

2. **Run the Docker Container with Specific Configuration**:

   ```sh
   docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/data/monolithic/results" ensemble-monolithic data/monolithic/configurations/wine
   ```

### Configuration Files

- **`system.json`**: Defines system settings, such as dataset delimiters and header specifications.
- **`job.json`**: Specifies dataset paths, base models, split ratios, and output locations for each experiment.

**Example `job.json`**:

```json
{
    "dataset_path": "data/datasets/wine/wine.data",
    "is_split": false,
    "base_models": ["rf", "svm", "nb", "lr"],
    "num_base_models_in_each_ensemble": 3,
    "num_ensembles": 2,
    "feature_names": ["feature1", "feature2", ...],
    "target_col": "target",
    "split_ratio": 0.2,
    "results_folder": "data/monolithic/results/wine"
}
```

### Monolithic Directory Structure

- **`src/main.py`**: Orchestrates the monolithic pipeline from data loading to result storage.
- **`src/trainer.py`**: Manages the training process for individual models and ensembles.
- **`src/metrics_evaluator.py`**: Calculates performance metrics like accuracy, precision, recall, and F1-score.
- **`src/plotter.py`**: Generates latency and sample count visualizations.
- **`data/monolithic/configurations`**: Contains configurations for various datasets.
- **`data/monolithic/results`**: Stores output results for different datasets, including performance metrics and plots.

---

## Distributed Version

The distributed solution enables deployment across multiple machines or Docker containers. This setup allows ensemble learning tasks to be split across a network for parallel processing, handling large datasets or complex models in a scalable environment.

### Deployment on Multiple Physical Machines

1. **Set Up Machines**:
   - Designate one machine as the **client** (or "organizer") and others as **workers**.
   - Place the code and configurations on each machine according to their roles (refer to `src/distributed/Client Machine` and `src/distributed/Worker Machine` for details).

2. **Configure Connection Details**:
   - Update the client machine’s configuration to include IP addresses or hostnames of each worker machine.
   - Each worker should run `server.py` to listen for tasks from the client.

3. **Run the Client and Workers**:
   - Start the workers by running their `server.py` script.
   - Start the client machine’s `main.py`, specifying the worker addresses as needed.

*Future Note*: The client can be configured in future versions to provide its address to each worker, allowing the workers to report back and be managed more dynamically.

### Deployment Using Docker (Docker Network Setup)

To simulate a distributed setup on a single machine with enough resources, you can create a Docker network with multiple containers, each representing a worker or client.

1. **Docker Compose File**: Use the `docker-compose.yml` to configure a client and multiple worker services.

2. **Sample `docker-compose.yml`**:

   ```yaml
   version: '3.8'

   services:
     client:
       image: ensemble-client  # Client image
       build:
         context: .
         dockerfile: ./src/distributed/Client\ Machine/Dockerfile
       networks:
         - ensemble_net
       depends_on:
         - worker1
         - worker2

     worker1:
       image: ensemble-worker  # Worker image
       build:
         context: .
         dockerfile: ./src/distributed/Worker\ Machine/Dockerfile
       networks:
         - ensemble_net

     worker2:
       image: ensemble-worker
       build:
         context: .
         dockerfile: ./src/distributed/Worker\ Machine/Dockerfile
       networks:
         - ensemble_net

   networks:
     ensemble_net:
       driver: bridge
   ```

3. **Build and Run with Docker Compose**:

   ```sh
   docker-compose up --build
   ```

4. **Scaling Workers**:
   To add more worker containers, you can add more `workerN` services to `docker-compose.yml` or scale directly with:

   ```sh
   docker-compose up --scale worker=4  # Replace 4 with the desired number of workers
   ```

### Distributed Directory Structure

- **`src/distributed/Client Machine`**: Contains the client code, which coordinates the distributed tasks and manages results from workers.
- **`src/distributed/Worker Machine`**: Contains the worker code, including classifier evaluation and network settings.
- **`src/distributed/Socket Program Demo`**: Contains socket connection demos to facilitate inter-machine communication and testing.

---

## Requirements

Dependencies are listed in `requirements.txt`, including libraries like `scikit-learn`, `pandas`, and `seaborn`. To install:

```sh
pip install -r requirements.txt
```

---

## Additional Notes

- **Configuration Files**: Ensure `system.json` and `job.json` files are correctly structured and available in the configuration directories for both setups.
- **Flexible Deployment**: The distributed version can be deployed on physical machines or Docker containers, adapting easily to different environments.
- **Future Modifications**: The distributed setup will support dynamic client-worker communication, where each worker reports to the client for better management.

This `README.md` provides a comprehensive guide for deploying and using both versions of the project. Let me know if you need further customization or if any additional details are required!
