import socket
import logging
import time
import os
import json
from trainer import Trainer
from evaluation import evaluate_and_save
from plotter import Plotter
from result_saver import ResultSaver

COORDINATOR_IP = 'coordinator'
COORDINATOR_PORT = 9999
RETRY_INTERVAL = 5

def connect_to_coordinator():
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                logging.info(f"Connecting to coordinator at {COORDINATOR_IP}:{COORDINATOR_PORT}")
                client_socket.connect((COORDINATOR_IP, COORDINATOR_PORT))
                
                while True:
                    # Receive configuration path and fold index
                    task = client_socket.recv(1024).decode()
                    if task == "shutdown":
                        logging.info("Received shutdown signal from coordinator.")
                        break
                    
                    # Parse the task (config path and fold index)
                    config_path, fold_index = task.split("|")
                    fold_index = int(fold_index)
                    
                    logging.info(f"Processing fold {fold_index} with configuration {config_path}")
                    process_k_fold(config_path, fold_index)  # Process specific fold
                    
                    client_socket.sendall(b"ACK")  # Send acknowledgment to coordinator
        except (socket.error, ConnectionRefusedError):
            logging.warning(f"Coordinator not available. Retrying in {RETRY_INTERVAL} seconds...")
            time.sleep(RETRY_INTERVAL)

def setup_logging():
    log_file = os.getenv('LOG_FILE_PATH', '/app/logs/app.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_json_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def process_k_fold(config_directory, fold_index):
    job_config_path = os.path.join(config_directory, 'job.json')
    system_config_path = os.path.join(config_directory, 'system.json')

    if not os.path.exists(job_config_path) or not os.path.exists(system_config_path):
        logging.warning("Configuration files missing in %s. Skipping...", config_directory)
        return

    config = load_json_config(job_config_path)
    syst_config = load_json_config(system_config_path)

    # Trainer setup for K-Fold with specific fold index
    trainer = Trainer(config, syst_config, k_folds=5, fold_index=fold_index)
    X_train, X_test, y_train, y_test, ensembles = trainer.train_k_fold()

    train_metrics, test_metrics, ensemble_names, train_df, test_df, ensemble_output_df = evaluate_and_save(config, X_train, X_test, y_train, y_test, ensembles)

    latency_df = Plotter.create_latency_df(ensemble_names, config['results_folder'])
    sample_count_df = Plotter.create_sample_count_df(X_train, X_test, ensemble_names)

    Plotter.plot_latency(latency_df, config['results_folder'])
    Plotter.plot_sample_count(sample_count_df, config['results_folder'])

    ResultSaver.save_results(train_df, test_df, ensemble_output_df, config['results_folder'])
    ResultSaver.save_confusion_matrices(ensemble_names, train_metrics, test_metrics, config['results_folder'])
    logging.info(f"Completed processing for fold {fold_index}")

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting worker process for K-Fold validation")
    connect_to_coordinator()
