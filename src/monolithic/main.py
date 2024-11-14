import json
import os
import sys
import logging
from trainer import Trainer
from evaluation import evaluate_and_save
from plotter import Plotter
from result_saver import ResultSaver

def setup_logging():
    # Get log file path from environment variable or default to 'app.log'
    log_file = os.getenv('LOG_FILE_PATH', 'app.log')
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Adding handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Initialize logging at the start
setup_logging()
logging.info("Starting main program")

def load_json_config(config_path):
    """
    Load a JSON configuration file.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: Parsed JSON configuration.
    """
    logging.info("Loading JSON configuration from %s", config_path)
    with open(config_path, 'r') as file:
        return json.load(file)

def process_directory(directory):
    """
    Process a directory containing the configuration files (system.json and job.json).
    Loads configurations, trains models, evaluates them, and saves the results.

    Args:
        directory (str): Path to the directory containing configuration files.
    """
    logging.info("Processing directory: %s", directory)

    # Define paths to the configuration files
    job_config_path = os.path.join(directory, 'job.json')
    system_config_path = os.path.join(directory, 'system.json')

    # Check if both configuration files exist
    if not os.path.exists(job_config_path) or not os.path.exists(system_config_path):
        logging.warning("Configuration files missing in %s. Skipping...", directory)
        return

    # Load configuration files
    config = load_json_config(job_config_path)
    syst_config = load_json_config(system_config_path)

    # Instantiate Trainer and train models
    trainer = Trainer(config, syst_config)
    X_train, X_test, y_train, y_test, ensembles = trainer.train()

    # Evaluate and save results
    train_metrics, test_metrics, ensemble_names, train_df, test_df, ensemble_output_df = evaluate_and_save(config, X_train, X_test, y_train, y_test, ensembles)

    # Create latency and sample count dataframes
    latency_df = Plotter.create_latency_df(ensemble_names, config['results_folder'])
    sample_count_df = Plotter.create_sample_count_df(X_train, X_test, ensemble_names)

    # Plot latency and sample count
    Plotter.plot_latency(latency_df, config['results_folder'])
    Plotter.plot_sample_count(sample_count_df, config['results_folder'])

    # Save results and confusion matrices
    ResultSaver.save_results(train_df, test_df, ensemble_output_df, config['results_folder'])
    ResultSaver.save_confusion_matrices(ensemble_names, train_metrics, test_metrics, config['results_folder'])
    logging.info("Completed processing for directory: %s", directory)

def main():
    """
    Main function to execute the model training, evaluation, and result saving process.
    Accepts a directory as a command-line argument, which should contain system.json and job.json,
    or subdirectories with these configuration files.
    """
    if len(sys.argv) < 2:
        logging.error("Usage: python main.py <config_directory>")
        sys.exit(1)

    config_directory = sys.argv[1]

    # Check if the provided argument is a directory
    if os.path.isdir(config_directory):
        # Check if the directory contains subdirectories
        subdirs = [os.path.join(config_directory, d) for d in os.listdir(config_directory) if os.path.isdir(os.path.join(config_directory, d))]
        if subdirs:
            # Process each subdirectory
            for subdir in subdirs:
                process_directory(subdir)
        else:
            # Process the directory directly if no subdirectories are found
            process_directory(config_directory)
    else:
        logging.error("%s is not a directory or does not exist.", config_directory)
        sys.exit(1)

if __name__ == "__main__":
    main()
