import logging
from metrics_evaluator import MetricsEvaluator
import pandas as pd

def evaluate_and_save(config, X_train, X_test, y_train, y_test, ensembles):
    """
    Evaluate performance metrics for ensembles and save results.

    Args:
        config (dict): Configuration parameters for evaluation.
        X_train (DataFrame): Features of the training set.
        X_test (DataFrame): Features of the testing set.
        y_train (Series): Target of the training set.
        y_test (Series): Target of the testing set.
        ensembles (list): List of ensemble models.

    Returns:
        train_metrics (list): Performance metrics for training set.
        test_metrics (list): Performance metrics for testing set.
        ensemble_names (list): Names of the ensembles.
        train_df (DataFrame): DataFrame containing training set metrics.
        test_df (DataFrame): DataFrame containing testing set metrics.
        ensemble_output_df (DataFrame): DataFrame containing ensemble output data.
    """
    results_folder = config['results_folder']
    logging.info("Starting evaluation and saving process. Results folder: %s", results_folder)

    # Evaluate performance metrics for training set
    logging.info("Evaluating performance metrics for the training set")
    train_metrics = MetricsEvaluator.evaluate_performance_metrics(
        ensembles, X_train, X_train, y_train, y_train, results_folder, 'train'
    )

    # Update ensemble names
    ensemble_names = ['_'.join([name for name, _ in ensemble.named_estimators.items()]) for ensemble in ensembles]
    logging.info("Ensemble names updated: %s", ensemble_names)

    # Convert train_metrics to DataFrame
    train_df = pd.DataFrame(
        train_metrics,
        columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix', 'Train Latency', 'Test Latency']
    )
    train_df.insert(0, 'Ensemble Name', ensemble_names)
    logging.info("Training metrics DataFrame created")

    # Evaluate performance metrics for test set
    logging.info("Evaluating performance metrics for the test set")
    test_metrics = MetricsEvaluator.evaluate_performance_metrics(
        ensembles, X_train, X_test, y_train, y_test, results_folder, 'test'
    )

    # Convert test_metrics to DataFrame
    test_df = pd.DataFrame(
        test_metrics,
        columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix', 'Train Latency', 'Test Latency']
    )
    test_df.insert(0, 'Ensemble Name', ensemble_names)
    logging.info("Test metrics DataFrame created")

    # Additional CSV output for ensemble predictions, probabilities, and class ranks
    logging.info("Generating additional output for ensemble predictions, probabilities, and class ranks")
    ensemble_predictions = [ensemble.predict(X_test) for ensemble in ensembles]
    ensemble_probs = [ensemble.predict_proba(X_test) for ensemble in ensembles]
    ensemble_class_ranks = [ensemble.predict_proba(X_test).argsort(axis=1) for ensemble in ensembles]

    ensemble_output_data = {}
    for i, (pred, prob, ranks) in enumerate(zip(ensemble_predictions, ensemble_probs, ensemble_class_ranks)):
        ensemble_output_data[f"{ensemble_names[i]} - Predicted Class"] = pred
        ensemble_output_data[f"{ensemble_names[i]} - Predicted Probabilities"] = [row for row in prob]
        ensemble_output_data[f"{ensemble_names[i]} - Class Rank"] = [row for row in ranks]
    ensemble_output_df = pd.DataFrame(ensemble_output_data)
    logging.info("Ensemble output DataFrame created")

    # Save metrics for each ensemble
    for i, ensemble_name in enumerate(ensemble_names):
        logging.info("Saving metrics for ensemble: %s", ensemble_name)
        MetricsEvaluator.save_metrics(train_metrics[i], ensemble_name, 'train', results_folder)
        MetricsEvaluator.save_metrics(test_metrics[i], ensemble_name, 'test', results_folder)

    logging.info("Evaluation and saving process completed successfully")
    return train_metrics, test_metrics, ensemble_names, train_df, test_df, ensemble_output_df
