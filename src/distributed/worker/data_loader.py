import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    @staticmethod
    def create_column_mapping(feature_mapping, header_present):
        """
        Create a dictionary mapping feature names to column names or IDs.

        Args:
        - feature_mapping (DataFrame): DataFrame containing feature mapping information.
        - header_present (bool): Indicates whether the dataset has a header row.

        Returns:
        - dict: Dictionary mapping feature names to column names or IDs.
        """
        logging.info("Creating column mapping with header_present=%s", header_present)
        if header_present:
            return dict(zip(feature_mapping['feature_name'], feature_mapping['column_name']))
        else:
            return dict(zip(feature_mapping['feature_name'], feature_mapping['column_id']))

    @staticmethod
    def create_feature_type_mapping(feature_mapping):
        """
        Create a dictionary mapping feature names to column types.

        Args:
        - feature_mapping (DataFrame): DataFrame containing feature mapping information.

        Returns:
        - dict: Dictionary mapping feature names to column types.
        """
        logging.info("Creating feature type mapping")
        return dict(zip(feature_mapping['feature_name'], feature_mapping['column_type']))

    @staticmethod
    def preprocess_features(df, feature_mapping, feature_types, target_name):
        """
        Preprocess features based on their types.

        Args:
        - df (DataFrame): DataFrame containing the dataset.
        - feature_mapping (dict): Dictionary mapping feature names to column names or IDs.
        - feature_types (dict): Dictionary mapping feature names to column types.
        - target_name (str): Name of the target column.

        Returns:
        - DataFrame: Preprocessed DataFrame with features.
        """
        logging.info("Preprocessing features")
        for feature, column_name in feature_mapping.items():
            if feature == target_name:
                continue  # Skip processing the target column
            column_type = feature_types[feature]
            logging.debug("Processing feature: %s, type: %s", feature, column_type)
            if column_type == 'numeric':
                if df[column_name].dtype in ['float64', 'int64']:
                    df[column_name] = df[column_name].fillna(df[column_name].mean())
                else:
                    logging.warning("Column '%s' is not numeric. Skipping mean imputation.", column_name)
            elif column_type == 'categorical':
                df[column_name] = df[column_name].fillna(df[column_name].mode()[0])
                df = pd.get_dummies(df, columns=[column_name])
                logging.debug("Processed categorical feature: %s", column_name)
            elif column_type == 'ordinal':
                df[column_name] = df[column_name].fillna(df[column_name].mode()[0])
                label_encoder = LabelEncoder()
                df[column_name] = label_encoder.fit_transform(df[column_name])
                logging.debug("Processed ordinal feature: %s", column_name)
        return df

    @staticmethod
    def preprocess_target(df, target_name, feature_mapping, feature_types):
        """
        Preprocess the target column based on its type.

        Args:
        - df (DataFrame): DataFrame containing the dataset.
        - target_name (str): Name of the target column.
        - feature_mapping (dict): Dictionary mapping feature names to column names or IDs.
        - feature_types (dict): Dictionary mapping feature names to column types.

        Returns:
        - DataFrame: Preprocessed DataFrame with target column.
        """
        logging.info("Preprocessing target column: %s", target_name)
        target_column = feature_mapping[target_name]
        target_column_type = feature_types[target_name]
        if target_column_type == 'numeric':
            label_encoder = LabelEncoder()
            df[target_column] = label_encoder.fit_transform(df[target_column])
        elif target_column_type == 'categorical':
            logging.debug("Target column '%s' is categorical", target_column)
        return df

    @staticmethod
    def split_dataset(X, y, split_ratio=0.2, random_state=None):
        """
        Split the dataset into training and test sets.

        Args:
        - X (DataFrame): DataFrame containing the features.
        - y (DataFrame): DataFrame containing the target variable.
        - split_ratio (float): Ratio of test set size to total dataset size.
        - random_state (int): Random seed for reproducibility.

        Returns:
        - tuple: X_train, X_test, y_train, y_test
        """
        logging.info("Splitting dataset with test size ratio of %s", split_ratio)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=random_state)
        logging.info("Dataset split completed")
        return X_train, X_test, y_train, y_test

    @staticmethod
    def load_dataset(dataset_path, feature_names, target_name, delimiter, use_feature_range, feature_range, header_present, feature_mapping_path, large_dataset_threshold=100000, sampling_ratio=1.0):
        """
        Load and preprocess the dataset, with optional chunk loading for large datasets.

        Args:
        - dataset_path (str): Path to the dataset file.
        - feature_names (list): List of feature names to include in the dataset.
        - target_name (str): Name of the target column.
        - delimiter (str): Delimiter used in the dataset file.
        - use_feature_range (bool): Whether to use a specified range of columns as features.
        - feature_range (list): Range of columns to use as features.
        - header_present (bool): Indicates whether the dataset has a header row.
        - feature_mapping_path (str): Path to the feature mapping file.
        - large_dataset_threshold (int): Row threshold above which the dataset is considered large.
        - sampling_ratio (float): Ratio for sampling rows if dataset is too large.

        Returns:
        - tuple: X (features), y (target)
        """
        logging.info("Loading dataset from %s", dataset_path)
        feature_mapping = pd.read_csv(feature_mapping_path)
        column_mapping = DataLoader.create_column_mapping(feature_mapping, header_present)
        feature_types = DataLoader.create_feature_type_mapping(feature_mapping)

        # Check the size of the dataset by loading only the first few rows
        initial_rows = pd.read_csv(dataset_path, delimiter=delimiter, header=0 if header_present else None, nrows=10)
        total_rows = initial_rows.shape[0]

        # Determine if chunk loading is necessary
        if total_rows > large_dataset_threshold:
            logging.info("Dataset exceeds large dataset threshold; loading in chunks with sampling ratio: %s", sampling_ratio)
            chunks = pd.read_csv(dataset_path, delimiter=delimiter, header=0 if header_present else None, chunksize=10000)
            sampled_data = [chunk.sample(frac=sampling_ratio) for chunk in chunks]
            df = pd.concat(sampled_data, ignore_index=True)
        else:
            df = pd.read_csv(dataset_path, delimiter=delimiter, header=0 if header_present else None)
        
        logging.info("Dataset loaded successfully with shape %s", df.shape)

        # Select feature columns and target column
        if use_feature_range:
            start_index = int(column_mapping[feature_range[0]]) if not header_present else df.columns.get_loc(feature_range[0])
            end_index = int(column_mapping[feature_range[1]]) if not header_present else df.columns.get_loc(feature_range[1])
            feature_columns = list(range(start_index, end_index + 1)) if not header_present else df.columns[start_index:end_index + 1].tolist()
        else:
            feature_columns = [column_mapping[feature] for feature in feature_names]
        target_column = column_mapping[target_name] if not header_present else target_name

        # Ensure target column is present
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

        # Preprocess features and target
        df_features = DataLoader.preprocess_features(df[feature_columns], column_mapping, feature_types, target_name)
        df_target = DataLoader.preprocess_target(df[[target_column]], target_name, column_mapping, feature_types)

        logging.info("Preprocessing complete")
        return df_features, df_target
