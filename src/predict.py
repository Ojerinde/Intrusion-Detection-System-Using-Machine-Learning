import pandas as pd
import os
from joblib import load
import numpy as np


def load_trained_intrusion_model(model_file_path="models/intrusion_detection_model.pkl"):
    """
    Load the trained intrusion detection model from disk.

    Args:
        model_file_path: Path to the saved model file

    Returns:
        Loaded trained model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(
            f"Trained intrusion detection model not found at: {model_file_path}"
        )

    print(f"Loading trained model from: {model_file_path}")
    return load(model_file_path)


def predict_network_anomaly(network_connection_sample: dict, trained_classifier):
    """
    Predict whether a single network connection is normal or an attack.

    Args:
        network_connection_sample: Dictionary containing network connection features
        trained_classifier: Trained machine learning model

    Returns:
        str: Predicted classification ("normal" or "attack")
        dict: Prediction details including confidence scores
    """

    # Convert sample to DataFrame for consistent processing
    connection_dataframe = pd.DataFrame([network_connection_sample])

    # Handle categorical features by converting to category codes
    categorical_columns = connection_dataframe.select_dtypes(
        include='object').columns
    for categorical_column in categorical_columns:
        connection_dataframe[categorical_column] = (
            connection_dataframe[categorical_column]
            .astype("category")
            .cat.codes
        )

    # Fill any missing values with zeros
    connection_dataframe = connection_dataframe.fillna(0)

    # Generate prediction
    predicted_class = trained_classifier.predict(connection_dataframe)[0]

    # Get prediction probabilities for confidence scoring
    try:
        prediction_probabilities = trained_classifier.predict_proba(
            connection_dataframe)[0]
        class_labels = trained_classifier.classes_

        # Create confidence dictionary
        confidence_scores = {
            class_label: float(probability)
            for class_label, probability in zip(class_labels, prediction_probabilities)
        }

        prediction_details = {
            "predicted_class": predicted_class,
            "confidence_scores": confidence_scores,
            "high_confidence": max(prediction_probabilities) > 0.8
        }

    except AttributeError:
        # Fallback if model doesn't support probability prediction
        prediction_details = {
            "predicted_class": predicted_class,
            "confidence_scores": None,
            "high_confidence": None
        }

    return prediction_details


def predict_multiple_connections(
    network_connections_list: list,
    trained_classifier,
    batch_size=1000
):
    """
    Predict classifications for multiple network connections efficiently.

    Args:
        network_connections_list: List of dictionaries, each containing connection features
        trained_classifier: Trained machine learning model
        batch_size: Number of connections to process at once

    Returns:
        list: List of prediction dictionaries for each connection
    """

    total_connections = len(network_connections_list)
    all_predictions = []

    print(f" Processing {total_connections} network connections...")

    for batch_start in range(0, total_connections, batch_size):
        batch_end = min(batch_start + batch_size, total_connections)
        connection_batch = network_connections_list[batch_start:batch_end]

        # Process batch
        batch_dataframe = pd.DataFrame(connection_batch)

        # Handle categorical features
        categorical_columns = batch_dataframe.select_dtypes(
            include='object').columns
        for categorical_column in categorical_columns:
            batch_dataframe[categorical_column] = (
                batch_dataframe[categorical_column]
                .astype("category")
                .cat.codes
            )

        # Fill missing values
        batch_dataframe = batch_dataframe.fillna(0)

        # Generate predictions for batch
        batch_predictions = trained_classifier.predict(batch_dataframe)

        try:
            batch_probabilities = trained_classifier.predict_proba(
                batch_dataframe)
            class_labels = trained_classifier.classes_

            # Create detailed predictions for batch
            for i, (prediction, probabilities) in enumerate(zip(batch_predictions, batch_probabilities)):
                confidence_scores = {
                    class_label: float(probability)
                    for class_label, probability in zip(class_labels, probabilities)
                }

                prediction_details = {
                    "connection_id": batch_start + i,
                    "predicted_class": prediction,
                    "confidence_scores": confidence_scores,
                    "high_confidence": max(probabilities) > 0.8
                }

                all_predictions.append(prediction_details)

        except AttributeError:
            # Fallback for models without probability prediction
            for i, prediction in enumerate(batch_predictions):
                prediction_details = {
                    "connection_id": batch_start + i,
                    "predicted_class": prediction,
                    "confidence_scores": None,
                    "high_confidence": None
                }
                all_predictions.append(prediction_details)

        # Progress indicator
        progress_percentage = (batch_end / total_connections) * 100
        print(
            f"Progress: {progress_percentage:.1f}% ({batch_end}/{total_connections})")

    return all_predictions
