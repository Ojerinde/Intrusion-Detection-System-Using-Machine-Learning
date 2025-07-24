from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from joblib import dump
import os


def train_intrusion_detection_model(
    training_feature_matrix,
    testing_feature_matrix,
    training_intrusion_labels,
    testing_intrusion_labels
):
    """
    Train Random Forest model and compute metrics.

    Parameters:
    - training_feature_matrix: Scaled training features.
    - testing_feature_matrix: Scaled testing features.
    - training_intrusion_labels: Training labels.
    - testing_intrusion_labels: Testing labels.

    Returns:
    - tuple: (trained model, metrics dictionary)
    """
    # Train the model
    intrusion_detection_classifier = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )

    intrusion_detection_classifier.fit(
        training_feature_matrix, training_intrusion_labels)
    predicted_intrusion_labels = intrusion_detection_classifier.predict(
        testing_feature_matrix)
    prediction_probabilities = intrusion_detection_classifier.predict_proba(
        testing_feature_matrix)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(testing_intrusion_labels, predicted_intrusion_labels),
        'precision': precision_score(testing_intrusion_labels, predicted_intrusion_labels, pos_label='attack'),
        'recall': recall_score(testing_intrusion_labels, predicted_intrusion_labels, pos_label='attack'),
        'f1_score': f1_score(testing_intrusion_labels, predicted_intrusion_labels, pos_label='attack'),
        'probabilities': prediction_probabilities,
        'predicted_labels': predicted_intrusion_labels
    }

    # Save model
    os.makedirs("models", exist_ok=True)
    dump(intrusion_detection_classifier, "models/intrusion_detection_model.pkl")

    return intrusion_detection_classifier, metrics
