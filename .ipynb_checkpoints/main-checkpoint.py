from src.data_loader import load_dataset
from src.feature_engineering import preprocess_network_traffic_data
from src.train_model import train_intrusion_detection_model
from src.predict import predict_network_anomaly
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """Main pipeline for intrusion detection model training and evaluation."""

    # Load network traffic datasets
    print("Loading network traffic datasets...")
    training_network_data = load_dataset(split="train")
    testing_network_data = load_dataset("test")

    print(f"Training dataset dimensions: {training_network_data.shape}")
    print(f"Testing dataset dimensions: {testing_network_data.shape}")

    # Display dataset information
    print("\nTraining Dataset Info:")
    print(training_network_data.info())
    print("\nTesting Dataset Info:")
    print(testing_network_data.info())

    # Preprocess network traffic features
    print("\nPreprocessing network traffic features...")
    (processed_training_features,
     training_labels,
     feature_scaler,
     categorical_encoders) = preprocess_network_traffic_data(
        training_network_data,
        is_training_phase=True
    )

    (processed_testing_features,
     testing_labels,
     _, _) = preprocess_network_traffic_data(
        testing_network_data,
        feature_scaler,
        categorical_encoders,
        is_training_phase=False
    )

    # Train intrusion detection model
    print("\nTraining intrusion detection model...")
    trained_classifier = train_intrusion_detection_model(
        processed_training_features,
        processed_testing_features,
        training_labels,
        testing_labels
    )

    # Test single prediction
    print("\nTesting single network connection prediction...")
    test_network_sample = testing_network_data.drop(
        columns=["class"]).iloc[0].to_dict()

    anomaly_prediction_result = predict_network_anomaly(
        test_network_sample, trained_classifier)
    print(f"Network anomaly prediction: {anomaly_prediction_result}")

    # Display model performance summary
    print("\nModel training and evaluation completed!")
    print(" Check the 'models/' directory for visualization plots:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - feature_importance.png")


if __name__ == "__main__":
    main()
