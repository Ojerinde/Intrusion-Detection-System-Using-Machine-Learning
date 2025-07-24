from src.data_loader import load_dataset
from src.preprocess import preprocess_network_traffic_data
from src.model import train_intrusion_detection_model
from src.visualize import (
    plot_data_quality, plot_performance_metrics, plot_confusion_matrix,
    plot_roc_curve, plot_feature_importance, plot_prediction_confidence
)


def main_intrusion_detection_pipeline():
    print("Intrusion Detection ML Pipeline")
    print("=" * 60)

    # Load datasets
    print("Loading network traffic datasets...")
    training_network_data = load_dataset(split="train")
    testing_network_data = load_dataset("test")

    print(f"Training dataset dimensions: {training_network_data.shape}")
    print(f"Testing dataset dimensions: {testing_network_data.shape}")

    # Preprocess data
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

    # Visualize data quality
    print("\n DATA QUALITY ANALYSIS")
    print("-" * 40)
    plot_data_quality(training_network_data)

    # Train model and get metrics
    print("\n TRAINING INTRUSION DETECTION MODEL")
    print("=" * 50)
    trained_classifier, metrics = train_intrusion_detection_model(
        processed_training_features,
        processed_testing_features,
        training_labels,
        testing_labels
    )

    # Visualize model performance
    plot_performance_metrics(metrics)
    plot_confusion_matrix(testing_labels, metrics['predicted_labels'])
    roc_auc = plot_roc_curve(testing_labels, metrics['probabilities'])
    plot_feature_importance(trained_classifier)

    # Print summary
    print("\n MODEL TRAINING SUMMARY")
    print("=" * 40)
    print(f" Accuracy:  {metrics['accuracy']:.3f}")
    print(f" Precision: {metrics['precision']:.3f}")
    print(f" Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1_score']:.3f}")
    print(f" AUC-ROC:   {roc_auc:.3f}")
    print(f" Model saved to: models/intrusion_detection_model.pkl")

    # Test single prediction
    print("\nTesting single network connection prediction...")
    test_network_sample = testing_network_data.drop(
        columns=["class"]).iloc[0].to_dict()
    anomaly_prediction_result = plot_prediction_confidence(
        test_network_sample, trained_classifier)
    print(f"Network anomaly prediction: {anomaly_prediction_result}")

    print("\nModel training and evaluation completed!")


if __name__ == "__main__":
    main_intrusion_detection_pipeline()
