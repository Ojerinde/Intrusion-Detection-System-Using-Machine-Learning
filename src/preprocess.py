import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_network_traffic_data(
    network_traffic_dataframe: pd.DataFrame,
    fitted_scaler=None,
    fitted_categorical_encoders=None,
    is_training_phase=True
):
    """
    Preprocess network traffic data with encoding and scaling.

    Parameters:
    - network_traffic_dataframe (pd.DataFrame): Input dataset.
    - fitted_scaler: Pre-fitted StandardScaler (used for testing).
    - fitted_categorical_encoders: Pre-fitted LabelEncoders (used for testing).
    - is_training_phase (bool): Whether in training or testing phase.

    Returns:
    - tuple: (scaled features, labels, scaler, encoders)
    """
    processed_dataframe = network_traffic_dataframe.copy()

    # Convert to binary classification
    processed_dataframe['class'] = processed_dataframe['class'].apply(
        lambda traffic_class: 'normal' if traffic_class == 'normal' else 'attack'
    )

    # Remove missing values
    processed_dataframe.dropna(inplace=True)

    # Separate features and labels
    network_features = processed_dataframe.drop('class', axis=1)
    intrusion_labels = processed_dataframe['class']

    if is_training_phase:
        # Encode categorical features
        categorical_feature_encoders = {}
        categorical_columns = network_features.select_dtypes(
            include='object').columns

        for categorical_column in categorical_columns:
            label_encoder = LabelEncoder()
            network_features[categorical_column] = label_encoder.fit_transform(
                network_features[categorical_column]
            )
            categorical_feature_encoders[categorical_column] = label_encoder

        # Scale features
        numerical_feature_scaler = StandardScaler()
        scaled_network_features = numerical_feature_scaler.fit_transform(
            network_features)

        return (scaled_network_features, intrusion_labels,
                numerical_feature_scaler, categorical_feature_encoders)
    else:
        # Apply fitted encoders and scaler
        for column_name, fitted_encoder in fitted_categorical_encoders.items():
            if column_name in network_features.columns:
                try:
                    network_features[column_name] = fitted_encoder.transform(
                        network_features[column_name]
                    )
                except ValueError:
                    known_classes = fitted_encoder.classes_
                    network_features[column_name] = network_features[column_name].apply(
                        lambda x: x if x in known_classes else known_classes[0]
                    )
                    network_features[column_name] = fitted_encoder.transform(
                        network_features[column_name]
                    )

        scaled_network_features = fitted_scaler.transform(network_features)
        return (scaled_network_features, intrusion_labels, fitted_scaler, fitted_categorical_encoders)
