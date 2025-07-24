import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# Set up plotting styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_data_quality(network_traffic_dataframe):
    """
    Visualize class distribution and missing values.
    """
    class_counts = network_traffic_dataframe['class'].value_counts()

    # Create class distribution plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Pie chart
    colors = ['#FF6B6B', '#4ECDC4']
    ax1.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, explode=(0.05, 0))
    ax1.set_title('Traffic Classification Distribution',
                  fontsize=14, fontweight='bold')

    # Bar chart
    bars = ax2.bar(class_counts.index, class_counts.values,
                   color=colors, alpha=0.8)
    ax2.set_title('Traffic Counts by Class', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Connections')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Missing values heatmap
    if network_traffic_dataframe.isnull().sum().sum() > 0:
        plt.figure(figsize=(12, 8))
        sns.heatmap(network_traffic_dataframe.isnull(), cbar=True, yticklabels=False,
                    cmap='viridis', alpha=0.8)
        plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
        plt.show()
    else:
        print("No missing values found!")


def plot_performance_metrics(metrics):
    """
    Visualize model performance metrics.
    """
    fig = go.Figure()

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [metrics['accuracy'], metrics['precision'],
              metrics['recall'], metrics['f1_score']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    fig.add_trace(go.Bar(
        x=metric_names,
        y=values,
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='auto',
        name='Performance Metrics'
    ))

    fig.update_layout(
        title='Model Performance Dashboard',
        title_font_size=20,
        xaxis_title='Metrics',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.show()


def plot_confusion_matrix(testing_intrusion_labels, predicted_labels):
    """
    Visualize confusion matrix as a heatmap.
    """
    cm = confusion_matrix(testing_intrusion_labels, predicted_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Attack', 'Normal'],
                yticklabels=['Attack', 'Normal'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)

    # Add performance annotations
    tn, fp, fn, tp = cm.ravel()
    plt.text(1.5, -0.1, f'True Positives: {tp:,}', transform=plt.gca().transAxes,
             fontsize=12, color='green', fontweight='bold')
    plt.text(1.5, -0.15, f'False Positives: {fp:,}', transform=plt.gca().transAxes,
             fontsize=12, color='red', fontweight='bold')
    plt.text(1.5, -0.2, f'False Negatives: {fn:,}', transform=plt.gca().transAxes,
             fontsize=12, color='orange', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_roc_curve(testing_intrusion_labels, prediction_probabilities):
    """
    Visualize ROC curve.
    """
    fpr, tpr, _ = roc_curve(
        testing_intrusion_labels.map({'attack': 1, 'normal': 0}),
        prediction_probabilities[:, 1]
    )
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Intrusion Detection Performance',
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return roc_auc


def plot_feature_importance(classifier, feature_count=20):
    """
    Visualize top feature importances.
    """
    feature_importance_scores = classifier.feature_importances_
    feature_indices = np.argsort(feature_importance_scores)[
        ::-1][:feature_count]

    plt.figure(figsize=(12, 10))
    top_scores = feature_importance_scores[feature_indices]

    bars = plt.barh(range(len(top_scores)), top_scores[::-1],
                    color=plt.cm.viridis(np.linspace(0, 1, len(top_scores))))

    plt.yticks(range(len(top_scores)),
               [f'Feature_{idx}' for idx in feature_indices[::-1]])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(' Top 20 Most Important Features',
              fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{width:.4f}', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_prediction_confidence(network_connection_sample, trained_classifier):
    """
    Visualize prediction confidence for a single sample.
    """
    connection_dataframe = pd.DataFrame([network_connection_sample])

    # Handle categorical features
    categorical_columns = connection_dataframe.select_dtypes(
        include='object').columns
    for categorical_column in categorical_columns:
        connection_dataframe[categorical_column] = (
            connection_dataframe[categorical_column].astype(
                "category").cat.codes
        )

    connection_dataframe = connection_dataframe.fillna(0)

    # Make prediction
    predicted_class = trained_classifier.predict(connection_dataframe)[0]
    prediction_probabilities = trained_classifier.predict_proba(
        connection_dataframe)[0]

    # Visualize prediction confidence
    classes = trained_classifier.classes_

    fig = go.Figure(data=[
        go.Bar(x=classes, y=prediction_probabilities,
               marker_color=['red' if c == 'attack' else 'green' for c in classes])
    ])

    fig.update_layout(
        title=f' Prediction: {predicted_class.upper()} (Confidence: {max(prediction_probabilities):.1%})',
        title_font_size=16,
        xaxis_title='Classification',
        yaxis_title='Probability',
        height=400
    )
    fig.show()

    return predicted_class
