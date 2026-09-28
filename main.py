"""
AI/ML Flood Risk Classification - Simplified Proof of Concept

This script implements the simplified PoC used for the AI/ML Intern
technical assessment.

IMPORTANT:
- The available dataset is synthetic/tabular and is NOT the original
  Sharjah geospatial dataset from the Hydro-TransformerNet paper.
- A hydrologically motivated synthetic target called
  "Synthetic Flood Risk" is generated from the available features.
- The script compares:
    1. 1D CNN
    2. 1D CNN + Self-Attention
    3. Random Forest

Usage:
    python main.py --csv flood_risk_dataset_india.csv

Optional:
    python main.py --csv data/flood_risk_dataset_india.csv --epochs 30
"""

import os
import random
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

warnings.filterwarnings("ignore")


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SYNTHETIC TARGET
# ============================================================

def create_synthetic_target(df):
    """
    Create a balanced, hydrologically motivated synthetic target.

    Higher rainfall, river discharge, water level and historical
    floods increase risk. Lower elevation increases risk.

    This is a synthetic PoC target and must not be described as
    real historical flood labels.
    """

    result = df.copy()

    def minmax(series):
        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:
            return pd.Series(0.0, index=series.index)

        return (series - minimum) / (maximum - minimum)

    rainfall_score = minmax(result["Rainfall (mm)"])
    discharge_score = minmax(result["River Discharge (m³/s)"])
    water_level_score = minmax(result["Water Level (m)"])

    # Lower elevation -> higher flood susceptibility
    elevation_score = 1.0 - minmax(result["Elevation (m)"])

    historical_score = result["Historical Floods"].astype(float)

    # Hydrologically motivated weighted score
    risk_score = (
        0.30 * rainfall_score
        + 0.25 * discharge_score
        + 0.20 * water_level_score
        + 0.15 * elevation_score
        + 0.10 * historical_score
    )

    # Small noise prevents a perfectly deterministic classification rule.
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 0.08, len(result))

    final_score = risk_score + noise

    # Median threshold produces an approximately balanced target.
    threshold = np.median(final_score)

    result["Synthetic Flood Risk"] = (
        final_score >= threshold
    ).astype(int)

    return result


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_data(df):
    target_column = "Synthetic Flood Risk"

    numeric_features = [
        "Latitude",
        "Longitude",
        "Rainfall (mm)",
        "Temperature (°C)",
        "Humidity (%)",
        "River Discharge (m³/s)",
        "Water Level (m)",
        "Elevation (m)",
        "Population Density",
        "Infrastructure",
        "Historical Floods",
    ]

    categorical_features = [
        "Land Cover",
        "Soil Type",
    ]

    required_columns = (
        numeric_features
        + categorical_features
        + [target_column]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    X = df[
        numeric_features + categorical_features
    ].copy()

    y = df[target_column].copy()

    # Convert categorical variables to one-hot encoded columns.
    X = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=True
    )

    X = X.astype(float)

    X_train_df, X_test_df, y_train_series, y_test_series = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=SEED,
            stratify=y
        )
    )

    # Standardize numerical representation.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    # Convert to PyTorch tensors.
    X_train = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train_series.to_numpy(),
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test_series.to_numpy(),
        dtype=torch.float32
    )

    # 1D CNN input format:
    # [samples, channels, features]
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        X.columns.tolist()
    )


# ============================================================
# CNN MODEL
# ============================================================

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x.squeeze(1)


# ============================================================
# CNN + SELF-ATTENTION MODEL
# ============================================================

class CNNAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()

        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.norm = nn.LayerNorm(64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # [batch, channels, features]
        # -> [batch, features, channels]
        x = x.transpose(1, 2)

        # Self-attention
        attention_output, _ = self.attention(
            x, x, x
        )

        # Residual connection
        x = x + attention_output

        # Layer normalization
        x = self.norm(x)

        # Global average pooling over feature positions
        x = x.mean(dim=1)

        # Classification head
        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# TRAINING
# ============================================================

def train_neural_model(
    model,
    train_loader,
    epochs=30,
    learning_rate=0.001,
    weight_decay=0.0001,
    model_name="Model"
):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    losses = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = (
            running_loss / len(train_loader)
        )

        losses.append(epoch_loss)

        print(
            f"{model_name} | "
            f"Epoch [{epoch + 1:02d}/{epochs}] "
            f"Training Loss: {epoch_loss:.4f}"
        )

    return losses


# ============================================================
# NEURAL MODEL EVALUATION
# ============================================================

def evaluate_neural_model(
    model,
    test_loader
):
    model.eval()

    predictions = []
    probabilities = []
    targets = []

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            outputs = model(batch_X)

            probs = torch.sigmoid(outputs)

            preds = (
                probs >= 0.5
            ).float()

            probabilities.extend(
                probs.cpu().numpy()
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            targets.extend(
                batch_y.cpu().numpy()
            )

    metrics = calculate_metrics(
        targets,
        predictions,
        probabilities
    )

    return metrics


# ============================================================
# RANDOM FOREST EVALUATION
# ============================================================

def evaluate_random_forest(
    model,
    X_test,
    y_test
):
    X_test_np = X_test.squeeze(1).numpy()
    y_test_np = y_test.numpy()

    predictions = model.predict(
        X_test_np
    )

    probabilities = model.predict_proba(
        X_test_np
    )[:, 1]

    metrics = calculate_metrics(
        y_test_np,
        predictions,
        probabilities
    )

    return metrics


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    targets,
    predictions,
    probabilities
):
    return {
        "Accuracy": accuracy_score(
            targets,
            predictions
        ),
        "Precision": precision_score(
            targets,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            targets,
            predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            targets,
            predictions,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            targets,
            probabilities
        ),
        "Confusion Matrix": confusion_matrix(
            targets,
            predictions
        ),
        "Targets": np.asarray(targets),
        "Predictions": np.asarray(predictions),
        "Probabilities": np.asarray(probabilities),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main(csv_path, output_dir, epochs=30):

    set_seed()

    print("=" * 70)
    print("AI/ML FLOOD RISK CLASSIFICATION - SIMPLIFIED PoC")
    print("=" * 70)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    models_dir = os.path.join(
        output_dir,
        "models"
    )

    results_dir = os.path.join(
        output_dir,
        "results"
    )

    figures_dir = os.path.join(
        output_dir,
        "figures"
    )

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    print("\nDataset shape:", df.shape)

    # --------------------------------------------------------
    # Create synthetic target
    # --------------------------------------------------------

    df = create_synthetic_target(df)

    print("\nSynthetic target distribution:")
    print(
        df["Synthetic Flood Risk"].value_counts()
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = prepare_data(df)

    print("\nPreprocessed data:")
    print("Number of features:", len(feature_names))
    print("Training samples:", len(y_train))
    print("Test samples:", len(y_test))

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    test_dataset = TensorDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING CNN")
    print("=" * 70)

    cnn_model = CNNModel()

    cnn_losses = train_neural_model(
        cnn_model,
        train_loader,
        epochs=epochs,
        model_name="CNN"
    )

    cnn_metrics = evaluate_neural_model(
        cnn_model,
        test_loader
    )

    # --------------------------------------------------------
    # CNN + Self-Attention
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING CNN + SELF-ATTENTION")
    print("=" * 70)

    attention_model = CNNAttentionModel()

    attention_losses = train_neural_model(
        attention_model,
        train_loader,
        epochs=epochs,
        model_name="CNN + Attention"
    )

    attention_metrics = evaluate_neural_model(
        attention_model,
        test_loader
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    X_train_rf = X_train.squeeze(1).numpy()
    X_test_rf = X_test.squeeze(1).numpy()

    y_train_rf = y_train.numpy()
    y_test_rf = y_test.numpy()

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=SEED,
        n_jobs=-1
    )

    rf_model.fit(
        X_train_rf,
        y_train_rf
    )

    rf_metrics = evaluate_random_forest(
        rf_model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    results = pd.DataFrame({
        "Model": [
            "CNN",
            "CNN + Self-Attention",
            "Random Forest"
        ],
        "Accuracy": [
            cnn_metrics["Accuracy"],
            attention_metrics["Accuracy"],
            rf_metrics["Accuracy"]
        ],
        "Precision": [
            cnn_metrics["Precision"],
            attention_metrics["Precision"],
            rf_metrics["Precision"]
        ],
        "Recall": [
            cnn_metrics["Recall"],
            attention_metrics["Recall"],
            rf_metrics["Recall"]
        ],
        "F1 Score": [
            cnn_metrics["F1 Score"],
            attention_metrics["F1 Score"],
            rf_metrics["F1 Score"]
        ],
        "ROC-AUC": [
            cnn_metrics["ROC-AUC"],
            attention_metrics["ROC-AUC"],
            rf_metrics["ROC-AUC"]
        ],
    })

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    print("\nCNN confusion matrix:")
    print(cnn_metrics["Confusion Matrix"])

    print("\nCNN + Self-Attention confusion matrix:")
    print(attention_metrics["Confusion Matrix"])

    print("\nRandom Forest confusion matrix:")
    print(rf_metrics["Confusion Matrix"])

    # --------------------------------------------------------
    # Random Forest feature importance
    # --------------------------------------------------------

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        feature_importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    torch.save(
        cnn_model.state_dict(),
        os.path.join(
            models_dir,
            "cnn_model.pth"
        )
    )

    torch.save(
        attention_model.state_dict(),
        os.path.join(
            models_dir,
            "cnn_attention_model.pth"
        )
    )

    joblib.dump(
        rf_model,
        os.path.join(
            models_dir,
            "random_forest_model.pkl"
        )
    )

    joblib.dump(
        scaler,
        os.path.join(
            models_dir,
            "scaler.pkl"
        )
    )

    # --------------------------------------------------------
    # Save CSV results
    # --------------------------------------------------------

    results.to_csv(
        os.path.join(
            results_dir,
            "model_comparison_results.csv"
        ),
        index=False
    )

    feature_importance.to_csv(
        os.path.join(
            results_dir,
            "feature_importance_results.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Figure 1: Training loss
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(cnn_losses) + 1),
        cnn_losses,
        label="CNN"
    )

    plt.plot(
        range(1, len(attention_losses) + 1),
        attention_losses,
        label="CNN + Self-Attention"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figures_dir,
            "training_loss.png"
        ),
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Figure 2: ROC curves
    # --------------------------------------------------------

    cnn_fpr, cnn_tpr, _ = roc_curve(
        cnn_metrics["Targets"],
        cnn_metrics["Probabilities"]
    )

    attention_fpr, attention_tpr, _ = roc_curve(
        attention_metrics["Targets"],
        attention_metrics["Probabilities"]
    )

    rf_fpr, rf_tpr, _ = roc_curve(
        rf_metrics["Targets"],
        rf_metrics["Probabilities"]
    )

    plt.figure(figsize=(9, 7))

    plt.plot(
        cnn_fpr,
        cnn_tpr,
        label=f"CNN (AUC = {cnn_metrics['ROC-AUC']:.4f})"
    )

    plt.plot(
        attention_fpr,
        attention_tpr,
        label=(
            "CNN + Self-Attention "
            f"(AUC = {attention_metrics['ROC-AUC']:.4f})"
        )
    )

    plt.plot(
        rf_fpr,
        rf_tpr,
        label=f"Random Forest (AUC = {rf_metrics['ROC-AUC']:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Flood Risk Classification")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figures_dir,
            "roc_curves.png"
        ),
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Figure 3: Confusion matrices
    # --------------------------------------------------------

    matrices = [
        (
            cnn_metrics["Confusion Matrix"],
            "CNN"
        ),
        (
            attention_metrics["Confusion Matrix"],
            "CNN + Self-Attention"
        ),
        (
            rf_metrics["Confusion Matrix"],
            "Random Forest"
        )
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )

    for ax, (cm, title) in zip(
        axes,
        matrices
    ):

        im = ax.imshow(cm)

        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["No Flood", "Flood"]
        )

        ax.set_yticklabels(
            ["No Flood", "Flood"]
        )

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center"
                )

    plt.suptitle(
        "Confusion Matrices - Flood Risk Classification"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figures_dir,
            "confusion_matrices.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Figure 4: Feature importance
    # --------------------------------------------------------

    top_features = feature_importance.head(10)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figures_dir,
            "feature_importance.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nSaved models:")
    print(" - models/cnn_model.pth")
    print(" - models/cnn_attention_model.pth")
    print(" - models/random_forest_model.pkl")
    print(" - models/scaler.pkl")

    print("\nSaved results:")
    print(" - results/model_comparison_results.csv")
    print(" - results/feature_importance_results.csv")

    print("\nSaved figures:")
    print(" - figures/training_loss.png")
    print(" - figures/roc_curves.png")
    print(" - figures/confusion_matrices.png")
    print(" - figures/feature_importance.png")


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Simplified CNN + Self-Attention "
            "Flood Risk Classification PoC"
        )
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="flood_risk_dataset_india.csv/flood_risk_dataset_india.csv",
        help="Path to the input CSV dataset"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory for models, results and figures"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )

    args = parser.parse_args()

    main(
        csv_path=args.csv,
        output_dir=args.output,
        epochs=args.epochs
    )
