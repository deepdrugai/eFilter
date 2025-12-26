import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef, average_precision_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import xgboost as xgb
import pickle
import joblib


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FPS_PATH = DATA_DIR / "drugbankfps-str-1024.p"
PROPS_PATH = DATA_DIR / "drugbank_properties.csv"


def preprocess_data():
    """Load and preprocess the data."""
    # Load embeddings
    with FPS_PATH.open("rb") as f:
        embeddings = pickle.load(f)

    # Convert the 'fpstr' column from a string of bits to an actual list of integers
    embeddings["fpstr"] = embeddings["fpstr"].apply(lambda x: np.array(list(map(int, list(x)))))

    # Extract the numpy array of embeddings
    X = np.stack(embeddings["fpstr"].values)

    # Load properties
    properties = pd.read_csv(PROPS_PATH)
    print("Calculated rows:", (properties["Property Category"] == "Calculated").sum())
    print("Experimental rows:", (properties["Property Category"] == "Experimental").sum())

    # Clean and prepare the properties data
    v = properties["Value"].astype(str)

    # strip < and >, trim whitespace
    v = v.str.replace(r"[<>]", "", regex=True).str.strip()

    # convert to numeric (handles ints, floats, negatives, sci notation); non-numeric -> NaN
    properties["Value"] = pd.to_numeric(v, errors="coerce")

    # keep only rows with a numeric value
    properties = properties.dropna(subset=["Value"])

    # Remove non-numeric 'Value' entries and strip '<' and '>' from numerical values
    # properties["Value"] = properties["Value"].str.replace("<", "").str.replace(">", "")
    # properties = properties[properties["Value"].str.isnumeric()]
    # properties["Value"] = pd.to_numeric(properties["Value"], errors="coerce")  # Coerce errors to NaN
    # properties.dropna(subset=["Value"], inplace=True)  # Drop rows where conversion failed  # type: ignore

    # Pivot the properties DataFrame to have one row per drug and one column per property type
    properties_pivot = properties.pivot_table(index="Drug ID", columns="Property Type", values="Value", aggfunc="mean", observed=False)

    # Join the embeddings DataFrame with the pivoted properties DataFrame
    data = embeddings.set_index("id").join(properties_pivot, how="inner")

    # Calculate non-NaN and NaN counts for each column
    nan_counts = data.isna().sum()
    non_nan_counts = data.notna().sum()

    # Create a DataFrame to store the counts
    counts_df = pd.DataFrame({"Non-NaN Count": non_nan_counts, "NaN Count": nan_counts})

    # Calculate percentage of non-NaN values
    total_rows = len(data)
    counts_df["Percentage Non-NaN"] = (counts_df["Non-NaN Count"] / total_rows) * 100

    # Sort by Percentage Non-NaN in descending order
    counts_df = counts_df.sort_values(by="Percentage Non-NaN", ascending=False)
    print("\n\nColumn Non-NaN Counts:\n", counts_df)

    # Extract features (embeddings) and labels (properties)
    X = np.array(data["fpstr"].tolist())  # Converting list of arrays into a 2D array

    # Always-drop columns
    always_drop = ["fpstr", "IUPAC Name", "Traditional IUPAC Name"]

    # Drop columns with < 80% non-NaN coverage
    low_coverage_cols = (counts_df.loc[counts_df["Percentage Non-NaN"] < 80].index.astype(str).tolist())
    dropped_cols = list(set(always_drop + low_coverage_cols))

    dropped_df = counts_df.loc[counts_df["Percentage Non-NaN"] < 80]
    print(f"\n\nDropping {len(dropped_df)} columns with <80% non-NaN coverage:")
    print(dropped_df)

    # Targets
    y_df = data.drop(columns=dropped_cols)
    y_df = y_df.apply(pd.to_numeric, errors="coerce")
    y = y_df.values  # Assuming all other columns are now properties
    y_cols = y_df.columns.tolist()

    # # Targets
    # y = data.drop(columns=dropped_cols).values  # Assuming all other columns are now properties
    # y_cols = data.drop(columns=dropped_cols).columns.tolist()

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure correct shapes and types
    print("\n\nData shapes after preprocessing:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Sanity Check
    # Initialize a dictionary to hold results of checks
    sanity_results = {}

    # Total number of columns and rows for percentage calculations
    total_columns = properties_pivot.shape[1]
    total_rows = properties_pivot.shape[0]

    # Check for columns with all NaN values
    nan_column_count = properties_pivot.isnull().all().sum()
    sanity_results["Columns with all NaN values"] = (nan_column_count, total_columns, nan_column_count / total_columns * 100)

    # Check for columns with all zero values
    zero_column_count = (properties_pivot == 0).all().sum()
    sanity_results["Columns with all zero values"] = (zero_column_count, total_columns, zero_column_count / total_columns * 100)

    # Check for columns with all the same value
    constant_column_count = sum(properties_pivot.nunique() == 1)
    sanity_results["Columns with all the same value"] = (constant_column_count, total_columns, constant_column_count / total_columns * 100)

    # Check for any duplicate rows
    duplicate_rows_count = properties_pivot.duplicated().sum()
    sanity_results["Duplicate rows"] = (duplicate_rows_count, total_rows, duplicate_rows_count / total_rows * 100 if total_rows > 0 else 0)

    # Display all sanity check results
    for check, (count, total, percentage) in sanity_results.items():
        print(f"{check}: {count}/{total} ({percentage:.2f}%)")

    # Check for NaNs and infinities in your data
    print(np.isnan(X_train).sum(), "NaNs in X_train")
    print(np.isinf(X_train).sum(), "Infs in X_train")
    print(np.isnan(y_train).sum(), "NaNs in y_train")
    print(np.isinf(y_train).sum(), "Infs in y_train")

    print("\n\n")

    print("\nSanity checks (properties_pivot):")
    n_rows, n_cols = properties_pivot.shape
    print(f"  Rows: {n_rows:,} | Columns: {n_cols:,}")

    checks = {
        "All-NaN columns": properties_pivot.isna().all().sum(),
        "All-zero columns": (properties_pivot == 0).all().sum(),
        "Constant columns": (properties_pivot.nunique(dropna=False) == 1).sum(),
        "Duplicate rows": properties_pivot.duplicated().sum(),
    }

    for name, count in checks.items():
        denom = n_cols if "columns" in name else n_rows
        pct = 0.0 if denom == 0 else 100 * count / denom
        print(f"  {name}: {count:,}/{denom:,} ({pct:.2f}%)")


    print("\nSanity checks (train arrays):")
    for label, arr in [("X_train", X_train), ("y_train", y_train)]:
        total = arr.size
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        nan_pct = 0.0 if total == 0 else 100 * n_nan / total
        inf_pct = 0.0 if total == 0 else 100 * n_inf / total
        print(
            f"  {label}: "
            f"NaNs {n_nan:,}/{total:,} ({nan_pct:.2f}%) | "
            f"Infs {n_inf:,}/{total:,} ({inf_pct:.2f}%)"
        )

    print("\nSanity checks (train arrays):")
    for label, arr in [("X_train", X_train), ("y_train", y_train)]:
        total = arr.size
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()

        print(
            f"{label:<8} | "
            f"  NaNs {n_nan:>6,}/{total:<10,} ({n_nan/total:.2%}) | "
            f"  Infs {n_inf:>6,}/{total:<10,} ({n_inf/total:.2%})"
        )

    return X_train, X_test, y_train, y_test, y_cols


def filter_valid_samples(X, y):
    """Filter out samples with NaNs in X or y."""
    valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X[valid_indices], y[valid_indices]


def standardize_features(X_train, X_test):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def build_nn_model(input_shape, is_binary):
    """Build a neural network model based on the task type."""
    input_layer = Input(shape=(input_shape,))
    x = Dense(512, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001))(input_layer)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001))(x)
    if is_binary:
        output_activation = "sigmoid"
        loss_function = "binary_crossentropy"
    else:
        output_activation = "linear"
        loss_function = "mse"
    output_layer = Dense(1, activation=output_activation, kernel_initializer="he_normal")(x)
    nn_model = Model(inputs=input_layer, outputs=output_layer)
    nn_model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0), loss=loss_function)
    return nn_model


def train_nn_model(nn_model, X_train, y_train, is_binary, target_name, class_weights, output_dir: Path):
    """Train the neural network model."""
    # Define callbacks
    checkpoint = ModelCheckpoint(
        output_dir / f"best_model_{target_name}.keras",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    # Fit the model
    if is_binary:
        history = nn_model.fit(
            X_train,
            y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            class_weight=class_weights,
            verbose=1,
        )
    else:
        history = nn_model.fit(
            X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[checkpoint, early_stopping, reduce_lr], verbose=1
        )
    return history


def train_xgb_model(X_train_features, y_train, is_binary, scale_pos_weight, output_dir: Path, target_name):
    """Train XGBoost model."""
    if is_binary:
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            colsample_bytree=0.7,
            learning_rate=0.01,
            max_depth=7,
            n_estimators=500,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            colsample_bytree=0.7,
            learning_rate=0.01,
            max_depth=7,
            n_estimators=500,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
        )
    xgb_model.fit(X_train_features, y_train)
    # Save the XGBoost model
    # xgb_model.save_model(output_dir / f"xgb_model_{target_name}.json") # This broke in recent xgboost versions
    xgb_model.get_booster().save_model(output_dir / f"xgb_model_{target_name}.json")

    # 2) sklearn wrapper (sometimes brittle across versions)
    joblib.dump(xgb_model, output_dir / f"xgb_sklearn_{target_name}.joblib")

    return xgb_model


def plot_learning_curve(history, target_name, output_dir: Path):
    """Plot and save the learning curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Learning Curves for {target_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"learning_curve_{target_name}.png")


def calculate_classification_metrics(y_true, y_proba):
    """Calculate classification metrics for a set of predictions."""
    metrics = {}
    metrics["auc"] = roc_auc_score(y_true, y_proba)
    metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr
    metrics["precision_curve"] = precision
    metrics["recall_curve"] = recall
    metrics["pr_thresholds"] = pr_thresholds
    opt_idx = np.argmax(tpr - fpr)
    metrics["threshold"] = thresholds[opt_idx]
    y_pred = (y_proba >= metrics["threshold"]).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["f1"] = (
        2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
        if (metrics["precision"] + metrics["recall"]) > 0
        else 0
    )
    return metrics

def plot_prediction_histogram(y_true, y_proba, threshold, model_name, target_name, output_dir: Path):
    """Plot histogram of prediction probabilities."""
    bins = np.linspace(0, 1, 50).tolist()
    plt.figure(figsize=(9, 6))
    plt.hist(y_proba[y_true == 1], bins, color="green", label="Positive Samples", alpha=0.5, density=True)
    plt.hist(y_proba[y_true == 0], bins, color="red", label="Negative Samples", alpha=0.5, density=True)
    plt.axvline(threshold, color="blue", linestyle="--", label=f"Optimal Threshold = {threshold:.2f}")
    plt.legend(loc="upper right")
    plt.xlabel("Prediction Probability")
    plt.ylabel("Density")
    plt.title(f"Prediction Distribution Histogram for {target_name} - {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"histogram_{target_name}_{model_name}.png")
    # plt.show()


def plot_roc_curves(roc_data_list, target_name, output_dir: Path):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(9, 6))
    plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
    for data in roc_data_list:
        plt.plot(data["fpr"], data["tpr"], lw=2, label=f"{data['model_name']} AUC = {data['auc']:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {target_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_dir / f"roc_curve_{target_name}.png")
    # plt.show()

def plot_pr_curves(pr_data_list, target_name, output_dir: Path):
    """Plot PR curves for multiple models."""
    plt.figure(figsize=(9, 6))
    for data in pr_data_list:
        plt.plot(
            data["recall"],
            data["precision"],
            lw=2,
            label=f"{data['model_name']} PR-AUC = {data['pr_auc']:.3f}",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve for {target_name}")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_dir / f"pr_curve_{target_name}.png")


def evaluate_regression_models(y_test_col, nn_predictions, xgb_predictions, combined_predictions, target_name, output_dir: Path):
    """Calculate error metrics and plot results for regression."""
    # Compute error metrics
    nn_mse = mean_squared_error(y_test_col, nn_predictions)
    nn_rmse = np.sqrt(nn_mse)
    nn_mae = mean_absolute_error(y_test_col, nn_predictions)

    xgb_mse = mean_squared_error(y_test_col, xgb_predictions)
    xgb_rmse = np.sqrt(xgb_mse)
    xgb_mae = mean_absolute_error(y_test_col, xgb_predictions)

    combined_mse = mean_squared_error(y_test_col, combined_predictions)
    combined_rmse = np.sqrt(combined_mse)
    combined_mae = mean_absolute_error(y_test_col, combined_predictions)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_col, y_test_col, alpha=0.5, label="Actual Values", color="green")
    plt.scatter(y_test_col, nn_predictions, alpha=0.5, label=f"NN Predictions (RMSE: {nn_rmse:.2f})", color="blue")
    plt.scatter(y_test_col, xgb_predictions, alpha=0.5, label=f"XGBoost Predictions (RMSE: {xgb_rmse:.2f})", color="red")
    plt.scatter(y_test_col, combined_predictions, alpha=0.5, label=f"Combined Predictions (RMSE: {combined_rmse:.2f})", color="purple")
    plt.plot([y_test_col.min(), y_test_col.max()], [y_test_col.min(), y_test_col.max()], "k--", label="Perfect Prediction")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted Values for {target_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"scatter_plot_{target_name}.png")
    # plt.show()

    # Residual Plot
    plt.figure(figsize=(8, 6))
    nn_residuals = y_test_col - nn_predictions
    xgb_residuals = y_test_col - xgb_predictions
    combined_residuals = y_test_col - combined_predictions
    plt.scatter(y_test_col, nn_residuals, alpha=0.5, label="NN Residuals", color="blue")
    plt.scatter(y_test_col, xgb_residuals, alpha=0.5, label="XGBoost Residuals", color="red")
    plt.scatter(y_test_col, combined_residuals, alpha=0.5, label="Combined Residuals", color="purple")
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot for {target_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"residual_plot_{target_name}.png")
    # plt.show()

    return {
        "nn": {"mse": nn_mse, "rmse": nn_rmse, "mae": nn_mae},
        "xgb": {"mse": xgb_mse, "rmse": xgb_rmse, "mae": xgb_mae},
        "combined": {"mse": combined_mse, "rmse": combined_rmse, "mae": combined_mae},
    }


def save_metrics_to_file(metrics_dict, target_name, is_binary, metrics_file):
    """Save metrics to a text file."""
    if is_binary:
        metrics_file.write(f"Target Column {target_name} - Binary Classification Metrics:\n")
        for model_name in ["nn", "xgb", "combined"]:
            m = metrics_dict[model_name]
            metrics_file.write(f"{model_name.upper()}:\n")
            metrics_file.write(f"  ROC AUC: {m['auc']:.4f}\n")
            metrics_file.write(f"  PR-AUC: {m['pr_auc']:.4f}\n")
            metrics_file.write(f"  Optimal Threshold: {m['threshold']:.4f}\n")
            metrics_file.write(f"  Accuracy: {m['accuracy']:.4f}\n")
            metrics_file.write(f"  Precision: {m['precision']:.4f}\n")
            metrics_file.write(f"  Recall: {m['recall']:.4f}\n")
            metrics_file.write(f"  F1 Score: {m['f1']:.4f}\n")
            metrics_file.write(f"  MCC: {m['mcc']:.4f}\n\n")
    else:
        metrics_file.write(f"Target Column {target_name} - Regression Metrics:\n")
        for model_name in ["nn", "xgb", "combined"]:
            m = metrics_dict[model_name]
            metrics_file.write(f"{model_name.upper()}:\n")
            metrics_file.write(f"  MSE: {m['mse']:.4f}\n")
            metrics_file.write(f"  RMSE: {m['rmse']:.4f}\n")
            metrics_file.write(f"  MAE: {m['mae']:.4f}\n\n")


def main(X_train, X_test, y_train, y_test, y_cols):
    # Create a timestamped output directory
    output_dir = Path(f'model_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open metrics file
    metrics_file_path = output_dir / "metrics_summary.txt"
    metrics_file = metrics_file_path.open("w")


    # Standardize features
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)

    # Initialize dictionaries to hold models and their corresponding scores
    nn_models = {}
    xgb_models = {}
    nn_scores = {}
    xgb_scores = {}
    combined_scores = {}

    # Number of target columns
    n_targets = y_train.shape[1]

    # Iterate over each column in the target variable
    for col_index in range(n_targets):
        target_name = y_cols[col_index]

        # Filter valid samples (no NaNs in X or y)
        X_train_col, y_train_col = filter_valid_samples(X_train_scaled, y_train[:, col_index])
        X_test_col, y_test_col = filter_valid_samples(X_test_scaled, y_test[:, col_index])

        # Skip if no valid data
        if len(y_train_col) == 0 or len(y_test_col) == 0:
            print(f"Target Column {target_name} - No valid data available. Skipping.")
            continue

        # Determine if the target is binary
        unique_values = np.unique(y_train_col)
        if unique_values.size <= 2 and set(unique_values).issubset({0, 1}):
            is_binary = True
        else:
            is_binary = False

        if is_binary:
            y_train_col_scaled = y_train_col
            y_test_col_scaled = y_test_col
            # Handle data imbalance
            num_positives = np.sum(y_train_col == 1)
            num_negatives = np.sum(y_train_col == 0)
            if num_positives == 0 or num_negatives == 0:
                print(f"Target Column {target_name} - Only one class present in y_train. Skipping.")
                continue
            scale_pos_weight = num_negatives / num_positives
            # Compute class weights
            class_weights = {0: scale_pos_weight, 1: 1.0}
        else:
            y_scaler = StandardScaler()
            y_train_col_dense: np.ndarray = np.asarray(y_train_col)
            y_test_col_dense: np.ndarray = np.asarray(y_test_col)
            y_train_col_scaled = y_scaler.fit_transform(y_train_col.reshape(-1, 1)).flatten()
            y_test_col_scaled = y_scaler.transform(y_test_col.reshape(-1, 1)).flatten()

        # Build and train NN model
        nn_model = build_nn_model(X_train_col.shape[1], is_binary)
        # Visualize the model architecture
        plot_model(
            nn_model,
            to_file=str(output_dir / f"neural_network_structure_{target_name}.png"),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
        )
        history = train_nn_model(
            nn_model, X_train_col, y_train_col_scaled, is_binary, target_name, class_weights if is_binary else None, output_dir
        )
        plot_learning_curve(history, target_name, output_dir)
        # Load the best model
        best_nn_model = load_model(output_dir / f"best_model_{target_name}.keras")

        # Extract intermediate features
        intermediate_layer_model = Model(inputs=best_nn_model.input, outputs=best_nn_model.layers[-2].output)
        intermediate_train_features = intermediate_layer_model.predict(X_train_col)
        intermediate_test_features = intermediate_layer_model.predict(X_test_col)

        # Train XGBoost model
        xgb_model = train_xgb_model(
            intermediate_train_features,
            y_train_col_scaled if not is_binary else y_train_col,
            is_binary,
            scale_pos_weight if is_binary else None,
            output_dir,
            target_name,
        )

        # Store models
        nn_models[col_index] = best_nn_model
        xgb_models[col_index] = xgb_model

        # Evaluate models
        nn_predictions_scaled = best_nn_model.predict(X_test_col)

        if is_binary:
            # Classification evaluation
            y_test_col_int = y_test_col.astype(int)
            nn_predictions_proba = nn_predictions_scaled.flatten()
            xgb_predictions_proba = xgb_model.predict_proba(intermediate_test_features)[:, 1]
            combined_proba = (nn_predictions_proba + xgb_predictions_proba) / 2

            # Calculate metrics for each model
            nn_metrics = calculate_classification_metrics(y_test_col_int, nn_predictions_proba)
            xgb_metrics = calculate_classification_metrics(y_test_col_int, xgb_predictions_proba)
            combined_metrics = calculate_classification_metrics(y_test_col_int, combined_proba)

            # Plot ROC Curves
            roc_data_list = [
                {"fpr": nn_metrics["fpr"], "tpr": nn_metrics["tpr"], "auc": nn_metrics["auc"], "model_name": "NN"},
                {"fpr": xgb_metrics["fpr"], "tpr": xgb_metrics["tpr"], "auc": xgb_metrics["auc"], "model_name": "XGBoost"},
                {"fpr": combined_metrics["fpr"], "tpr": combined_metrics["tpr"], "auc": combined_metrics["auc"], "model_name": "Combined"},
            ]
            plot_roc_curves(roc_data_list, target_name, output_dir)

            # Plot PR Curves
            pr_data_list = [
                {
                    "recall": nn_metrics["recall_curve"],
                    "precision": nn_metrics["precision_curve"],
                    "pr_auc": nn_metrics["pr_auc"],
                    "model_name": "NN",
                },
                {
                    "recall": xgb_metrics["recall_curve"],
                    "precision": xgb_metrics["precision_curve"],
                    "pr_auc": xgb_metrics["pr_auc"],
                    "model_name": "XGBoost",
                },
                {
                    "recall": combined_metrics["recall_curve"],
                    "precision": combined_metrics["precision_curve"],
                    "pr_auc": combined_metrics["pr_auc"],
                    "model_name": "Combined",
                },
            ]
            plot_pr_curves(pr_data_list, target_name, output_dir)

            # Plot Histograms
            plot_prediction_histogram(
                y_test_col_int, nn_predictions_proba, nn_metrics["threshold"], "Neural Network", target_name, output_dir
            )
            plot_prediction_histogram(y_test_col_int, xgb_predictions_proba, xgb_metrics["threshold"], "XGBoost", target_name, output_dir)
            plot_prediction_histogram(
                y_test_col_int, combined_proba, combined_metrics["threshold"], "Combined Model", target_name, output_dir
            )

            # Collect metrics
            metrics_dict = {"nn": nn_metrics, "xgb": xgb_metrics, "combined": combined_metrics}

            # Save metrics
            save_metrics_to_file(metrics_dict, target_name, is_binary, metrics_file)

            # Store scores
            nn_scores[col_index] = nn_metrics
            xgb_scores[col_index] = xgb_metrics
            combined_scores[col_index] = combined_metrics

        else:
            # Regression evaluation
            y_test_col = y_test_col.flatten()
            nn_predictions = y_scaler.inverse_transform(nn_predictions_scaled).flatten()
            xgb_predictions_scaled = xgb_model.predict(intermediate_test_features)
            xgb_predictions = y_scaler.inverse_transform(xgb_predictions_scaled.reshape(-1, 1)).flatten()
            combined_predictions = (nn_predictions + xgb_predictions) / 2

            metrics_dict = evaluate_regression_models(
                y_test_col, nn_predictions, xgb_predictions, combined_predictions, target_name, output_dir
            )
            # Save metrics
            save_metrics_to_file(metrics_dict, target_name, is_binary, metrics_file)

            # Store scores
            nn_scores[col_index] = metrics_dict["nn"]
            xgb_scores[col_index] = metrics_dict["xgb"]
            combined_scores[col_index] = metrics_dict["combined"]

    # Close the metrics file
    metrics_file.close()

    # After processing all targets, print summary
    print("Summary of Model Performance:")
    for col_index in range(n_targets):
        target_name = y_cols[col_index]
        if col_index in nn_scores:
            if "auc" in nn_scores[col_index]:
                print(
                    f"{target_name} - NN AUC: {nn_scores[col_index]['auc']:.4f}, "
                    f"XGBoost AUC: {xgb_scores[col_index]['auc']:.4f}, "
                    f"Combined AUC: {combined_scores[col_index]['auc']:.4f}"
                )
            else:
                print(
                    f"{target_name} - NN RMSE: {nn_scores[col_index]['rmse']:.4f}, "
                    f"XGBoost RMSE: {xgb_scores[col_index]['rmse']:.4f}, "
                    f"Combined RMSE: {combined_scores[col_index]['rmse']:.4f}"
                )
        else:
            print(f"{target_name} - No valid data for this target.")


if __name__ == "__main__":
    # Run the preprocessing
    X_train, X_test, y_train, y_test, y_cols = preprocess_data()

    # Run the main function
    main(X_train, X_test, y_train, y_test, y_cols)
