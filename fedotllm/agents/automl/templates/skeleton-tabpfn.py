### UNMODIFIABLE IMPORT BEGIN ###
import random
from pathlib import Path
import numpy as np
import pandas as pd
from automl import train_model, evaluate_model, automl_predict
### UNMODIFIABLE IMPORT END ###
# USER CODE BEGIN IMPORTS #
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# USER CODE END IMPORTS #

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

### UNMODIFIABLE CODE BEGIN ###
DATASET_PATH = Path("{%dataset_path%}") # path for saving and loading dataset(s)
WORKSPACE_PATH = Path("{%work_dir_path%}")
PIPELINE_PATH = WORKSPACE_PATH / "pipeline" # path for saving and loading the trained model
SUBMISSION_PATH = WORKSPACE_PATH / "submission.csv"
EVAL_SET_SIZE = 0.2 # 20% of the data for evaluation
### UNMODIFIABLE CODE END ###

# --- TODO: Update these paths for your specific dataset ---
TRAIN_FILE = DATASET_PATH / "train.csv" # Replace with your actual filename
TEST_FILE = DATASET_PATH / "test.csv" # Replace with your actual filename
SAMPLE_SUBMISSION_FILE = DATASET_PATH / "sample_submission.csv" # Replace with your actual filename or None
ID_COLUMN = "id" # Replace with your actual ID column name, if any
TARGET_COLUMNS = ["target"] # Replace with your actual target column name(s)


def load_data():
    """Loads train, test, and optionally sample submission files."""
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        sample_sub_df = None
        if SAMPLE_SUBMISSION_FILE and (DATASET_PATH / SAMPLE_SUBMISSION_FILE).exists():
           sample_sub_df = pd.read_csv(DATASET_PATH / SAMPLE_SUBMISSION_FILE)
        else:
            print("Sample submission file not found or not specified.")
        print("Data loaded successfully.")
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df, sample_sub_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please check filenames.")
        return None, None, None

# --------------------------------------------------------------------------- #
# Section 2: Data Cleaning
# --------------------------------------------------------------------------- #

def cleaning_data(df_train: pd.DataFrame, df_test: pd.DataFrame, target_cols: list):
    print("\nCleaning data...")
    X_train = df_train.copy()
    X_test = df_test.copy()

    # --- Step 0: Handle specific data cleaning tasks ---

    y_train = None
    if all(col in X_train.columns for col in target_cols):
        y_train = X_train[target_cols].copy()
        X_train = X_train.drop(columns=target_cols)
    else:
        missing_cols = [col for col in target_cols if col not in X_train.columns]
        print(f"Warning: Target column(s) {missing_cols} not in training data.")
        present_target_cols = [col for col in target_cols if col in X_train.columns]
        if present_target_cols:
            y_train = X_train[present_target_cols].copy()
            X_train = X_train.drop(columns=present_target_cols)
            print(f"Using available target columns: {present_target_cols}")

    test_cols_to_drop = [col for col in target_cols if col in X_test.columns]
    if test_cols_to_drop:
        print(f"Warning: Target column(s) {test_cols_to_drop} found in test data and removed.")
        X_test.drop(columns=test_cols_to_drop, inplace=True)

    if ID_COLUMN in X_train: X_train.drop(columns=[ID_COLUMN], inplace=True)
    if ID_COLUMN in X_test: X_test.drop(columns=[ID_COLUMN], inplace=True)

    return X_train, X_test, y_train

# --------------------------------------------------------------------------- #
# Section 3: Data Preprocessing
# --------------------------------------------------------------------------- #
def preprocess_data(train_features: pd.DataFrame, test_features: pd.DataFrame):
    """Transforms descriptors into a suitable format for TabPFN modeling.

    Guidelines for protein/antibody descriptor data:
    1. Missing Value Imputation:
       - Use median imputation for numerical descriptor columns.
       - Fit imputers ONLY on training data, then transform both train and test.
    2. Feature Scaling:
       - StandardScaler improves TabPFN stability on descriptors with very different scales.
       - Fit ONLY on training data.
    3. Column Dropping:
       - Drop non-informative columns (zero-variance, near-constant, or ID-like).
    4. All features must be numeric (int/float) before returning.
    5. Consistency: all transforms applied to train MUST be applied identically to test.
    """

    print("\nPreprocessing data...")
    X_train = train_features.copy()
    X_test = test_features.copy()

    # --- Step 0: Column Dropping ---
    # Leave only important features
    important_features = [ ]
    X_train = X_train[important_features]
    X_test = X_test[important_features]

    # --- Step 1: Identify feature types (based on training data) ---
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- Step 2: Handle Missing Values (Fit on Train, Transform Train & Test) ---
    # Numerical Imputation
    # if numerical_cols:
    #     num_imputer = SimpleImputer(strategy='median')
    #     X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
    #     X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

    # --- Step 3: Feature Scaling (Fit on Train, Transform Train & Test) ---
    # Example:
    # scaler = StandardScaler()
    # X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    # X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # --- Step 4: Feature Engineering ---
    # Create new features from existing ones if domain knowledge suggests it.
    # Example:
    # X_train['feature_name'] = X_train['col_a'] / (X_train['col_b'] + 1e-6)
    # X_test['feature_name']  = X_test['col_a']  / (X_test['col_b']  + 1e-6)

    print("Preprocessing complete.")
    print(f"Train processed shape: {X_train.shape}, Test processed shape: {X_test.shape}")
    return X_train, X_test

# --------------------------------------------------------------------------- #
# Section 6: Final Model Training & Submission
# --------------------------------------------------------------------------- #
def create_submission_file(test_ids, test_predictions, sample_sub_df, submission_filename="submission.csv"):
    """Creates the submission file in the required format."""
    print(f"\nCreating submission file: {submission_filename}")
    if isinstance(test_predictions, np.ndarray) and test_predictions.ndim > 1:
        test_predictions = test_predictions.flatten()
    if not isinstance(test_ids, (pd.Series, np.ndarray)):
        try:
            if sample_sub_df is not None and ID_COLUMN in sample_sub_df.columns:
                test_ids = sample_sub_df[ID_COLUMN]
            else:
                test_ids = np.arange(len(test_predictions))
                print(f"Warning: Using default range for test_ids as {ID_COLUMN} not found in sample submission.")
        except Exception as e:
            print(f"Warning: Could not load sample submission to get test_ids: {e}. Using default range.")
            test_ids = np.arange(len(test_predictions))

    submission_df = pd.DataFrame(test_predictions, columns=[TARGET_COLUMNS])
    submission_df.insert(0, ID_COLUMN, test_ids)

    submission_file_path = WORKSPACE_PATH / submission_filename
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file created at: {submission_file_path}")
    print("Submission file head:")
    print(submission_df.head())
    return submission_df

# --------------------------------------------------------------------------- #
# Main Orchestration Logic
# --------------------------------------------------------------------------- #
def main():
    """Main function to orchestrate the ML pipeline."""
    print("Starting ML Workflow...")
    # --- TODO: Define problem_type ('classification' or 'regression') ---
    current_problem_type = 'classification' # Example

    # --- Step 0: Load Data ---
    train_df, test_df, sample_sub_df = load_data()
    if train_df is None or test_df is None:
        print("Exiting due to data loading failure.")
        return

    # Store original test IDs for submission
    original_test_ids = test_df[ID_COLUMN] if ID_COLUMN in test_df.columns else test_df.index

    # --- Step 1: Cleaning ---
    X_train_cleaned, X_test_cleaned, y_train_full = cleaning_data(train_df, test_df, TARGET_COLUMNS)

    # --- Step 2: Preprocessing ---
    X_train_processed, X_test_processed = preprocess_data(X_train_cleaned, X_test_cleaned)

    # --- Create a fixed validation split ---
    print(f"\nCreating a fixed train-evaluation split (Eval size: {EVAL_SET_SIZE*100}%) ...")
    stratify_target = None
    if current_problem_type == 'classification' and len(TARGET_COLUMNS) == 1:
        stratify_target = y_train_full
    X_train_main, X_eval_holdout, y_train_main, y_eval_holdout = train_test_split(
        X_train_processed, y_train_full,
        test_size=EVAL_SET_SIZE,
        random_state=SEED,
        stratify=stratify_target
    )

    # --- Step 3: Train model ---
    # You must use pre-defined train_model function to train the model.
    model = train_model(X_train_main, y_train_main)

    # --- Step 4: Evaluate the trained model ---
    # You must use pre-defined evaluate_model function to evaluate the model.
    model_performance = evaluate_model(model, X_eval_holdout, y_eval_holdout)

    # --- Step 5: Predict on test set ---
    # You must use pre-defined automl_predict function to predict.
    predictions: np.ndarray = automl_predict(model, X_test_processed)

    # --- Step 6: Create submission file ---
    if predictions is not None:
        create_submission_file(original_test_ids, predictions, sample_sub_df, "submission.csv")
    else:
        print("No test predictions generated, skipping submission file creation.")
    print("\nML Workflow finished.")

if __name__ == "__main__":
    print("Files and directories:")
    paths = {
        "Dataset Path": DATASET_PATH,
        "Workspace Path": WORKSPACE_PATH,
        "Pipeline Path": PIPELINE_PATH,
        "Submission Path": SUBMISSION_PATH,
        "Train File": TRAIN_FILE,
        "Test File": TEST_FILE,
        "Sample Submission File": SAMPLE_SUBMISSION_FILE
    }
    for name, path in paths.items():
        print(f"{name}: {path}")
    main()
