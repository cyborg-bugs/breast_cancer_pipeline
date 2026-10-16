import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import argparse
import json
import joblib # Import joblib for saving/loading the preprocessor

def encode_scale(processed_data_path, selected_features_path, encoded_scaled_data_path, preprocessor_output_path):
    """
    Encodes categorical features and scales numerical features based on selected features.

    Args:
        processed_data_path (str): Path to the input processed CSV dataset.
        selected_features_path (str): Path to the JSON file containing the list of selected features.
        encoded_scaled_data_path (str): Path to save the encoded and scaled CSV data.
        preprocessor_output_path (str): Path to save the fitted ColumnTransformer object.
    """
    print(f"Loading processed data from: {processed_data_path}")
    try:
        df = pd.read_csv(processed_data_path)
        print("Processed data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Processed data not found at {processed_data_path}")
        return
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return

    print(f"Loading selected features from: {selected_features_path}")
    try:
        with open(selected_features_path, 'r') as f:
            selected_features = json.load(f)
        print(f"Selected features loaded successfully: {selected_features}")
    except FileNotFoundError:
        print(f"Error: Selected features file not found at {selected_features_path}")
        return
    except Exception as e:
        print(f"Error loading selected features: {e}")
        return

    # Define target column (assuming it's in the original data but not selected as a feature)
    target_col = 'Relapse Free Status'

    # Separate features (X) and target (y)
    # Keep only the selected features for preprocessing
    X = df[selected_features].copy()
    y = df[target_col].copy() # Keep the target separately

    print("Original data shape:", df.shape)
    print("Features (X) shape before preprocessing:", X.shape)
    print("Target (y) shape:", y.shape)

    # Identify numerical and categorical columns among the selected features
    numerical_cols_selected = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_selected = X.select_dtypes(include='object').columns.tolist()

    print(f"Selected Numerical Columns for Scaling: {numerical_cols_selected}")
    print(f"Selected Categorical Columns for Encoding: {categorical_cols_selected}")


    # --- Preprocessing Pipeline ---
    # Create transformers for numerical and categorical columns
    transformers = []
    if numerical_cols_selected:
        transformers.append(('num', StandardScaler(), numerical_cols_selected))
    if categorical_cols_selected:
        # Handle unknown categories by ignoring them
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols_selected)) # Use sparse_output

    if not transformers:
        print("No numerical or categorical features selected for preprocessing.")
        # If no features were selected, just save the target column (or an empty dataframe + target)
        # For now, assume at least one feature is selected, or handle this edge case later if needed.
        # If no features, the pipeline might not be necessary, but let's proceed assuming features exist.
        # If no features were selected, we might just save the target or raise an error.
        # For this script, let's assume selected_features is not empty.
        # If it is empty, the script will likely fail later, which is acceptable for now.
        pass # Continue, the ColumnTransformer will handle empty list

    # Create a ColumnTransformer to apply different transformations to different columns
    # Use remainder='passthrough' to keep other columns (like Patient ID if it was somehow included, though we excluded it in selection)
    # Or use remainder='drop' if we only want the preprocessed selected features.
    # Given the goal is to prepare features for modeling, 'drop' is more appropriate for the feature set X.
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')


    # Create a preprocessing pipeline
    preprocess_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    print("Applying preprocessing pipeline...")
    try:
        X_processed = preprocess_pipeline.fit_transform(X)
        print("Preprocessing completed successfully.")
        print("Processed features (X_processed) shape:", X_processed.shape)

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    # Save the fitted preprocessor
    try:
        os.makedirs(os.path.dirname(preprocessor_output_path), exist_ok=True)
        joblib.dump(preprocess_pipeline, preprocessor_output_path)
        print(f"Preprocessor saved successfully to: {preprocessor_output_path}")
    except Exception as e:
        print(f"Error saving preprocessor: {e}")
        return

    # Convert the processed features back to a DataFrame
    # Get the feature names after preprocessing
    if categorical_cols_selected:
        # Get the feature names from the OneHotEncoder
        ohe_feature_names = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols_selected)
        all_processed_feature_names = numerical_cols_selected + list(ohe_feature_names)
    else:
         all_processed_feature_names = numerical_cols_selected # Only numerical features were processed

    df_processed = pd.DataFrame(X_processed, columns=all_processed_feature_names, index=X.index)

    # Add the target column back to the processed DataFrame
    df_processed[target_col] = y

    print("DataFrame after encoding, scaling, and adding target:")
    print(df_processed.head())
    print("Final DataFrame shape:", df_processed.shape)


    # Ensure the output directory exists
    output_dir = os.path.dirname(encoded_scaled_data_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Save the encoded and scaled DataFrame
    try:
        df_processed.to_csv(encoded_scaled_data_path, index=False)
        print(f"Encoded and scaled data saved successfully to: {encoded_scaled_data_path}")
    except Exception as e:
        print(f"Error saving encoded and scaled data: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode and Scale Features Script")
    parser.add_argument("--processed_data_path", type=str, help="Path to the input processed CSV dataset", required=True)
    parser.add_argument("--selected_features_path", type=str, help="Path to the JSON file containing the list of selected features", required=True)
    parser.add_argument("--encoded_scaled_data_path", type=str, help="Path to save the encoded and scaled CSV data", required=True)
    parser.add_argument("--preprocessor_output_path", type=str, help="Path to save the fitted preprocessor object", required=True)
    args = parser.parse_args()

    encode_scale(args.processed_data_path, args.selected_features_path, args.encoded_scaled_data_path, args.preprocessor_output_path)
