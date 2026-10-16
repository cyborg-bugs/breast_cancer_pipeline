
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os
import argparse
import json

def feature_selection(processed_data_path, selected_features_path):
    """
    Performs feature selection on processed data based on statistical tests.

    Args:
        processed_data_path (str): Path to the input processed CSV dataset.
        selected_features_path (str): Path to save the list of selected features (as a JSON file).
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

    target_col = 'Relapse Free Status'
    all_selected_features = []

    # --- Feature Selection for Numerical Variables (based on correlation) ---
    print("\n--- Feature Selection for Numerical Variables ---")

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude the target column if it somehow became numerical and was included
    numerical_cols = [col for col in numerical_cols if col != target_col]

    if numerical_cols and target_col in df.columns:
        # Temporarily encode the target for correlation calculation if it's categorical
        # This assumes the target is binary and will be converted to 0/1
        # A more robust approach might involve checking the target dtype
        temp_df = df.copy()
        if temp_df[target_col].dtype == 'object':
             print(f"Encoding target variable '{target_col}' for correlation calculation.")
             # Basic binary encoding assuming two unique values after imputation
             unique_target_values = temp_df[target_col].unique()
             if len(unique_target_values) == 2:
                  temp_df[target_col] = temp_df[target_col].apply(lambda x: 1 if x == 'Recurred' else 0)
                  correlation_matrix_numerical = temp_df[numerical_cols + [target_col]].corr()
                  correlations_with_target = correlation_matrix_numerical[target_col].abs().sort_values(ascending=False)
                  # Remove the target's self-correlation
                  correlations_with_target = correlations_with_target.drop(target_col, errors='ignore')

                  print(f"Absolute correlation of numerical features with '{target_col}':")
                  print(correlations_with_target.to_string())

                  # Set a threshold for selecting numerical features (e.g., absolute correlation > 0.1)
                  numerical_features_selected = correlations_with_target[correlations_with_target > 0.1].index.tolist()
                  print(f"Numerical features selected (absolute correlation > 0.1): {numerical_features_selected}")
                  all_selected_features.extend(numerical_features_selected)
             else:
                print(f"Target variable '{target_col}' is categorical but not binary for simple correlation calculation.")
                print("Skipping numerical feature selection based on correlation.")
        else:
             print(f"Target variable '{target_col}' is already numerical. Calculating correlation.")
             correlation_matrix_numerical = temp_df[numerical_cols + [target_col]].corr()
             correlations_with_target = correlation_matrix_numerical[target_col].abs().sort_values(ascending=False)
             # Remove the target's self-correlation
             correlations_with_target = correlations_with_target.drop(target_col, errors='ignore')

             print(f"Absolute correlation of numerical features with '{target_col}':")
             print(correlations_with_target.to_string())

             # Set a threshold for selecting numerical features (e.g., absolute correlation > 0.1)
             numerical_features_selected = correlations_with_target[correlations_with_target > 0.1].index.tolist()
             print(f"Numerical features selected (absolute correlation > 0.1): {numerical_features_selected}")
             all_selected_features.extend(numerical_features_selected)

    else:
        print("No numerical columns found or target column not present for correlation analysis.")


    # --- Feature Selection for Categorical Variables (based on Chi-squared test) ---
    print("\n--- Feature Selection for Categorical Variables ---")

    # Identify categorical columns (excluding Patient ID and the target itself)
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    cols_to_exclude = ['Patient ID', target_col]
    categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude]


    categorical_features_selected = []

    print(f"Performing Chi-squared test for categorical features vs '{target_col}':")

    for col in categorical_cols:
        if col in df.columns and target_col in df.columns:
            contingency_table = pd.crosstab(df[col], df[target_col])
            # Perform Chi-squared test only if the contingency table is not empty and has more than one row/column
            if not contingency_table.empty and min(contingency_table.shape) > 1:
                try:
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    # Select feature if p-value is less than a significance level (e.g., 0.05)
                    if p < 0.05:
                        categorical_features_selected.append(col)
                    print(f"  '{col}': p-value = {p:.4f}")
                except ValueError as e:
                     print(f"  '{col}': Could not perform Chi-squared test - {e}")
                except Exception as e:
                     print(f"  '{col}': An unexpected error occurred during Chi-squared test - {e}")

            else:
                 print(f"  '{col}': Contingency table is empty or insufficient for Chi-squared test.")
        elif col not in df.columns:
            print(f"  '{col}': Column not found in the DataFrame.")


    print(f"\nCategorical features selected (p-value < 0.05): {categorical_features_selected}")
    all_selected_features.extend(categorical_features_selected)

    print("\n--- Summary of Selected Features ---")
    if all_selected_features:
        print("Selected features:")
        for feature in all_selected_features:
            print(f"- {feature}")
    else:
        print("No features were selected based on the criteria.")


    # Ensure the output directory exists for the selected features file
    output_dir = os.path.dirname(selected_features_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory for selected features: {output_dir}")


    # Save the list of selected features
    try:
        with open(selected_features_path, 'w') as f:
            json.dump(all_selected_features, f)
        print(f"List of selected features saved successfully to: {selected_features_path}")
    except Exception as e:
        print(f"Error saving selected features list: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Selection Script")
    parser.add_argument("--processed_data_path", type=str, help="Path to the input processed CSV dataset", required=True)
    parser.add_argument("--selected_features_path", type=str, help="Path to save the list of selected features (JSON file)", required=True)
    args = parser.parse_args()

    feature_selection(args.processed_data_path, args.selected_features_path)
