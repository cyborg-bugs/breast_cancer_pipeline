
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os
import argparse

def initial_prep(dataset_path, processed_data_path):
    """
    Loads the dataset, handles missing values, and saves the processed data.

    Args:
        dataset_path (str): Path to the input CSV dataset.
        processed_data_path (str): Path to save the processed CSV data.
    """
    print(f"Loading dataset from: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return


    print("Handling missing values...")

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Impute missing values in numerical columns with the median
    if numerical_cols:
        imputer_numerical = SimpleImputer(strategy='median')
        df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])
        print("Numerical values imputed with the median.")

    # Impute missing values in categorical columns with a constant value 'Unknown'
    if categorical_cols:
        imputer_categorical = SimpleImputer(strategy='constant', fill_value='Unknown')
        df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])
        print("Categorical values imputed with 'Unknown'.")

    print("Missing values after imputation:")
    print(df.isnull().sum().to_string()) # Use to_string() to display all rows


    # Ensure the output directory exists
    output_dir = os.path.dirname(processed_data_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")


    # Save the processed DataFrame
    try:
        df.to_csv(processed_data_path, index=False)
        print(f"Processed data saved successfully to: {processed_data_path}")
    except Exception as e:
        print(f"Error saving processed data: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initial Data Preparation Script")
    parser.add_argument("--dataset_path", type=str, help="Path to the input CSV dataset", required=True)
    parser.add_argument("--processed_data_path", type=str, help="Path to save the processed CSV data", required=True)
    args = parser.parse_args()

    initial_prep(args.dataset_path, args.processed_data_path)
