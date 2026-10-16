
import argparse
import json
import os
import mlflow
import mlflow.sklearn

def register_best_model(best_model_run_id_path, model_name, registered_model_output_path):
    """
    Registers the best performing model (identified by Run ID) in the
    Azure ML Model Registry.

    Args:
        best_model_run_id_path (str): Path to the file containing the Run ID of the best model.
        model_name (str): The name to use when registering the model in the registry.
        registered_model_output_path (str): Path to save the registered model name and version.
    """
    print("Starting model registration...")

    # Read the best model's Run ID from the input file
    best_model_run_id_file = os.path.join(best_model_run_id_path, 'best_model_run_id.txt')
    print(f"Reading best model Run ID from: {best_model_run_id_file}")
    try:
        with open(best_model_run_id_file, 'r') as f:
            best_model_run_id = f.read().strip()
        print(f"Best model Run ID read: {best_model_run_id}")
    except FileNotFoundError:
        print(f"Error: Best model Run ID file not found at {best_model_run_id_file}")
        return
    except Exception as e:
        print(f"Error reading best model Run ID: {e}")
        return

    if not best_model_run_id:
        print("Error: Best model Run ID is empty.")
        return

    comparison_results_file = os.path.join(os.path.dirname(best_model_run_id_path), 'comparison_results.json') # Assume comparison results are in the parent folder of best_model_run_id_path or passed as another input
    print(f"Attempting to read comparison results from: {comparison_results_file}")
    best_model_artifact_path_in_run = None
    try:
        comparison_results_dir = os.path.dirname(best_model_run_id_path)
        # Assuming comparison_results.json is in the same output folder as best_model_run_id.txt
        comparison_results_file = os.path.join(comparison_results_dir, 'comparison_results.json')
        print(f"Reading comparison results from: {comparison_results_file}")

        with open(comparison_results_file, 'r') as f:
            comparison_summary = json.load(f)

        best_model_name_from_summary = comparison_summary.get("Best_Model_Name")
        print(f"Best model name from summary: {best_model_name_from_summary}")

        # Infer the artifact path based on the best model name
        if best_model_name_from_summary == "SVM":
            best_model_artifact_path_in_run = 'svm_model' # This was the artifact_path used in svm_training.py
        elif best_model_name_from_summary == "Random Forest":
            best_model_artifact_path_in_run = 'rf_model' # This was the artifact_path used in rf_training.py
        else:
            print(f"Warning: Could not infer artifact path for best model '{best_model_name_from_summary}'. Assuming default 'model'.")
            best_model_artifact_path_in_run = 'model' # Fallback assumption


    except FileNotFoundError:
        print(f"Warning: Comparison results file not found at {comparison_results_file}. Cannot infer artifact path. Assuming default 'model'.")
        best_model_artifact_path_in_run = 'model' # Fallback assumption
    except Exception as e:
        print(f"Error reading comparison results: {e}. Cannot infer artifact path. Assuming default 'model'.")
        best_model_artifact_path_in_run = 'model' # Fallback assumption


    if not best_model_artifact_path_in_run:
         print("Error: Could not determine the artifact path for the best model.")
         return


    model_uri = f"runs:/{best_model_run_id}/{best_model_artifact_path_in_run}"
    print(f"Constructed MLflow Model URI: {model_uri}")

    # Register the model
    print(f"Registering model '{model_name}' from URI: {model_uri}")
    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        print(f"Model '{model_name}' registered successfully with version {registered_model.version}.")
        print(f"Registered Model Details: {registered_model}")

        # Save the registered model name and version to the output path
        output_data = {
            "registered_model_name": registered_model.name,
            "registered_model_version": registered_model.version
        }
        os.makedirs(registered_model_output_path, exist_ok=True)
        output_file = os.path.join(registered_model_output_path, "registered_model_info.json")
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Registered model info saved to: {output_file}")

    except Exception as e:
        print(f"Error registering model: {e}")
        # Log a failure or handle appropriately


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register Best Model Script")
    parser.add_argument("--best_model_run_id_path", type=str, help="Path to the folder containing the best model Run ID file", required=True)
    parser.add_argument("--model_name", type=str, help="Name for the registered model in Azure ML Model Registry", required=True)
    parser.add_argument("--registered_model_output_path", type=str, help="Path to save the registered model name and version", required=True)
    args = parser.parse_args()

    register_best_model(args.best_model_run_id_path, args.model_name, args.registered_model_output_path)

