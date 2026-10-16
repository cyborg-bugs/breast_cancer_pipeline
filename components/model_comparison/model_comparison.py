
import argparse
import json
import os
import mlflow
from mlflow.tracking import MlflowClient

def compare_models(best_model_run_id_path, comparison_results_path):
    """
    Compares trained models logged with MLflow based on a metric (ROC AUC)
    and identifies the best performing model.

    Args:
        best_model_run_id_path (str): Path to save the Run ID of the best model.
        comparison_results_path (str): Path to save the comparison results summary.
    """
    print("Starting model comparison...")

    # Initialize MLflow Client
    client = MlflowClient()

    # Get experiment details
    experiment_name = "Breast Cancer Recurrence Prediction"
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Error: Experiment '{experiment_name}' not found.")
        # Log a failure or handle appropriately
        return

    experiment_id = experiment.experiment_id
    print(f"Experiment '{experiment_name}' found with ID: {experiment_id}")

    # Search for the latest runs for SVM and Random Forest, ordered by ROC AUC descending
    print("Searching for latest SVM and Random Forest runs...")
    try:
        svm_runs = client.search_runs(
            experiment_id,
            filter_string="tags.mlflow.runName = 'SVM Model Training'",
            order_by=["metrics.roc_auc DESC"],
            max_results=1
        )

        rf_runs = client.search_runs(
            experiment_id,
            filter_string="tags.mlflow.runName = 'Random Forest Model Training'",
            order_by=["metrics.roc_auc DESC"],
            max_results=1
        )

        best_svm_run = svm_runs[0] if svm_runs else None
        best_rf_run = rf_runs[0] if rf_runs else None

        print("Latest runs retrieved.")

    except Exception as e:
        print(f"Error searching for MLflow runs: {e}")
        return


    best_model_run = None
    best_metric_value = -1
    best_model_name = None

    comparison_summary = {}

    print("\nComparing models based on ROC AUC:")

    if best_svm_run:
        svm_roc_auc = best_svm_run.data.metrics.get("roc_auc")
        comparison_summary["SVM_Run_ID"] = best_svm_run.info.run_id
        comparison_summary["SVM_ROC_AUC"] = svm_roc_auc
        print(f"SVM Run ID: {best_svm_run.info.run_id}, ROC AUC: {svm_roc_auc:.4f}" if svm_roc_auc is not None else f"SVM Run ID: {best_svm_run.info.run_id}, ROC AUC: N/A")
        if svm_roc_auc is not None and svm_roc_auc > best_metric_value:
            best_metric_value = svm_roc_auc
            best_model_run = best_svm_run
            best_model_name = "SVM"

    else:
        print("No SVM runs found with 'SVM Model Training' tag.")
        comparison_summary["SVM_Run_ID"] = "N/A"
        comparison_summary["SVM_ROC_AUC"] = "N/A"


    if best_rf_run:
        rf_roc_auc = best_rf_run.data.metrics.get("roc_auc")
        comparison_summary["RandomForest_Run_ID"] = best_rf_run.info.run_id
        comparison_summary["RandomForest_ROC_AUC"] = rf_roc_auc
        print(f"Random Forest Run ID: {best_rf_run.info.run_id}, ROC AUC: {rf_roc_auc:.4f}" if rf_roc_auc is not None else f"Random Forest Run ID: {best_rf_run.info.run_id}, ROC AUC: N/A")
        if rf_roc_auc is not None and rf_roc_auc > best_metric_value:
            best_metric_value = rf_roc_auc
            best_model_run = best_rf_run
            best_model_name = "Random Forest"
    else:
        print("No Random Forest runs found with 'Random Forest Model Training' tag.")
        comparison_summary["RandomForest_Run_ID"] = "N/A"
        comparison_summary["RandomForest_ROC_AUC"] = "N/A"


    if best_model_run:
        print(f"\nBest model based on ROC AUC is: {best_model_name} (Run ID: {best_model_run.info.run_id})")
        print(f"Best ROC AUC: {best_metric_value:.4f}")
        best_model_run_id = best_model_run.info.run_id
        comparison_summary["Best_Model_Name"] = best_model_name
        comparison_summary["Best_Model_Run_ID"] = best_model_run_id
        comparison_summary["Best_ROC_AUC"] = best_metric_value

        # Ensure output directory exists for the best model run ID file
        output_dir_run_id = os.path.dirname(best_model_run_id_path)
        if output_dir_run_id and not os.path.exists(output_dir_run_id):
            os.makedirs(output_dir_run_id, exist_ok=True)
            print(f"Created output directory for best model run ID: {output_dir_run_id}")

        # Save the best model's run ID to a file
        try:
            # Save as a simple text file
            with open(os.path.join(best_model_run_id_path, 'best_model_run_id.txt'), 'w') as f:
                f.write(best_model_run_id)
            print(f"Best model Run ID saved to: {best_model_run_id_path}")
        except Exception as e:
            print(f"Error saving best model Run ID: {e}")


        # Ensure output directory exists for the comparison results file
        output_dir_results = os.path.dirname(comparison_results_path)
        if output_dir_results and not os.path.exists(output_dir_results):
            os.makedirs(output_dir_results, exist_ok=True)
            print(f"Created output directory for comparison results: {output_dir_results}")

        # Save the comparison summary to a JSON file
        try:
            with open(os.path.join(comparison_results_path, 'comparison_results.json'), 'w') as f:
                json.dump(comparison_summary, f, indent=4)
            print(f"Comparison results saved to: {comparison_results_path}")
        except Exception as e:
            print(f"Error saving comparison results: {e}")


    else:
        print("\nCould not determine the best model based on ROC AUC.")
        comparison_summary["Best_Model_Name"] = "N/A"
        comparison_summary["Best_Model_Run_ID"] = "N/A"
        comparison_summary["Best_ROC_AUC"] = "N/A"

        # Still attempt to save comparison results even if no best model was found
        output_dir_results = os.path.dirname(comparison_results_path)
        if output_dir_results and not os.path.exists(output_dir_results):
            os.makedirs(output_dir_results, exist_ok=True)
            print(f"Created output directory for comparison results: {output_dir_results}")

        try:
            with open(os.path.join(comparison_results_path, 'comparison_results.json'), 'w') as f:
                json.dump(comparison_summary, f, indent=4)
            print(f"Comparison results (partial) saved to: {comparison_results_path}")
        except Exception as e:
            print(f"Error saving comparison results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Comparison Script")
    parser.add_argument("--best_model_run_id_path", type=str, help="Path to save the Run ID of the best model", required=True)
    parser.add_argument("--comparison_results_path", type=str, help="Path to save the comparison results summary", required=True)
    args = parser.parse_args()

    # Ensure output paths exist before passing to the function
    os.makedirs(args.best_model_run_id_path, exist_ok=True)
    os.makedirs(args.comparison_results_path, exist_ok=True)


    compare_models(args.best_model_run_id_path, args.comparison_results_path)

