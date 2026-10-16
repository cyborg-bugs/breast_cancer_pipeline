
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import mlflow
import mlflow.sklearn
import numpy as np # Import numpy for handling potential np.bool_ in y_test
import shutil

def train_svm_model(X_train_path, X_test_path, y_train_path, y_test_path, svm_model_output_path, preprocessor_path, selected_features_path):
    """
    Trains an SVM model, evaluates it, and logs metrics and artifacts with MLflow.

    Args:
        X_train_path (str): Path to the training features CSV file.
        X_test_path (str): Path to the testing features CSV file.
        y_train_path (str): Path to the training labels CSV file.
        y_test_path (str): Path to the testing labels CSV file.
        svm_model_output_path (str): Path to save the trained SVM model artifact.
        preprocessor_path (str): Path to the fitted preprocessor object.
        selected_features_path (str): Path to the JSON file containing the list of selected features.
    """
    print(f"Loading data from: {X_train_path}, {X_test_path}, {y_train_path}, {y_test_path}")
    try:
        X_train = pd.read_csv(os.path.join(X_train_path, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(X_test_path, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(y_train_path, 'y_train.csv')).squeeze() # Use squeeze to get a Series
        y_test = pd.read_csv(os.path.join(y_test_path, 'y_test.csv')).squeeze() # Use squeeze to get a Series

        # Convert target Series to int type (0 or 1) as it was binarized in split_data.py
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        print("Data loaded successfully.")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Start an MLflow run
    # Set the MLflow experiment name explicitly
    mlflow.set_experiment("Breast Cancer Recurrence Prediction")
    print("Starting MLflow run...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Define and log parameters
        svm_params = {
            'kernel': 'rbf',
            'probability': True, # Enable probability prediction for ROC AUC
            'random_state': 42
        }
        print(f"Logging parameters: {svm_params}")
        mlflow.log_params(svm_params)

        # Initialize and train the SVM classifier
        print("Training SVM model...")
        svm_model = SVC(**svm_params)
        svm_model.fit(X_train, y_train)
        print("SVM model training completed.")

        # Make predictions on the test set
        print("Making predictions on test set...")
        y_pred = svm_model.predict(X_test)

        # Calculate evaluation metrics
        print("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report_dict = classification_report(y_test, y_pred, output_dict=True)

        # Log metrics
        print("Logging metrics...")
        mlflow.log_metric("accuracy", accuracy)

        # Safely access classification report metrics, handling potential missing labels
        precision_recurrent = class_report_dict.get('True', {}).get('precision', 0.0)
        recall_recurrent = class_report_dict.get('True', {}).get('recall', 0.0)
        f1_score_recurrent = class_report_dict.get('True', {}).get('f1-score', 0.0)

        mlflow.log_metric("precision_recurrent", precision_recurrent)
        mlflow.log_metric("recall_recurrent", recall_recurrent)
        mlflow.log_metric("f1_score_recurrent", f1_score_recurrent)


        # Log ROC AUC (using predict_proba as probability=True is set)
        try:
            y_scores = svm_model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class (True)
            roc_auc = roc_auc_score(y_test, y_scores)
            mlflow.log_metric("roc_auc", roc_auc)
            print(f"Logged ROC AUC: {roc_auc:.4f}")

            # Create and log ROC Curve plot
            print("Generating and logging ROC Curve plot...")
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC - SVM')
            plt.legend(loc="lower right")
            roc_plot_path = "svm_roc_curve.png"
            plt.savefig(roc_plot_path) # Save the plot as an artifact
            mlflow.log_artifact(roc_plot_path)
            plt.close() # Close the plot to free memory
            print(f"ROC Curve plot logged as artifact: {roc_plot_path}")

        except AttributeError:
            print("SVM model does not have predict_proba even with probability=True.")
        except ValueError as e:
             print(f"Error calculating or plotting ROC AUC: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during ROC AUC processing: {e}")


        # Create and log Confusion Matrix plot
        print("Generating and logging Confusion Matrix plot...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión - SVM')
        conf_matrix_plot_path = "svm_confusion_matrix.png"
        plt.savefig(conf_matrix_plot_path) # Save the plot as an artifact
        mlflow.log_artifact(conf_matrix_plot_path)
        plt.close() # Close the plot
        print(f"Confusion Matrix plot logged as artifact: {conf_matrix_plot_path}")


        # Copiar el preprocesador y las características seleccionadas a la carpeta de salida del modelo
        print(f"Copiando preprocesador desde {preprocessor_path} a {svm_model_output_path}/")
        shutil.copy(preprocessor_path, svm_model_output_path)
        print(f"Copiando selected_features desde {selected_features_path} a {svm_model_output_path}/")
        shutil.copy(selected_features_path, svm_model_output_path)


        # Save the trained model locally
        print(f"Saving trained SVM model locally to: {svm_model_output_path}")
        mlflow.sklearn.save_model(
            sk_model=svm_model,
            path=svm_model_output_path
        )
        print("SVM model saved locally successfully.")

        # Log the saved model directory as an MLflow artifact
        print(f"Logging saved SVM model directory as MLflow artifact: {svm_model_output_path}")
        mlflow.log_artifact(local_path=svm_model_output_path, artifact_path="svm_model")
        print("SVM model directory logged as MLflow artifact successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Model Training Script")
    parser.add_argument("--X_train_path", type=str, help="Path to the training features folder", required=True)
    parser.add_argument("--X_test_path", type=str, help="Path to the testing features folder", required=True)
    parser.add_argument("--y_train_path", type=str, help="Path to the training labels folder", required=True)
    parser.add_argument("--y_test_path", type=str, help="Path to the testing labels folder", required=True)
    parser.add_argument("--svm_model_output_path", type=str, help="Path to save the trained SVM model artifact", required=True)
    parser.add_argument("--preprocessor_path", type=str, help="Path to the fitted preprocessor object", required=True)
    parser.add_argument("--selected_features_path", type=str, help="Path to the JSON file containing the list of selected features", required=True)
    args = parser.parse_args()

    # Ensure output path exists before passing to the function
    os.makedirs(args.svm_model_output_path, exist_ok=True)

    train_svm_model(
        args.X_train_path,
        args.X_test_path,
        args.y_train_path,
        args.y_test_path,
        args.svm_model_output_path,
        args.preprocessor_path,
        args.selected_features_path
    )
