import subprocess
import os
import sys

# Definición de rutas
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
COMPONENTS_DIR = os.path.join(BASE_DIR, "components")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. Preparación inicial de datos
print("Ejecutando preparación inicial de datos...")
processed_data_path = os.path.join(OUTPUT_DIR, "processed_data.csv")
subprocess.run([
    sys.executable,
    os.path.join(COMPONENTS_DIR, "initial_prep", "initial_prep.py"),
    "--dataset_path", os.path.join(DATA_DIR, "Breast Cancer METABRIC.csv"),
    "--processed_data_path", processed_data_path
], check=True)

# 2. Selección de características
print("Ejecutando selección de características...")
selected_features_path = os.path.join(OUTPUT_DIR, "selected_features.json")
subprocess.run([
    sys.executable,
    os.path.join(COMPONENTS_DIR, "feature_selection", "feature_selection.py"),
    "--processed_data_path", processed_data_path,
    "--selected_features_path", selected_features_path
], check=True)

# 3. Codificación y escalado
print("Ejecutando codificación y escalado...")
encoded_scaled_data_path = os.path.join(OUTPUT_DIR, "encoded_scaled_data.csv")
subprocess.run([
    sys.executable,
    os.path.join(COMPONENTS_DIR, "encode_scale", "encode_scale.py"),
    "--processed_data_path", processed_data_path,
    "--selected_features_path", selected_features_path,
    "--encoded_scaled_data_path", encoded_scaled_data_path
], check=True)

# 4. Split (entrenamiento/prueba)
print("Ejecutando split de datos...")
X_train_path = os.path.join(OUTPUT_DIR, "X_train")
X_test_path = os.path.join(OUTPUT_DIR, "X_test")
y_train_path = os.path.join(OUTPUT_DIR, "y_train")
y_test_path = os.path.join(OUTPUT_DIR, "y_test")
for d in [X_train_path, X_test_path, y_train_path, y_test_path]:
    if not os.path.exists(d):
        os.makedirs(d)
subprocess.run([
    sys.executable,
    os.path.join(COMPONENTS_DIR, "split_data", "split_data.py"),
    "--encoded_scaled_data_path", encoded_scaled_data_path,
    "--X_train_path", X_train_path,
    "--X_test_path", X_test_path,
    "--y_train_path", y_train_path,
    "--y_test_path", y_test_path
], check=True)

# 5. Entrenamiento SVM
print("Entrenando modelo SVM...")
svm_model_output_path = os.path.join(OUTPUT_DIR, "svm_model")
if not os.path.exists(svm_model_output_path):
    os.makedirs(svm_model_output_path)
subprocess.run([
    sys.executable,
    os.path.join(COMPONENTS_DIR, "svm_training", "svm_training.py"),
    "--X_train_path", X_train_path,
    "--X_test_path", X_test_path,
    "--y_train_path", y_train_path,
    "--y_test_path", y_test_path,
    "--svm_model_output_path", svm_model_output_path
], check=True)

# 6. Entrenamiento Random Forest
print("Entrenando modelo Random Forest...")
rf_model_output_path = os.path.join(OUTPUT_DIR, "rf_model")
if not os.path.exists(rf_model_output_path):
    os.makedirs(rf_model_output_path)
subprocess.run([
    sys.executable,
    os.path.join(COMPONENTS_DIR, "rf_training", "rf_training.py"),
    "--X_train_path", X_train_path,
    "--X_test_path", X_test_path,
    "--y_train_path", y_train_path,
    "--y_test_path", y_test_path,
    "--rf_model_output_path", rf_model_output_path
], check=True)

# 7. Comparación de modelos
print("Comparando modelos...")
best_model_run_id_path = os.path.join(OUTPUT_DIR, "best_model_run_id")
comparison_results_path = os.path.join(OUTPUT_DIR, "comparison_results")
for d in [best_model_run_id_path, comparison_results_path]:
    if not os.path.exists(d):
        os.makedirs(d)
subprocess.run([
    sys.executable,
    os.path.join(COMPONENTS_DIR, "model_comparison", "model_comparison.py"),
    "--best_model_run_id_path", best_model_run_id_path,
    "--comparison_results_path", comparison_results_path
], check=True)

# 8. Registro del mejor modelo
print("Registrando el mejor modelo...")
subprocess.run([
    sys.executable,
    os.path.join(COMPONENTS_DIR, "register_best_model", "register_best_model.py"),
    "--best_model_run_id_path", best_model_run_id_path,
    "--model_name", "breast_cancer_best_model"
], check=True)

print("\n¡Pipeline ejecutado y orquestado con éxito de principio a fin!")
