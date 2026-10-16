import os
import json
import numpy as np
import pandas as pd
import joblib
from mlflow.pyfunc import PythonModel # Import PythonModel to ensure compatibility

# Variables globales para el modelo, preprocesador y características seleccionadas
model = None
preprocessor = None
selected_features = None

def init():
    global model, preprocessor, selected_features
    # Asegúrate de que las rutas se ajusten a cómo MLflow organiza los artefactos
    # Cuando se registra un modelo, MLflow lo guarda en un directorio llamado 'model'
    # dentro del artefacto del run.

    # Ruta donde MLflow monta el modelo registrado
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")

    # Cargar el modelo del mejor run
    # Para cargar un modelo de MLflow (sklearn), se usa mlflow.pyfunc.load_model o mlflow.sklearn.load_model
    # Aquí usamos PythonModel para mayor flexibilidad, aunque mlflow.sklearn.load_model también es una opción.
    print(f"Cargando el modelo desde: {model_path}")
    try:
        model = joblib.load(os.path.join(model_path, "model.pkl")) # Asumiendo que el modelo se guarda como model.pkl
        print("Modelo cargado con éxito.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise

    # Cargar el preprocesador (ColumnTransformer) y las selected_features
    # Estos se deben haber guardado como artefactos separados en el run de encode_scale
    # Para el despliegue, asumiremos que se han registrado como artefactos separados
    # y están disponibles en el directorio del script o en un path conocido.
    # Por simplicidad aquí, asumiremos que están en la misma carpeta que score.py
    # En un escenario real de MLOps, estas rutas deberían ser manejadas por Azure ML
    # a través de artefactos o inputs del despliegue.

    # Para este ejemplo, asumiremos que los artefactos del preprocesador y las features
    # están disponibles en la misma carpeta 'scripts' o montados de alguna forma.
    # En un pipeline de Azure ML, estos serían outputs de un paso anterior y estarían en una ruta de montaje.

    # Temporalmente, buscaremos estos archivos en el directorio actual del script
    # En un despliegue real de Azure ML, las rutas de estos artefactos deben ser proporcionadas
    # como variables de entorno o cargarse desde el directorio del modelo si se empaquetaron juntos.

    # Adaptaremos esto en azure_ml_orchestrator.py para que se monten correctamente.
    current_script_dir = os.path.dirname(os.path.realpath(__file__))

    preprocessor_path = os.path.join(current_script_dir, "preprocessor.pkl") # Asumiendo nombre de archivo
    selected_features_path = os.path.join(current_script_dir, "selected_features.json")

    print(f"Cargando preprocesador desde: {preprocessor_path}")
    try:
        preprocessor = joblib.load(preprocessor_path)
        print("Preprocesador cargado con éxito.")
    except Exception as e:
        print(f"Error al cargar el preprocesador: {e}")
        raise

    print(f"Cargando características seleccionadas desde: {selected_features_path}")
    try:
        with open(selected_features_path, 'r') as f:
            selected_features = json.load(f)
        print("Características seleccionadas cargadas con éxito.")
    except Exception as e:
        print(f"Error al cargar las características seleccionadas: {e}")
        raise

def run(raw_data):
    try:
        # Convertir los datos de entrada a DataFrame
        data = pd.DataFrame(json.loads(raw_data))

        # Aplicar selección de características
        data_selected = data[selected_features]

        # Aplicar preprocesamiento
        data_processed = preprocessor.transform(data_selected)

        # Realizar predicciones
        predictions = model.predict(data_processed)
        probabilities = model.predict_proba(data_processed)[:, 1]

        # Convertir a lista y devolver
        return {"predictions": predictions.tolist(), "probabilities": probabilities.tolist()}
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
