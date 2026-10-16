import os, json
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, dsl, Input, Output
from azure.ai.ml.entities import PipelineJob, Data, Environment
from azure.ai.ml import load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration
from azure.ai.ml.entities import Model

# === Configure your workspace ===
SUBSCRIPTION_ID = "442b281b-4257-4fcc-80bd-bcc85cf55c6d" # Replace with your Subscription ID
RESOURCE_GROUP  = "rg-miop-ucb" # Replace with your Resource Group name
WORKSPACE_NAME  = "mlops-ucb" # Replace with your Workspace name
COMPUTE_NAME    = "cpu-cluster" # Assuming you have a compute cluster named 'cpu-cluster' - update if necessary

# Instantiate MLClient
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

# ──────────────────────────────────────────────────────────────────────────────
# 0) Conexión al Workspace
# ──────────────────────────────────────────────────────────────────────────────
print("0) Conectando al Workspace…")
ws = ml_client.workspaces.get(WORKSPACE_NAME)
print(f"   ✔ Conectado a Workspace: {ws.name}")

# ──────────────────────────────────────────────────────────────────────────────
# 0.5) Registro del entorno de ejecución
# ──────────────────────────────────────────────────────────────────────────────
print("0.5) Registrando entorno breast_cancer_env…")
env_name = "breast_cancer_env"
env_file = os.path.join(os.path.dirname(__file__), "env", "conda.yml")

try:
    # Intenta cargar el entorno existente para verificar si está registrado
    existing_env = ml_client.environments.get(name=env_name, label="latest")
    print(f"   ✔ Entorno '{env_name}' ya existe y está registrado.")
    breast_cancer_env_asset = existing_env
except Exception:
    # Si no existe, créalo y regístralo
    print(f"   Creando y registrando entorno '{env_name}'...")
    breast_cancer_env_asset = Environment(
        name=env_name,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file=env_file,
        description="Entorno base para el pipeline de Breast Cancer"
    )
    ml_client.environments.create_or_update(breast_cancer_env_asset)
    print(f"   ✔ Entorno '{env_name}' creado y registrado con éxito.")

# ──────────────────────────────────────────────────────────────────────────────
# 1) Carga y registro de componentes YAML
# ──────────────────────────────────────────────────────────────────────────────
print("1) Cargando y registrando componentes YAML…")
components_folder = os.path.join(os.path.dirname(__file__), "components")

# Cargar y registrar los componentes de forma explícita
initial_prep_component_loaded = load_component(source=os.path.join(components_folder, "initial_prep/initial_prep.yml"))
initial_prep_component = ml_client.components.create_or_update(initial_prep_component_loaded)
initial_prep_component = ml_client.components.get(name=initial_prep_component.name, version="1.0.2")

feature_selection_component_loaded = load_component(source=os.path.join(components_folder, "feature_selection/feature_selection.yml"))
feature_selection_component = ml_client.components.create_or_update(feature_selection_component_loaded)
feature_selection_component = ml_client.components.get(name=feature_selection_component.name, version="1.0.2")

encode_scale_component_loaded = load_component(source=os.path.join(components_folder, "encode_scale/encode_scale.yml"))
encode_scale_component = ml_client.components.create_or_update(encode_scale_component_loaded)
encode_scale_component = ml_client.components.get(name=encode_scale_component.name, version="1.0.3")

split_data_component_loaded = load_component(source=os.path.join(components_folder, "split_data/split_data.yml"))
split_data_component = ml_client.components.create_or_update(split_data_component_loaded)
split_data_component = ml_client.components.get(name=split_data_component.name, version="1.0.2")

svm_training_component_loaded = load_component(source=os.path.join(components_folder, "svm_training/svm_training.yml"))
svm_training_component = ml_client.components.create_or_update(svm_training_component_loaded)
svm_training_component = ml_client.components.get(name=svm_training_component.name, version="1.0.2")

rf_training_component_loaded = load_component(source=os.path.join(components_folder, "rf_training/rf_training.yml"))
rf_training_component = ml_client.components.create_or_update(rf_training_component_loaded)
rf_training_component = ml_client.components.get(name=rf_training_component.name, version="1.0.2")

model_comparison_component_loaded = load_component(source=os.path.join(components_folder, "model_comparison/model_comparison.yml"))
model_comparison_component = ml_client.components.create_or_update(model_comparison_component_loaded)
model_comparison_component = ml_client.components.get(name="model_comparison", version="1.0.5") # Version 1.0.4 because 1.0.3 was used before

register_best_model_component_loaded = load_component(source=os.path.join(components_folder, "register_best_model/register_best_model.yml"))
register_best_model_component = ml_client.components.create_or_update(register_best_model_component_loaded)
register_best_model_component = ml_client.components.get(name="register_best_model_v2", version="1.0.6") # Increment version of register_best_model_v2 too

print("   ✔ Componentes cargados y registrados: initial_prep, feature_selection, encode_scale, split_data, svm_training, rf_training, model_comparison, register_best_model_v2")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Registro/actualización del Data Asset
# ──────────────────────────────────────────────────────────────────────────────
print("2) Creando/actualizando Data Asset breast-cancer-metabric…")

data_path = os.path.join(os.path.dirname(__file__), "data", "Breast Cancer METABRIC.csv")
data_asset = Data(
    name="breast-cancer-metabric",
    description="Breast Cancer METABRIC dataset for pipeline training",
    path=data_path,
    type="uri_file"
)
try:
    created_data_asset = ml_client.data.create_or_update(data_asset)
    print(f"   ✔ Data asset registrado: {created_data_asset.name} v{created_data_asset.version}")
    data_ref = Input(
        type="uri_file",
        path=f"azureml:{created_data_asset.name}:{created_data_asset.version}"
    )
    print(f"   ✔ Data asset reference created: {data_ref.path}")
except Exception as ex:
    print("Failed to create or update Data Asset.")
    print(ex)
    exit()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Definición del Pipeline
# ──────────────────────────────────────────────────────────────────────────────
print("3) Definiendo pipeline breast_cancer_pipeline…")

@pipeline(compute=COMPUTE_NAME, description="Breast Cancer Prediction Pipeline", tags={"dataset": "breast_cancer", "model_training": "svm_rf"})
def breast_cancer_pipeline(
    raw_data: Input,
    registered_model_output_uri: Output
):
    initial_prep_step = initial_prep_component(dataset_path=raw_data)
    feature_selection_step = feature_selection_component(processed_data_path=initial_prep_step.outputs.processed_data_path)
    encode_scale_step = encode_scale_component(
        processed_data_path=initial_prep_step.outputs.processed_data_path,
        selected_features_path=feature_selection_step.outputs.selected_features_path
    )
    split_data_step = split_data_component(encoded_scaled_data_path=encode_scale_step.outputs.encoded_scaled_data_path)
    svm_training_step = svm_training_component(
        X_train_path=split_data_step.outputs.X_train_path,
        y_train_path=split_data_step.outputs.y_train_path,
        X_test_path=split_data_step.outputs.X_test_path,
        y_test_path=split_data_step.outputs.y_test_path,
        preprocessor_path=encode_scale_step.outputs.preprocessor_output_path,
        selected_features_path=feature_selection_step.outputs.selected_features_path
    )
    rf_training_step = rf_training_component(
        X_train_path=split_data_step.outputs.X_train_path,
        y_train_path=split_data_step.outputs.y_train_path,
        X_test_path=split_data_step.outputs.X_test_path,
        y_test_path=split_data_step.outputs.y_test_path,
        preprocessor_path=encode_scale_step.outputs.preprocessor_output_path,
        selected_features_path=feature_selection_step.outputs.selected_features_path
    )
    model_comparison_step = model_comparison_component(
        svm_model_path=svm_training_step.outputs.svm_model_output_path,
        rf_model_path=rf_training_step.outputs.rf_model_output_path,
        test_data_path=split_data_step.outputs.X_test_path
    )
    register_best_model_step = register_best_model_component(
        best_model_run_id_path=model_comparison_step.outputs.best_model_run_id_path,
        model_name="breast_cancer_best_model"
    )

    registered_model_output_uri = register_best_model_step.outputs.registered_model_output_path

print("   ✔ Pipeline definido")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Construcción y Envío del job de pipeline
# ──────────────────────────────────────────────────────────────────────────────
print("4) Construyendo y enviando objeto PipelineJob a Azure ML…")
pipeline_job = breast_cancer_pipeline(
    raw_data=data_ref
)
pipeline_job.display_name = "Breast-Cancer-Prediction-Pipeline-AzureML"
pipeline_job.experiment_name = "breast-cancer-mlops-azureml"
pipeline_job.tags = {
    "dataset": "breast_cancer",
    "model_training": "svm_rf",
    "pipeline_type": "training"
}

returned_job = ml_client.jobs.create_or_update(pipeline_job)
print(f"   ✔ Pipeline enviado con éxito: {returned_job.name}")
print(f"   🔗 URL en Azure ML Studio: {returned_job.studio_url}")

# Wait for the pipeline job to complete to get the registered model name
ml_client.jobs.stream(returned_job.name)

# Obtener el modelo registrado directamente del registro de modelos de Azure ML
registered_model_name_for_deployment = "breast_cancer_best_model"
try:
    registered_model_for_deployment = ml_client.models.get(name=registered_model_name_for_deployment, label="latest")
    model_path = f"azureml:{registered_model_for_deployment.name}:{registered_model_for_deployment.version}"
    print(f"   ✔ Modelo registrado para despliegue: {model_path}")
except Exception as ex:
    print(f"Error al obtener el modelo registrado para despliegue: {ex}")
    print("Asegúrate de que el pipeline se haya ejecutado exitosamente y el modelo esté registrado.")
    exit()

# ──────────────────────────────────────────────────────────────────────────────
# 5) Despliegue del mejor modelo como endpoint en línea
# ──────────────────────────────────────────────────────────────────────────────
print("5) Desplegando el mejor modelo como endpoint en línea…")

# Endpoint creation
endpoint_name = f"breast-cancer-endpoint-{SUBSCRIPTION_ID[:4]}{RESOURCE_GROUP[-4:]}"
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Endpoint para predicción de cáncer de mama",
    auth_mode="key"
)

# Intenta obtener el endpoint existente, si no existe, créalo
try:
    ml_client.online_endpoints.get(name=endpoint_name)
    print(f"   ✔ Endpoint '{endpoint.name}' ya existe.")
except Exception:
    print(f"   Creando endpoint '{endpoint.name}'...")
    ml_client.begin_create_or_update(endpoint).wait()
    print(f"   ✔ Endpoint '{endpoint.name}' creado/actualizado.")

# Define the custom environment for the deployment
inference_env = Environment(
    name="breast_cancer_inference_env",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    conda_file=os.path.join(os.path.dirname(__file__), "scripts", "inference_conda.yml")
)
created_inference_env = ml_client.environments.create_or_update(inference_env)
# Obtener el entorno por su nombre y versión exacta
inference_env_asset = ml_client.environments.get(name=created_inference_env.name, version=created_inference_env.version)
print(f"   ✔ Entorno de inferencia '{inference_env_asset.name}' v{inference_env_asset.version} creado/actualizado.")

# Deployment creation
# model_path = f"azureml:{registered_model_name_for_deployment}:latest" # This line is now redundant as model_path is set above

demo_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model_path,
    environment=f"azureml:{inference_env_asset.name}:{inference_env_asset.version}", # Usar la versión exacta
    code_configuration=CodeConfiguration(
        code=os.path.join(os.path.dirname(__file__), "scripts"),
        scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1
)

ml_client.begin_create_or_update(demo_deployment).wait()
print(f"   ✔ Despliegue '{demo_deployment.name}' en el endpoint '{endpoint.name}' creado/actualizado.")

# Set all traffic to the new deployment
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).wait()
print("   ✔ Tráfico del endpoint configurado para el despliegue 'blue'.")

print("\n¡Pipeline ejecutado en Azure ML y modelo desplegado como servicio en línea con éxito!")
print(f"   🔗 URL del endpoint: {endpoint.scoring_uri}")
