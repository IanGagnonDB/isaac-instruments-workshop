# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy the SAM3 V2 endpoint
# MAGIC
# MAGIC Run after registering `model_sam3_frames_v2.py`. By default scale-to-zero is off so
# MAGIC the GPU replica stays warm for interactive latency. Set `scale_to_zero=true` to save
# MAGIC cost when the app is idle (the first request then waits for a cold start).
# MAGIC
# MAGIC When `app_name` is set, the app's service principal is granted CAN_QUERY on the
# MAGIC endpoint. The endpoint is created here rather than by the bundle, so the app can
# MAGIC be deployed before the endpoint exists.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "cv")
dbutils.widgets.text("model_name", "transformers-sam3-frames-v2")
dbutils.widgets.text("alias", "performance-v2")
dbutils.widgets.text("endpoint_name", "sam3-video-frames-v2")
dbutils.widgets.text("scale_to_zero", "false")
dbutils.widgets.text("app_name", "")
dbutils.widgets.text("gpu_workload_type", "GPU_MEDIUM")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
model_alias = dbutils.widgets.get("alias")
endpoint_name = dbutils.widgets.get("endpoint_name")
app_name = dbutils.widgets.get("app_name").strip()
if not catalog:
    raise ValueError("Set the catalog parameter (the bundle passes it).")
gpu_workload_type = dbutils.widgets.get("gpu_workload_type").strip()
scale_to_zero = dbutils.widgets.get("scale_to_zero").lower() in {"1", "true", "yes"}
registered_model_name = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
    ServedModelInputWorkloadType,
)
from mlflow import MlflowClient

mlflow_client = MlflowClient(registry_uri="databricks-uc")
versions = mlflow_client.search_model_versions(f"name='{registered_model_name}'")
if not versions:
    raise RuntimeError(f"No registered versions found for {registered_model_name}.")
version = max(versions, key=lambda item: int(item.version))
mlflow_client.set_registered_model_alias(
    registered_model_name,
    model_alias,
    version.version,
)
served_models = [
    ServedModelInput(
        model_name=registered_model_name,
        model_version=version.version,
        workload_size=ServedModelInputWorkloadSize.SMALL,
        workload_type=ServedModelInputWorkloadType(gpu_workload_type),
        scale_to_zero_enabled=scale_to_zero,
    )
]

workspace = WorkspaceClient()
try:
    workspace.serving_endpoints.get(endpoint_name)
except NotFound:
    workspace.serving_endpoints.create_and_wait(
        endpoint_name,
        config=EndpointCoreConfigInput(served_models=served_models),
        timeout=datetime.timedelta(minutes=60),
    )
    action = "created"
else:
    workspace.serving_endpoints.update_config_and_wait(
        endpoint_name,
        served_models=served_models,
        timeout=datetime.timedelta(minutes=60),
    )
    action = "updated"

print(
    f"{action.capitalize()} {endpoint_name} with "
    f"{registered_model_name} version {version.version}; scale-to-zero={scale_to_zero}."
)

# COMMAND ----------

from databricks.sdk.service.serving import (
    ServingEndpointAccessControlRequest,
    ServingEndpointPermissionLevel,
)

if app_name:
    app = workspace.apps.get(app_name)
    endpoint_id = workspace.serving_endpoints.get(endpoint_name).id
    workspace.serving_endpoints.update_permissions(
        endpoint_id,
        access_control_list=[
            ServingEndpointAccessControlRequest(
                service_principal_name=app.service_principal_client_id,
                permission_level=ServingEndpointPermissionLevel.CAN_QUERY,
            )
        ],
    )
    print(f"Granted CAN_QUERY on {endpoint_name} to app {app_name}.")
else:
    print("app_name not set; skipped the app permission grant.")
