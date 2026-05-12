from kfp import dsl


@dsl.component(
    base_image="python:3.13",
    packages_to_install=["google-cloud-aiplatform", "fsspec", "gcsfs"],
)
def register_model(
    project: str,
    location: str,
    model_display_name: str,
    container_uri: str,
    experiment_name: str,
    model_artifact: dsl.Input[dsl.Model],
    proportions: dsl.Input[dsl.Dataset],
):
    """Register the trained model in Vertex AI Model Registry.

    Bundles the proportions.csv into the model artifact directory before
    uploading, so the CPR container can access it at serving time.
    Logs registration metadata to Vertex AI Experiments for lineage tracking.
    """
    import shutil
    import os
    from datetime import datetime
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location, experiment=experiment_name)

    # Bundle proportions.csv into the model artifact directory
    # so the CPR container can load it alongside model.pkl
    dest_proportions = os.path.join(model_artifact.path, "proportions.csv")
    shutil.copy2(proportions.path, dest_proportions)
    print(f"Bundled proportions.csv into model artifact directory")

    print(f"Registering model from artifact URI: {model_artifact.uri}")
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_artifact.uri,
        serving_container_image_uri=container_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        description="Seasonal ARIMAX Cashflow Model with CPR and proportional disaggregation",
    )
    print(f"Successfully registered model! Version: {model.version_id}")

    # Log model registration metadata to Vertex AI Experiments
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    with aiplatform.start_run(f"model-registration-{ts}") as run:
        run.log_params({
            "model_display_name": model_display_name,
            "model_version": model.version_id,
            "container_uri": container_uri,
            "artifact_uri": model_artifact.uri,
            "stage": "registration",
        })
