from kfp import dsl


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"],
)
def register_model(
    project: str,
    location: str,
    model_display_name: str,
    container_uri: str,
    bucket_uri: str,
    target_store_nbr: int,
    target_family: str,
    model_artifact: dsl.Input[dsl.Model],
):
    """Register the trained model in Vertex AI Model Registry.

    No proportions needed — the model predicts directly for a single store×family.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)

    print(f"Registering model from artifact URI: {model_artifact.uri}")
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_artifact.uri,
        serving_container_image_uri=container_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_environment_variables={
            "BUCKET_URI": bucket_uri,
            "TARGET_STORE_NBR": str(target_store_nbr),
            "TARGET_FAMILY": target_family,
        },
        description="Seasonal ARIMAX model for single store×family combination",
    )
    print(f"Successfully registered model! Version: {model.version_id}")
