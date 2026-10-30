"""CPU video frame classification on Spark: shared by notebooks, jobs and tests."""

__version__ = "0.1.0"

# ImageNet-1k has no "stop sign" class, so scoring uses a labelled proxy class.
# 919 "street sign" is the closest, but it is almost never top-1 on dashcam frames;
# 920 "traffic light" is the default proxy because it fires on ~15% of D2-City frames.
STREET_SIGN_CLASS = 919
TRAFFIC_LIGHT_CLASS = 920

PLACEHOLDER = "CHANGE_ME"


def check_location(catalog, schema):
    """Fail fast if the bundle variables were not edited before deploying."""
    for name, value in (("catalog", catalog), ("schema", schema)):
        if not value or PLACEHOLDER in value:
            raise ValueError(f"Set `{name}` under `variables:` in databricks.yml, redeploy the bundle and "
                             f"rerun (got {value!r}).")
