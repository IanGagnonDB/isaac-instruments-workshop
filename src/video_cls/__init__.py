"""CPU video frame classification on Spark: shared by notebooks, jobs and tests."""

__version__ = "0.1.0"

# ImageNet-1k has no "stop sign" class, so scoring uses a labelled proxy class.
# 919 "street sign" is the closest, but it is almost never top-1 on dashcam frames;
# 920 "traffic light" is the default proxy because it fires on ~15% of D2-City frames.
STREET_SIGN_CLASS = 919
TRAFFIC_LIGHT_CLASS = 920
