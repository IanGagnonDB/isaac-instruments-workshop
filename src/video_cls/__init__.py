"""CPU video frame classification on Spark: shared by notebooks, jobs and tests."""

__version__ = "0.1.0"

# ImageNet-1k has no "stop sign" class. 919 is "street sign": a labelled proxy only.
STREET_SIGN_CLASS = 919
