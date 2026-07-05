import os


def configure_offline_quiet_env() -> None:
    """Must be called before importing anything that touches
    huggingface_hub/transformers (revdict.search, or anything that imports
    revdict.models.*) -- those libraries snapshot these env vars into
    module-level constants the moment they're first imported, so setting
    them any later than that has no effect."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
