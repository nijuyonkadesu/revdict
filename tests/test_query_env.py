import os

from revdict.query_env import configure_offline_quiet_env


def test_configure_offline_quiet_env_sets_all_three_vars_without_overwriting_existing(
    monkeypatch,
):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "debug")

    configure_offline_quiet_env()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    # Pre-existing value is respected, not clobbered (setdefault semantics).
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "debug"
