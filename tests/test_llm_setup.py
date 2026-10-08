"""How the language-model client is found and configured.

None of these reach a provider: the sister client is replaced with a stand-in
that reports whichever provider the test needs.
"""

from __future__ import annotations

import pytest

from autoopensn import kp_bridge
from autoopensn.llm import LLMUnavailable, ProviderLLM


class StandIn:
    def __init__(self, provider):
        self.provider = provider
        self.model = "whatever"


@pytest.fixture
def sister_present(monkeypatch):
    monkeypatch.setattr(kp_bridge, "available", lambda: True)
    monkeypatch.setattr(kp_bridge, "llm_models", lambda: {"fast": "f", "thorough": "t"})


def test_a_missing_key_is_an_error_not_a_quiet_switch_to_ollama(monkeypatch, sister_present):
    """The client settles on a local Ollama when it has no key, without a word.
    This project once reported a day of results from a local llama3.2 while
    believing the TAMU key was in use."""
    monkeypatch.delenv("AUTOOPENSN_ALLOW_OLLAMA", raising=False)
    monkeypatch.setattr(kp_bridge, "llm_client", lambda quiet=True: StandIn("ollama"))
    with pytest.raises(LLMUnavailable, match="no API key was found"):
        ProviderLLM.from_env()


def test_the_error_says_where_the_key_goes(monkeypatch, sister_present):
    monkeypatch.delenv("AUTOOPENSN_ALLOW_OLLAMA", raising=False)
    monkeypatch.setattr(kp_bridge, "llm_client", lambda quiet=True: StandIn("ollama"))
    with pytest.raises(LLMUnavailable) as excinfo:
        ProviderLLM.from_env()
    assert str(kp_bridge.ENV_FILE) in str(excinfo.value)
    assert "TAMU_API_KEY" in str(excinfo.value)


def test_ollama_is_used_when_asked_for(monkeypatch, sister_present):
    monkeypatch.setenv("AUTOOPENSN_ALLOW_OLLAMA", "1")
    monkeypatch.setattr(kp_bridge, "llm_client", lambda quiet=True: StandIn("ollama"))
    assert ProviderLLM.from_env().client.provider == "ollama"


def test_a_real_provider_is_unaffected(monkeypatch, sister_present):
    monkeypatch.delenv("AUTOOPENSN_ALLOW_OLLAMA", raising=False)
    monkeypatch.setattr(kp_bridge, "llm_client", lambda quiet=True: StandIn("tamu"))
    assert ProviderLLM.from_env().client.provider == "tamu"


def test_the_env_file_is_read_by_its_own_path_not_the_working_directory(tmp_path, monkeypatch):
    """Started from anywhere but the repository, a cwd-relative lookup found no
    key at all."""
    pytest.importorskip("dotenv")
    env = tmp_path / ".env"
    env.write_text("AUTOOPENSN_TEST_ONLY_KEY=from-the-file\n")
    monkeypatch.setattr(kp_bridge, "ENV_FILE", env)
    monkeypatch.delenv("AUTOOPENSN_TEST_ONLY_KEY", raising=False)
    monkeypatch.chdir(tmp_path.parent)  # deliberately somewhere else
    assert kp_bridge.load_own_env() == env
    import os
    assert os.environ["AUTOOPENSN_TEST_ONLY_KEY"] == "from-the-file"
    monkeypatch.delenv("AUTOOPENSN_TEST_ONLY_KEY")


def test_a_shell_export_still_wins_over_the_file(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    env = tmp_path / ".env"
    env.write_text("AUTOOPENSN_TEST_ONLY_KEY=from-the-file\n")
    monkeypatch.setattr(kp_bridge, "ENV_FILE", env)
    monkeypatch.setenv("AUTOOPENSN_TEST_ONLY_KEY", "from-the-shell")
    kp_bridge.load_own_env()
    import os
    assert os.environ["AUTOOPENSN_TEST_ONLY_KEY"] == "from-the-shell"


def test_the_llm_extra_lists_what_the_client_imports():
    """And not the sister repository's whole requirements file, which pulls in
    PyTorch through sentence-transformers."""
    import tomllib
    from pathlib import Path

    extras = tomllib.loads(Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text())
    llm = " ".join(extras["project"]["optional-dependencies"]["llm"])
    for package in ("python-dotenv", "tamu-chat", "openai", "groq"):
        assert package in llm
    assert "sentence-transformers" not in llm and "torch" not in llm
