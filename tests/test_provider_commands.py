"""Tests for nanobot provider CLI subcommands."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.config.schema import Config

runner = CliRunner()


@pytest.fixture
def tmp_config(tmp_path):
    """Provide a real but isolated config file and load/save functions."""
    config_path = tmp_path / "config.json"

    def _save(config: Config, path=None):
        from nanobot.config.loader import convert_to_camel
        data = convert_to_camel(config.model_dump())
        (path or config_path).write_text(json.dumps(data, indent=2))

    def _load(path=None):
        from nanobot.config.loader import convert_keys
        raw = json.loads((path or config_path).read_text())
        return Config.model_validate(convert_keys(raw))

    # Write a default config to disk
    _save(Config())

    with (
        patch("nanobot.config.loader.get_config_path", return_value=config_path),
        patch("nanobot.config.loader.load_config", side_effect=lambda p=None: _load(p)),
        patch("nanobot.config.loader.save_config", side_effect=lambda c, p=None: _save(c, p)),
    ):
        yield config_path, _load, _save


# ---------------------------------------------------------------------------
# provider list
# ---------------------------------------------------------------------------


def test_provider_list_shows_table(tmp_config):
    """provider list should print a table with provider names."""
    result = runner.invoke(app, ["provider", "list"])
    assert result.exit_code == 0
    # Key provider names should appear
    assert "OpenRouter" in result.stdout
    assert "Anthropic" in result.stdout
    assert "Active model" in result.stdout


def test_provider_list_marks_active_provider(tmp_config):
    """★ marker should appear next to the active provider when a key is set."""
    config_path, _load, _save = tmp_config
    config = _load()
    config.providers.openrouter.api_key = "sk-or-test12345678"
    config.agents.defaults.model = "openrouter/claude-3-opus"
    _save(config)

    result = runner.invoke(app, ["provider", "list"])
    assert result.exit_code == 0
    assert "★" in result.stdout


# ---------------------------------------------------------------------------
# provider set-key
# ---------------------------------------------------------------------------


def test_provider_set_key_updates_config(tmp_config):
    """set-key should write the API key to config and confirm."""
    config_path, _load, _save = tmp_config
    result = runner.invoke(app, ["provider", "set-key", "openrouter", "sk-or-abc123xyz"])
    assert result.exit_code == 0
    assert "✓" in result.stdout
    assert "OpenRouter" in result.stdout
    # Verify key was saved
    config = _load()
    assert config.providers.openrouter.api_key == "sk-or-abc123xyz"


def test_provider_set_key_anthropic(tmp_config):
    """set-key should work for direct providers like anthropic."""
    config_path, _load, _save = tmp_config
    result = runner.invoke(app, ["provider", "set-key", "anthropic", "sk-ant-test"])
    assert result.exit_code == 0
    config = _load()
    assert config.providers.anthropic.api_key == "sk-ant-test"


def test_provider_set_key_unknown_provider(tmp_config):
    """set-key should fail gracefully for unknown provider names."""
    result = runner.invoke(app, ["provider", "set-key", "nonexistent", "somekey"])
    assert result.exit_code != 0
    assert "Unknown provider" in result.stdout


def test_provider_set_key_oauth_provider(tmp_config):
    """set-key should reject OAuth providers with a helpful message."""
    result = runner.invoke(app, ["provider", "set-key", "openai_codex", "somekey"])
    assert result.exit_code != 0
    assert "OAuth" in result.stdout


def test_provider_set_key_local_provider(tmp_config):
    """set-key should reject local providers (vllm) with a helpful message."""
    result = runner.invoke(app, ["provider", "set-key", "vllm", "somekey"])
    assert result.exit_code != 0
    assert "api_base" in result.stdout


# ---------------------------------------------------------------------------
# provider set-model
# ---------------------------------------------------------------------------


def test_provider_set_model_updates_config(tmp_config):
    """set-model should write the model to config and confirm."""
    config_path, _load, _save = tmp_config
    result = runner.invoke(app, ["provider", "set-model", "gpt-4o"])
    assert result.exit_code == 0
    assert "✓" in result.stdout
    assert "gpt-4o" in result.stdout
    config = _load()
    assert config.agents.defaults.model == "gpt-4o"


def test_provider_set_model_warns_no_provider(tmp_config):
    """set-model should warn when no configured provider can serve the model."""
    result = runner.invoke(app, ["provider", "set-model", "unknown-model-xyz"])
    assert result.exit_code == 0
    assert "Warning" in result.stdout or "no configured provider" in result.stdout


def test_provider_set_model_shows_provider_when_key_set(tmp_config):
    """set-model should show which provider will be used when a key is set."""
    config_path, _load, _save = tmp_config
    config = _load()
    config.providers.anthropic.api_key = "sk-ant-test"
    _save(config)

    result = runner.invoke(app, ["provider", "set-model", "claude-opus-4-5"])
    assert result.exit_code == 0
    assert "Anthropic" in result.stdout


# ---------------------------------------------------------------------------
# status command (enhanced)
# ---------------------------------------------------------------------------


def test_status_shows_providers_and_channels(tmp_config):
    """status command should display provider and channel tables."""
    config_path, _load, _save = tmp_config
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Providers" in result.stdout
    assert "Channels" in result.stdout
    assert "Model" in result.stdout


def test_status_shows_active_model(tmp_config):
    """status should show the configured default model."""
    config_path, _load, _save = tmp_config
    config = _load()
    config.agents.defaults.model = "deepseek-chat"
    _save(config)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "deepseek-chat" in result.stdout
