"""Tests for CLI entry point."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from gnn_playground.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestCLI:
    def test_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout).lower()
        assert "run" in out or "options" in out

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "--task" in out
        assert "--dataset" in out
        assert "--model" in out
        assert "--config" in out

    def test_run_no_args_shows_help_or_error(self):
        result = runner.invoke(app, ["run"])
        # Typer may show usage error (exit code 1) or our custom error
        assert result.exit_code != 0

    def test_run_unknown_task_errors(self):
        result = runner.invoke(app, ["run", "--task", "nonexistent_task"])
        assert result.exit_code != 0
        out = _strip_ansi(result.stdout)
        assert "Unknown task" in out or "Error" in out

    def test_run_with_valid_config_unknown_task(self, tmp_path):
        import yaml

        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"task": "fake_task", "dataset": "cora"}))
        result = runner.invoke(app, ["run", "--config", str(config_file)])
        # Should error on unknown task, not on config loading
        assert result.exit_code != 0
        out = _strip_ansi(result.stdout)
        assert "Unknown task" in out or "Error" in out

    def test_run_with_task_flag(self):
        result = runner.invoke(app, ["run", "--task", "nonexistent"])
        assert result.exit_code != 0

    def test_seed_and_epochs_options(self):
        result = runner.invoke(app, ["run", "--help"])
        out = _strip_ansi(result.stdout)
        assert "--seed" in out
        assert "--epochs" in out
        assert "--lr" in out
