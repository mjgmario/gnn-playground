"""Tests for CLI entry point."""

from __future__ import annotations

from typer.testing import CliRunner

from gnn_playground.cli import app

runner = CliRunner(env={"NO_COLOR": "1"})


class TestCLI:
    def test_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout.lower() or "options" in result.stdout.lower()

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--task" in result.stdout
        assert "--dataset" in result.stdout
        assert "--model" in result.stdout
        assert "--config" in result.stdout

    def test_run_no_args_shows_help_or_error(self):
        result = runner.invoke(app, ["run"])
        # Typer may show usage error (exit code 1) or our custom error
        assert result.exit_code != 0

    def test_run_unknown_task_errors(self):
        result = runner.invoke(app, ["run", "--task", "nonexistent_task"])
        assert result.exit_code != 0
        assert "Unknown task" in result.stdout or "Error" in result.stdout

    def test_run_with_valid_config_unknown_task(self, tmp_path):
        import yaml

        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"task": "fake_task", "dataset": "cora"}))
        result = runner.invoke(app, ["run", "--config", str(config_file)])
        # Should error on unknown task, not on config loading
        assert result.exit_code != 0
        assert "Unknown task" in result.stdout or "Error" in result.stdout

    def test_run_with_task_flag(self):
        result = runner.invoke(app, ["run", "--task", "nonexistent"])
        assert result.exit_code != 0

    def test_seed_and_epochs_options(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "--seed" in result.stdout
        assert "--epochs" in result.stdout
        assert "--lr" in result.stdout
