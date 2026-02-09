"""Tests for configuration loading and merging."""

from __future__ import annotations

import pytest
import yaml

from gnn_playground.config import BaseConfig, build_config, load_config, merge_config


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"task": "node_classification", "seed": 123}))
        result = load_config(str(config_file))
        assert result["task"] == "node_classification"
        assert result["seed"] == 123

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("nonexistent.yaml")

    def test_load_empty_file_raises(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        with pytest.raises(ValueError, match="Config file is empty"):
            load_config(str(config_file))

    def test_load_non_dict_raises(self, tmp_path):
        config_file = tmp_path / "list.yaml"
        config_file.write_text(yaml.dump([1, 2, 3]))
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            load_config(str(config_file))


class TestMergeConfig:
    def test_override_scalar(self):
        base = {"seed": 42, "lr": 0.01}
        overrides = {"seed": 123}
        result = merge_config(base, overrides)
        assert result["seed"] == 123
        assert result["lr"] == 0.01

    def test_deep_merge_dicts(self):
        base = {"nested": {"a": 1, "b": 2}}
        overrides = {"nested": {"b": 3, "c": 4}}
        result = merge_config(base, overrides)
        assert result["nested"] == {"a": 1, "b": 3, "c": 4}

    def test_none_values_ignored(self):
        base = {"seed": 42, "lr": 0.01}
        overrides = {"seed": None, "lr": 0.001}
        result = merge_config(base, overrides)
        assert result["seed"] == 42
        assert result["lr"] == 0.001

    def test_add_new_keys(self):
        base = {"a": 1}
        overrides = {"b": 2}
        result = merge_config(base, overrides)
        assert result == {"a": 1, "b": 2}


class TestBuildConfig:
    def test_defaults_applied(self):
        cfg = build_config()
        assert cfg["seed"] == 42
        assert cfg["epochs"] == 200
        assert cfg["lr"] == 0.01

    def test_yaml_overrides_defaults(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"seed": 999, "epochs": 50}))
        cfg = build_config(config_path=str(config_file))
        assert cfg["seed"] == 999
        assert cfg["epochs"] == 50
        assert cfg["lr"] == 0.01  # default preserved

    def test_cli_overrides_yaml(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"seed": 999, "epochs": 50}))
        cfg = build_config(config_path=str(config_file), seed=1, epochs=10)
        assert cfg["seed"] == 1
        assert cfg["epochs"] == 10

    def test_cli_none_does_not_override(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"seed": 999}))
        cfg = build_config(config_path=str(config_file), seed=None)
        assert cfg["seed"] == 999


class TestBaseConfig:
    def test_default_values(self):
        cfg = BaseConfig()
        assert cfg.seed == 42
        assert cfg.epochs == 200
        assert cfg.lr == 0.01
        assert cfg.hidden_dim == 64
        assert cfg.device == "auto"
        assert cfg.models == []

    def test_custom_values(self):
        cfg = BaseConfig(seed=123, epochs=50, models=["gcn", "gat"])
        assert cfg.seed == 123
        assert cfg.epochs == 50
        assert cfg.models == ["gcn", "gat"]
