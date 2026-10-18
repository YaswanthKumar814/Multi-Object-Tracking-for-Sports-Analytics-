from pathlib import Path

import pytest
import yaml

from soccer_tracker.config import AppConfig, config_from_dict, config_to_dict, load_config
from soccer_tracker.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def test_builtin_defaults_are_valid():
    cfg = load_config(None)
    assert cfg.players.detector.classes == ["person"]
    assert cfg.ball.detector.classes == ["sports ball"]


def test_default_yaml_matches_builtin_defaults():
    # configs/default.yaml is documentation; it must not drift from the code.
    assert config_to_dict(load_config(ROOT / "configs" / "default.yaml")) == config_to_dict(AppConfig())


def test_example_finetuned_config_loads():
    cfg = load_config(ROOT / "configs" / "soccer_finetuned.example.yaml")
    assert cfg.ball.detector.classes == ["ball"]
    assert cfg.ball.detector.tiling.enabled is True
    # unspecified keys keep their defaults
    assert cfg.ball.tracker.gate_px == AppConfig().ball.tracker.gate_px


def test_partial_override_keeps_other_defaults():
    cfg = config_from_dict({"ball": {"tracker": {"gate_px": 50}}})
    assert cfg.ball.tracker.gate_px == 50.0
    assert cfg.ball.tracker.max_missed_frames == AppConfig().ball.tracker.max_missed_frames


@pytest.mark.parametrize(
    "data, fragment",
    [
        ({"players": {"detector": {"confidance": 0.3}}}, "Unknown key"),
        ({"players": {"detector": {"confidence": "high"}}}, "must be a number"),
        ({"players": {"detector": {"confidence": 1.5}}}, "confidence must be in"),
        ({"players": {"detector": {"imgsz": 1000}}}, "multiple of 32"),
        ({"players": {"detector": {"classes": []}}}, "at least one class"),
        ({"io": {"fourcc": "h264x"}}, "4 characters"),
        ({"io": {"max_frames": 0}}, "max_frames"),
        ({"ball": {"tracker": {"min_hits": 0}}}, "min_hits"),
        ({"ball": {"enabled": "yes"}}, "true/false"),
        ({"players": "nope"}, "must be a mapping"),
    ],
)
def test_invalid_values_raise_config_error(data, fragment):
    with pytest.raises(ConfigError, match=fragment):
        config_from_dict(data)


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("players: [unclosed\n")
    with pytest.raises(ConfigError, match="parse YAML"):
        load_config(p)


def test_empty_yaml_gives_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert config_to_dict(load_config(p)) == config_to_dict(AppConfig())


def test_roundtrip_dict(tmp_path):
    p = tmp_path / "rt.yaml"
    p.write_text(yaml.safe_dump(config_to_dict(AppConfig())))
    assert config_to_dict(load_config(p)) == config_to_dict(AppConfig())
