import sys
import tomllib
from pathlib import Path

from skvaider.config import Config as SkvaiderConfig
from skvaider.inference.config import Config as InferenceConfig


def check_skvaider_config() -> None:
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.toml"
    path = Path(config_file)
    if not path.exists():
        print(f"Configuration file {config_file} not found.")
        sys.exit(1)

    try:
        with path.open("rb") as f:
            config_data = tomllib.load(f)
        SkvaiderConfig.model_validate(config_data)
        print(f"Configuration file {config_file} is valid.")
    except Exception as e:
        print(
            f"Configuration file {config_file} is invalid: {e}", file=sys.stderr
        )
        sys.exit(1)


def check_inference_config() -> None:
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config-inference.toml"
    path = Path(config_file)
    if not path.exists():
        print(f"Inference configuration file {config_file} not found.")
        sys.exit(1)

    try:
        with path.open("rb") as f:
            config_data = tomllib.load(f)
        InferenceConfig.model_validate(config_data)
        print(f"Inference configuration file {config_file} is valid.")
    except Exception as e:
        print(
            f"Inference configuration file {config_file} is invalid: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
