import pytest

from skvaider.config import ModelInstanceConfig, parse_size


@pytest.mark.parametrize(
    "value,expected",
    [
        (42, 42),
        ("1K", 1024),
        ("2M", 2 * 1024**2),
        ("4G", 4 * 1024**3),
        ("1T", 1024**4),
        ("8GB", 8 * 1024**3),
        ("1.5G", int(1.5 * 1024**3)),
        ("512", 512),
        ("  4G  ", 4 * 1024**3),
    ],
)
def test_parse_size_valid(value: str | int, expected: int) -> None:
    assert parse_size(value) == expected


def test_parse_size_invalid() -> None:
    with pytest.raises(ValueError):
        parse_size("not-a-size")


def test_model_instance_config_parses_string_memory() -> None:
    cfg = ModelInstanceConfig(
        id="m", instances=1, memory={"ram": parse_size("2G")}, task="chat"
    )
    assert cfg.memory["ram"] == 2 * 1024**3


def test_model_instance_config_total_size() -> None:
    cfg = ModelInstanceConfig(
        id="m",
        instances=1,
        memory={"ram": parse_size("2G"), "vram": parse_size("1G")},
        task="chat",
    )
    assert cfg.total_size() == 3 * 1024**3
