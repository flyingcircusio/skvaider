from skvaider.proxy.backends import ModelConfig


def test_model_config():
    cfg = ModelConfig({})
    assert cfg.get("gemma3b") == {}

    cfg.config["gemma3b"] = {"num_ctx": 3072}
    assert cfg.get("gemma3b") == {"num_ctx": 3072}

    assert cfg.get("gemma3b:latest") == {"num_ctx": 3072}
    cfg.config["gemma3b"] = {"num_ctx": 2048}
    assert cfg.get("gemma3b:latest") == {"num_ctx": 2048}

    cfg.config["__default__"] = {"num_ctx": 1024}
    assert cfg.get("unknown") == {"num_ctx": 1024}
