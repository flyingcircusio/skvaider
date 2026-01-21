{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  packages = with pkgs; [
    llama-cpp
  ];

  enterTest = ''
    run-tests
  '';

  env = {
    PYTHONUNBUFFERED = "1"; # makes output from subprocesses in tests more reliably visible
  };

  scripts.run-tests.exec = ''
    uv run pytest -vv "$@"
  '';

  processes = {
    skvaider.exec = ''
      uv run gunicorn "skvaider:app_factory()" -k uvicorn_worker.UvicornWorker -b 127.0.0.1:8000 --reload-extra-file config.toml
    '';
    skvaider-inference-1.exec = ''
      export SKVAIDER_CONFIG_FILE=config-inference-1.toml
      uv run gunicorn "skvaider.inference:app_factory()" -k uvicorn_worker.UvicornWorker -b 127.0.0.1:8001 --reload-extra-file $SKVAIDER_CONFIG_FILE
    '';
    skvaider-inference-2.exec = ''
      export SKVAIDER_CONFIG_FILE=config-inference-2.toml
      uv run gunicorn "skvaider.inference:app_factory()" -k uvicorn_worker.UvicornWorker -b 127.0.0.1:8002 --reload-extra-file $SKVAIDER_CONFIG_FILE

    '';
  };

  languages.python = {
    enable = true;
    package = pkgs.python312;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };
}
