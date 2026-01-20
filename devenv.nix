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
      uv run gunicorn "skvaider:app_factory()" -k uvicorn_worker.UvicornWorker --reload-extra-file config.toml
    '';
    skvaider-inference.exec = ''
      uv run gunicorn "skvaider.inference:app_factory()" -k uvicorn_worker.UvicornWorker --reload-extra-file config-inference.toml
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
