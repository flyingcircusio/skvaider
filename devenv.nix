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
      uv run skvaider-proxy -c config.toml
    '';
    skvaider-inference-1.exec = ''
      uv run skvaider-inference -c config-inference-1.toml
    '';
    skvaider-inference-2.exec = ''
      uv run skvaider-inference -c config-inference-2.toml
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
