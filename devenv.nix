{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs; [
    ollama
  ];

  enterTest = ''
    wait_for_port 11435
    # Wait for model (defined in processes:ollama) to be available
    until ${lib.getExe pkgs.curl} -s http://127.0.0.1:11435/v1/models | ${lib.getExe pkgs.yq-go} -e ".data"; do sleep 0.5; done
    run-tests
  '';

  env = {
    OLLAMA_HOST = "127.0.0.1:11435";
    PYTHONUNBUFFERED = "1"; # makes output from subprocesses in tests more reliably visible
  };

  scripts.run-tests.exec = ''
    uv run pytest -vv "$@"
  '';

  processes = {
    skvaider.exec = ''
      uv run gunicorn "skvaider:app_factory()" -k uvicorn_worker.UvicornWorker --reload-extra-file config.toml
    '';
    ollama1 = {
      exec = ''
        export OLLAMA_HOST="127.0.0.1:11435"
        export OLLAMA_DEBUG="1"
        export OLLAMA_NUM_PARALLEL="10"
        export OLLAMA_FLASH_ATTENTION="1"
        export OLLAMA_SCHED_SPREAD="0"
        export OLLAMA_MULTIUSER_CACHE="1"
        export OLLAMA_NEW_ENGINE="1"
        export OLLAMA_NEW_ESTIMATES="1"
        export OLLAMA_KEEP_ALIVE="-1"

        ollama serve&
        timeout 15 bash -c "until ${lib.getExe pkgs.curl} http://localhost:11435 -s; do sleep 0.5; done"
        ollama pull gemma3:1b
        ollama pull nomic-embed-text:v1.5
        ollama list
        wait
      '';
    };
    ollama2 = {
      exec = ''
        export OLLAMA_HOST="127.0.0.1:11436";
        export OLLAMA_DEBUG="1"
        export OLLAMA_NUM_PARALLEL="10"
        export OLLAMA_FLASH_ATTENTION="1"
        export OLLAMA_SCHED_SPREAD="0"
        export OLLAMA_MULTIUSER_CACHE="1"
        export OLLAMA_NEW_ENGINE="1"
        export OLLAMA_NEW_ESTIMATES="1"
        export OLLAMA_KEEP_ALIVE="-1"

        ollama serve&
        timeout 15 bash -c "until ${lib.getExe pkgs.curl} http://localhost:11436 -s; do sleep 0.5; done"
        ollama pull gemma3:1b
        ollama pull nomic-embed-text:v1.5
        ollama list
        wait
      '';
    };
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
