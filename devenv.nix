{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs; [
    postgresql.lib
    ollama
  ];

  enterTest = ''
    wait_for_port 11435
    wait_for_port 5432
    # Wait for model (defined in processes:ollama) to be available
    until ${lib.getExe pkgs.curl} -s http://127.0.0.1:11435/v1/models | ${lib.getExe pkgs.yq-go} -e ".data"; do sleep 0.5; done
    bootstrap-db
    run-tests
  '';

  env = {
    OLLAMA_HOST = "127.0.0.1:11435";
    PYTHONUNBUFFERED = "1"; # makes output from subprocesses in tests more reliably visible
  };

  scripts.bootstrap-db.exec = ''
    dropdb skvaider || true
    createdb -O skvaider skvaider
    psql -d skvaider -p 5432 -U skvaider < migrations/0001_init.sql
    dropdb test || true
    createdb -O skvaider test
    psql -d test -p 5432 -U skvaider < migrations/0001_init.sql
  '';

  scripts.run-tests.exec = ''
    uv run pytest -vv "$@"
  '';

  processes = {
    skvaider.exec = "uv run uvicorn skvaider:app_factory --reload-include 'config.toml' --factory --reload ";
    ollama = {
      exec = ''
        ollama serve&
        timeout 15 bash -c "until ${lib.getExe pkgs.curl} http://localhost:11435 -s; do sleep 0.5; done"
        ollama pull gemma3:1b
        ollama pull nomic-embed-text:v1.5
        ollama list
        wait
      '';
    };
  };

  services.postgres = {
    enable = true;
    listen_addresses = "localhost";
    initialDatabases = [
      {
        name = "skvaider";
        user = "skvaider";
        pass = "foobar";
      }
    ];
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
