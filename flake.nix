{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.devenv.url = "github:cachix/devenv";
  inputs.flake-parts.url = "github:hercules-ci/flake-parts";

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = inputs: inputs.flake-parts.lib.mkFlake { inherit inputs; } {
    imports = [
      inputs.devenv.flakeModule
    ];
    systems = inputs.nixpkgs.lib.systems.flakeExposed;
    perSystem = {  pkgs, ... }: {

      devenv.shells.default = {
        packages = with pkgs; [
          postgresql.lib
          ollama
        ];

        env = {
          OLLAMA_HOST = "127.0.0.1:11435";
        };

        scripts.bootstrap-db.exec = ''
          psql -d skvaider -p 5432 -U skvaider < migrations/0001_init.sql
        '';

        scripts.run-tests.exec = ''
          uv run pytest -vv
        '';

        processes = {
          skvaider.exec = "uv run uvicorn skvaider:app_factory --reload-include '*.toml' --factory --reload ";
          ollama.exec = ''
            ollama serve&
            ollama pull gemma3:1b
            ollama list
            wait
          '';
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
      };
    };
  };
}
