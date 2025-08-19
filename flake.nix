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
    perSystem = { config, pkgs, inputs', self', system, ... }: {
      devenv.shells.default = {
        packages = with pkgs; [
          postgresql.lib
        ];
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
