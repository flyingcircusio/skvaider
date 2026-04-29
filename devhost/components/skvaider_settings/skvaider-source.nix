{ pkgs, ... }:
let
  skvaiderSrc = builtins.path {
    path = /srv/s-dev/deployment;
    name = "skvaider-src";
    filter =
      path: _type:
      !(builtins.elem (baseNameOf path) [
        ".appenv"
        ".venv"
        "devhost"
        "insecure-private.key"
      ]);
  };
in
{
  nixpkgs.overlays = [
    (final: prev: {
      fc = prev.fc // {
        skvaider = final.callPackage skvaiderSrc {
          src = skvaiderSrc;
        };
      };
    })
  ];
}
