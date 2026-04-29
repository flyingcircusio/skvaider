{
  lib,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  python312,
  callPackage,
  src ? ./.,
  ...
}:

let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = src; };

  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  pythonSet =
    (callPackage pyproject-nix.build.packages {
      python = python312;
    }).overrideScope
      (
        lib.composeManyExtensions [
          pyproject-build-systems.default
          overlay
        ]
      );
in
(pythonSet.mkVirtualEnv "skvaider-env" workspace.deps.default).overrideAttrs (_old: {
  venvIgnoreCollisions = [
    "*"
  ];
  passthru.src = src;
  passthru.testEnv =
    (pythonSet.mkVirtualEnv "skvaider-test-env" workspace.deps.all).overrideAttrs
      (_: {
        venvIgnoreCollisions = [ "*" ];
      });
})
