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
  skvaider = pkgs.callPackage skvaiderSrc {
    src = skvaiderSrc;
  };
  configFile = (pkgs.formats.toml { }).generate "skvaider-gateway.toml" {
    auth.admin_tokens = [ "{{component.token}}" ];
    debug.slow_threshold = 5;
    server = {
      host = "127.0.0.1";
      port = 23211;
      directory = "/var/lib/skvaider";
    };
    backend = [
      {
        type = "skvaider";
        url = "{{component.inference_url}}";
      }
    ];
    models = [
      {
        id = "tiny-gpt2";
        instances = 1;
        memory.ram = 1;
        task = "chat";
      }
    ];
    logging = {
      log_level = "DEBUG";
      log_dir = "/var/log/skvaider";
    };
  };
in
{
  users.users.skvaider = {
    description = "Skvaider user";
    group = "service";
    isSystemUser = true;
  };

  systemd.tmpfiles.rules = [
    "d /var/lib/skvaider 0755 skvaider service -"
    "d /var/log/skvaider 0755 skvaider service -"
  ];

  systemd.services.skvaider-gateway = {
    description = "Skvaider dev gateway";
    wantedBy = [ "multi-user.target" ];
    after = [ "network-online.target" ];
    requires = [ "network-online.target" ];
    serviceConfig = {
      User = "skvaider";
      Group = "service";
      StateDirectory = "skvaider";
      StateDirectoryMode = "0755";
      Restart = "on-failure";
      ExecStart = "${skvaider}/bin/skvaider-proxy -c ${configFile}";
    };
  };
}
