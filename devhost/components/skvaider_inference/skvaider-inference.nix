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

  nixpkgs-unstable = import (builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/0182a361324364ae3f436a63005877674cf45efb.tar.gz";
    sha256 = "1i04bclcxsqhk172wvj74fcgk2sd7037mi9bgxp7jdx42886bl6h";
  }) { };

  vllm-cpu = nixpkgs-unstable.vllm;

  tiny-gpt2 =
    let
      fetch =
        name: sha256:
        pkgs.fetchurl {
          url = "https://huggingface.co/sshleifer/tiny-gpt2/resolve/main/${name}";
          inherit sha256;
        };
    in
    pkgs.runCommand "tiny-gpt2" { } ''
      mkdir $out
      cp ${fetch "config.json" "1c20ncwg0nxyq0b5bmqs92s9i3rrgkly0qawky9bmgis1j1ykabp"} $out/config.json
      cp ${fetch "tokenizer_config.json" "07wk83wkzd6ykm6y8xzy6ipwh5b6dcm6rqs2199q659sdrhfn12y"} $out/tokenizer_config.json
      cp ${fetch "vocab.json" "09rgyz8xllry92darghnji34rnyjhzkl6ykwdsv1iikhpi9ph203"} $out/vocab.json
      cp ${fetch "merges.txt" "1idd4rvkpqqbks51i2vjbd928inw7slij9l4r063w3y5fd3ndq8w"} $out/merges.txt
      cp ${fetch "pytorch_model.bin" "1rh4bk5fqjy74k5r1dwmm6ax40fj0djapmfycpkxyaq36i0b41mp"} $out/pytorch_model.bin
    '';

  configFile = (pkgs.formats.toml { }).generate "skvaider-inference.toml" {
    models_dir = "/var/lib/skvaider/model";
    server = {
      host = "0.0.0.0";
      port = 8000;
    };
    logging = {
      log_level = "DEBUG";
      log_dir = "/var/log/skvaider";
    };
    openai.models = [
      {
        id = "tiny-gpt2";
        task = "chat";
        engine = "vllm";
        repo = "${tiny-gpt2}";
        revision = "main";
        context_size = 128;
        max_requests = 1;
        port = 8001;
        cmd_args = [
          "--device"
          "cpu"
        ];
      }
    ];
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
    "d /var/lib/skvaider/model 0755 skvaider service -"
    "d /var/log/skvaider 0755 skvaider service -"
  ];

  networking.firewall.allowedTCPPorts = [ 8000 ];

  systemd.services.skvaider-inference = {
    description = "Skvaider dev inference";
    wantedBy = [ "multi-user.target" ];
    after = [ "network-online.target" ];
    requires = [ "network-online.target" ];
    path = [ vllm-cpu ];
    environment = {
      HF_HUB_DISABLE_PROGRESS_BARS = "1";
      HF_TOKEN = "";
      HOME = "/var/lib/skvaider/model";
    };
    serviceConfig = {
      User = "skvaider";
      Group = "service";
      StateDirectory = "skvaider";
      StateDirectoryMode = "0755";
      Restart = "on-failure";
      ExecStart = "${skvaider}/bin/skvaider-inference -c ${configFile}";
    };
  };
}
