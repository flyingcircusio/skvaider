{ pkgs, lib, ... }:
let
  nixpkgs-unstable = import (pkgs.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "0182a361324364ae3f436a63005877674cf45efb";
    hash = "sha256-0NBlEBKkN3lufyvFegY4TYv5mCNHbi5OmBDrzihbBMQ=";
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
in
{
  flyingcircus.roles.ai-model-server.enable = true;
  flyingcircus.roles.ai-model-server.enableRocm = false;
  flyingcircus.roles.ai-model-server.skvaider-inference.hf_token = "";
  flyingcircus.roles.ai-model-server.skvaider-inference.enable = true;

  systemd.services.skvaider-inference.path = lib.mkAfter [ vllm-cpu ];

  flyingcircus.roles.ai-model-server.skvaider-inference.settings = {
    models_dir = "/var/lib/skvaider/model";
    server.host = "0.0.0.0";
    server.port = 8000;
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

  networking.firewall.allowedTCPPorts = [ 8000 ];
}
