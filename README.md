## About

skvaider is a proxy and router for (Open)AI-APIs.

Features:

* route requests to different backend servers based on model availability and load
* automated health checks
* automated model discovery
* token authentication

> The name `skvaider` is derived from the Swedish word `skvader` which is one
> of those mythical beasts concocted from multiple real animals. Incidentally
> a flying rabbit seems more than apt as this project was started by the
> Flying Circus.

## Development environment

> This requires a local Nix installation.
> This also requires a devenv installation.

This will start the development server, postgresql, and other auxiliary services.

```bash
$ devenv up
```

### Variations of the development environment

You can also activate the development environment with devenv shell:

```bash
$ devenv shell
```

### CPU-only vLLM on a devhost VM

The `dev` batou environment provisions a devhost VM and deploys a local
CPU-only vLLM setup. This is the closest equivalent to an editable `fc.qemu`
workflow: batou copies this checkout into the VM's service user, writes the
NixOS config, and triggers the platform rebuild.

Create or update the VM and deploy skvaider from the batou deployment directory:

```bash
cd devhost
./batou deploy dev
```

To rebuild the VM from scratch:

```bash
cd devhost
./batou deploy dev --provision-rebuild
```

After deployment, batou writes `devhost/ssh_config_dev`. Check the local gateway through
the provisioned VM:

```bash
ssh -F devhost/ssh_config_dev skvaider \
  "curl -sf -H 'Authorization: Bearer developer' http://127.0.0.1:23211/openai/v1/models"
```

The served test model is `tiny-gpt2`; inference is forced through vLLM with
`--device cpu`, so the workflow works without GPU hardware. The provisioning
target and VM resources live in `devhost/environments/dev/environment.cfg`.

## Testing

```bash
$ devenv shell
$ run-tests
```


## Authentication

skvaider authenticates API clients via bearer tokens. Add one or more tokens to
your `config.toml`:

```toml
[auth]
admin_tokens = ["<token>"]
```

Generate a secure token:

```bash
openssl rand -hex 32
```

The same token is used by `check-skvaider` (read from `config.auth.admin_tokens[0]`)
and by API clients in the `Authorization: Bearer <token>` header.


## Embedding reference files

Pre-recorded embedding vectors used for numerical stability checks. Format:
`{model_id: {text: [float, ...]}}`. Consumed by the inference health check
Consumed by the inference health check (`embedding_verification_file` in inference config).

Generate from a live instance:

```bash
skvaider-generate-embedding-reference \
    --config config.toml \
    --url https://ai.example.com \
    --dataset dataset.txt \
    --output embeddings-reference.json
```

`dataset.txt` is one sentence per line. The tool reads `task = "embedding"` models from the
proxy config — re-run after any weight update.

In fc-nixos the default reference is `nixos/roles/embeddings-reference.json` (`lib.mkDefault`).
Regenerate it and commit when weights change; override per-server in local NixOS config if needed.
