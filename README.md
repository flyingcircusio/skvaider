
## Start dev server

Ideally, this happens in a combination with a Nix-based devenv, as this provides a local postgresql.

You can start the devenv with `nix develop --impure` or by using direnv with a `.envrc` containing `use flake . --impure`.
To start the postgres inside the devenv run `devenv up`.

This project uses manual database migrations at the moment, these can be applied with the following command

```
psql -d skvaider -p 5432 -U skvaider < migrations/0001_init.sql
```

Then the server can be started with

```
uv run uvicorn skvaider:app_factory --factory --reload
```

By default it expects Ollama at `127.0.0.1:11434`. Use `OLLAMA_HOST` to change this.

## Testing

```bash
uv run pytest
```

Models are configured in `models.toml`.
