
## Start dev server

```
uv run uvicorn skvaider:app_factory --factory --reload
```

By default it expects Ollama at `127.0.0.1:11434`. Use `OLLAMA_HOST` to change this.

## Testing

```bash
uv run pytest
```

Models are configured in `models.toml`.
