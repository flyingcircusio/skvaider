
## Start dev server

```
uv run uvicorn skvaider:app_factory --factory --reload
```

By default it expects Ollama at `127.0.0.1:11434`. Use `OLLAMA_HOST` to change this.

## Testing

```bash
python test_endpoints.py
python test_openai_client.py
```

Models are configured in `models.toml`.
