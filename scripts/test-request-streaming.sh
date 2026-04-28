#!/usr/bin/env sh

# -H "X-Skvaider-Debug: asdf" \
curl -v http://127.0.0.1:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer developer" \
  -d '{
    "model": "gemma",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
