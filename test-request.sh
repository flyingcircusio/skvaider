#!/usr/bin/env sh
# -H "X-Skvaider-Debug-Id: asdf" \
curl -v http://127.0.0.1:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer developer" \
  -H "X-Skvaider-Debug-Id: asdf" \
  -d '{     "model": "gemma",     "messages": [{"role": "user", "content": "Say this is a test!"}],     "temperature": 0.7   }' | jq
