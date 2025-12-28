#!/bin/bash

END_TIME=$((SECONDS + 3*60*60))  # 3 hours

while [ $SECONDS -lt $END_TIME ]; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "/data/code/exp/models/Qwen/Qwen2.5-7B-Instruct",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
      ]
    }' > /dev/null

  # Optional: small sleep to avoid overwhelming CPU/network
  sleep 0.2
done
