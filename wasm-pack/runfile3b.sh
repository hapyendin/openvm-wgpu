#!/bin/bash
# Phase 3b run: classic+batched gate_c1 timing + in-browser W2C grind.
# Do NOT add --virtual-time-budget (fast-forwards timers, kills async GPU legs).
cd "$(dirname "$0")"
curl -s -m5 -o /dev/null -w "ENGINE_BEFORE:%{http_code}\n" http://127.0.0.1:8201/v1/models
timeout 120 chromium --headless=new --no-sandbox --enable-unsafe-webgpu \
  --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface \
  --enable-webgpu-developer-features --enable-logging=stderr \
  "file://$(pwd)/webnm/index.html"
RC=$?
curl -s -m5 -o /dev/null -w "ENGINE_AFTER:%{http_code}\n" http://127.0.0.1:8201/v1/models
echo "RUN_RC=$RC"
