#!/bin/bash
# Run the file:// wasm harness in headless Chromium and stream the console.
# NOTE: do NOT use --virtual-time-budget — it fast-forwards timers and kills
# the renderer before the async GPU legs finish.
cd "$(dirname "$0")"
timeout 90 chromium --headless=new --no-sandbox --enable-unsafe-webgpu \
  --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface \
  --enable-webgpu-developer-features --enable-logging=stderr \
  "file://$(pwd)/webnm/index.html"
