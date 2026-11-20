#!/bin/bash
# W2B-BROWSER run: logup-leaf + GKR leg in-browser (file:// page; wasm does
# sync file:// XHRs for trace inputs). $1 = evidence log path. Do NOT add
# --virtual-time-budget. Poll for ALL_TESTS_RAN then pkill (teardown quirk).
LOG="${1:-/tmp/w2b-browser-run.log}"
cd "$(dirname "$0")"
: > "$LOG"
curl -s -m5 -o /dev/null -w "ENGINE_BEFORE:%{http_code}\n" http://127.0.0.1:8201/v1/models | tee -a "$LOG"
setsid chromium --headless=new --no-sandbox --enable-unsafe-webgpu \
  --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface \
  --enable-webgpu-developer-features --enable-logging=stderr \
  --allow-file-access-from-files \
  --js-flags="--max-old-space-size=8192" \
  "file://$(pwd)/webw2b/index.html" >>"$LOG" 2>&1 &
CPID=$!
DONE=0
for i in $(seq 1 1800); do
  sleep 1
  grep -aq ALL_TESTS_RAN "$LOG" && { DONE=1; break; }
  kill -0 $CPID 2>/dev/null || break
done
sleep 1
pkill -x chromium 2>/dev/null
RC=0; [ "$DONE" = 1 ] || RC=124
curl -s -m5 -o /dev/null -w "ENGINE_AFTER:%{http_code}\n" http://127.0.0.1:8201/v1/models | tee -a "$LOG"
echo "RUN_RC=$RC LOG=$LOG"
grep -aE 'W2B_|ALL_TESTS_RAN|FATAL|ENGINE_' "$LOG" | tail -50
