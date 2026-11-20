#!/bin/bash
# W2C-BROWSER run: WHIR fold leg in-browser (file:// page; JS XHRs f0/alphas
# per instance). $1 = evidence log path. Poll for ALL_TESTS_RAN then pkill.
LOG="${1:-/tmp/w2c-browser-run.log}"
cd "$(dirname "$0")"
: > "$LOG"
curl -s -m5 -o /dev/null -w "ENGINE_BEFORE:%{http_code}\n" http://127.0.0.1:8201/v1/models | tee -a "$LOG"
setsid chromium --headless=new --no-sandbox --enable-unsafe-webgpu \
  --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface \
  --enable-webgpu-developer-features --enable-logging=stderr \
  --allow-file-access-from-files \
  --js-flags="--max-old-space-size=8192" \
  "file://$(pwd)/webw2c/index.html" >>"$LOG" 2>&1 &
CPID=$!
DONE=0
for i in $(seq 1 3600); do
  sleep 1
  grep -aq ALL_TESTS_RAN "$LOG" && { DONE=1; break; }
  kill -0 $CPID 2>/dev/null || break
done
sleep 1
pkill -x chromium 2>/dev/null
RC=0; [ "$DONE" = 1 ] || RC=124
curl -s -m5 -o /dev/null -w "ENGINE_AFTER:%{http_code}\n" http://127.0.0.1:8201/v1/models | tee -a "$LOG"
echo "RUN_RC=$RC LOG=$LOG"
grep -aE 'W2C_|ALL_TESTS_RAN|FATAL|ENGINE_' "$LOG" | tail -80
