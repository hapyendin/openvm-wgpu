#!/usr/bin/env bash
# Runner failure self-diagnosis: called by ci.yml steps with `if: failure()`.
# Dumps full runner state into the run log (persistent feedback for "random"
# runner errors) and posts a one-line summary as a commit status.
{
  echo "=== RUNNER DIAG ==="
  date -u; uptime; uname -a
  echo "--- memory ---";     free -h
  echo "--- disk ---";       df -h 2>/dev/null | sort -u | head -8
  echo "--- cpu ---";        nproc
  echo "--- toolchain ---";  cargo --version 2>/dev/null; rustc --version 2>/dev/null; echo "PATH=$PATH"
  echo "--- gpu ---";        ls -la /dev/dri 2>/dev/null || echo "NO /dev/dri"
  vulkaninfo --summary 2>&1 | sed -n '/deviceName/p' || true
  echo "--- runner journal ---"; journalctl -u act_runner -n 60 --no-pager 2>&1 | tail -60 || echo "journal not readable"
  echo "--- dmesg tail ---"; dmesg 2>&1 | tail -30 || echo "dmesg not readable"
  echo "--- top mem procs ---"; ps aux --sort=-%mem 2>/dev/null | head -8
  echo "--- target dir ---"; du -sh target 2>/dev/null || true
} 2>&1 | tee /tmp/diag.txt

# Kill-reason sniff: what actually terminated us?
SUMMARY=$(grep -oiE "oom-kill|out of memory|no space left on device|killed process|device (lost|hung)|panic|SIGKILL|signal 9|exit status [0-9]+" /tmp/diag.txt | sort -u | head -3 | paste -sd', ' -)
SUMMARY="${SUMMARY:-job failed; full diag block in run log}"
SUMMARY=$(echo "$SUMMARY" | cut -c1-130 | tr '"' "'")

API="${GITHUB_API_URL:-}"
REPO="${GITHUB_REPOSITORY:-}"
SHA="${GITHUB_SHA:-}"
TOK="${GITHUB_TOKEN:-}"
if [ -n "$SHA" ] && [ -n "$TOK" ]; then
  curl -s -o /dev/null -m 15 -X POST "$API/repos/$REPO/statuses/$SHA" \
    -H "Authorization: token $TOK" -H "Content-Type: application/json" \
    -d "{\"state\":\"failure\",\"context\":\"ci/runner-diag\",\"description\":\"$SUMMARY\"}" || true
fi
echo "DIAG_POSTED $SUMMARY"
