#!/bin/bash
# Build the wasm bundle. Run detached: setsid nohup bash build.sh > /tmp/wasmbuild.log 2>&1 &
set -e
export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH
cd ~/appz/openvm-wgpu/wasm-pack
echo "[build] cargo build wasm ($(date))"
nice -n 10 cargo build --release --target wasm32-unknown-unknown
VER=$(grep -A1 'name = "wasm-bindgen"' Cargo.lock | grep '^version' | cut -d'"' -f2)
echo "[build] wasm-bindgen version: $VER"
WB=$HOME/.cargo/bin/wasm-bindgen-$VER
if [ ! -x "$WB" ]; then
  echo "[build] fetching wasm-bindgen-cli $VER (prebuilt)"
  ( cd /tmp
  for T in x86_64-unknown-linux-musl x86_64-unknown-linux-gnu; do
    if nice -n 10 curl -sSfLo wbc.tar.gz https://github.com/rustwasm/wasm-bindgen/releases/download/$VER/wasm-bindgen-$VER-$T.tar.gz; then
      tar xzf wbc.tar.gz
      cp wasm-bindgen-$VER-$T/wasm-bindgen "$WB"
      cp wasm-bindgen-$VER-$T/wasm-bindgen-test-runner "$HOME/.cargo/bin/" 2>/dev/null || true
      rm -rf wbc.tar.gz wasm-bindgen-$VER-$T
      break
    fi
  done )
fi
[ -x "$WB" ] || { echo "[build] NO wasm-bindgen-cli for $VER"; exit 1; }
mkdir -p web
"$WB" --target web --out-dir web target/wasm32-unknown-unknown/release/openvm_wasm.wasm
echo "[build] DONE $(date)"; ls -la web/
