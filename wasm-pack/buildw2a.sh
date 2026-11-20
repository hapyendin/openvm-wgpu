#!/bin/bash
# W2A-BROWSER build: wasm lib + no-modules glue into webw2a/ + base64 page.
set -e
export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH
cd ~/appz/openvm-wgpu/wasm-pack
echo "[w2a] cargo build wasm $(date)"
nice -n 10 cargo build --release --target wasm32-unknown-unknown
VER=$(grep -A1 "name = \"wasm-bindgen\"" Cargo.lock | grep "^version" | cut -d"\"" -f2)
WB=$HOME/.cargo/bin/wasm-bindgen-$VER
[ -x "$WB" ] || { echo "[w2a] NO wasm-bindgen-cli $VER"; exit 1; }
nice -n 10 "$WB" --target no-modules --out-dir webw2a target/wasm32-unknown-unknown/release/openvm_wasm.wasm
nice -n 10 python3 mkpage_w2a.py
echo "[w2a] DONE $(date)"
