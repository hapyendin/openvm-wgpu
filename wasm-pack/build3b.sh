#!/bin/bash
# Phase 3b build: wasm lib + no-modules glue into webnm/ + base64 page.
set -e
export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH
cd ~/appz/openvm-wgpu/wasm-pack
echo "[3b] cargo build wasm $(date)"
nice -n 10 cargo build --release --target wasm32-unknown-unknown
VER=$(grep -A1 "name = \"wasm-bindgen\"" Cargo.lock | grep "^version" | cut -d"\"" -f2)
WB=$HOME/.cargo/bin/wasm-bindgen-$VER
[ -x "$WB" ] || { echo "[3b] NO wasm-bindgen-cli $VER"; exit 1; }
nice -n 10 "$WB" --target no-modules --out-dir webnm target/wasm32-unknown-unknown/release/openvm_wasm.wasm
nice -n 10 python3 mkpage.py webnm
echo "[3b] DONE $(date)"
