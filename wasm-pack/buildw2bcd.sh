#!/bin/bash
# W2B/W2C-BROWSER build (Phase 3d): wasm lib + no-modules glue into webw2b/
# and webw2c/ + base64 pages.
set -e
export PATH=$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH
cd ~/appz/openvm-wgpu/wasm-pack
echo "[w2bcd] cargo build wasm $(date)"
nice -n 10 cargo build --release --target wasm32-unknown-unknown
VER=$(grep -A1 "name = \"wasm-bindgen\"" Cargo.lock | grep "^version" | cut -d"\"" -f2)
WB=$HOME/.cargo/bin/wasm-bindgen-$VER
[ -x "$WB" ] || { echo "[w2bcd] NO wasm-bindgen-cli $VER"; exit 1; }
mkdir -p webw2b webw2c
nice -n 10 "$WB" --target no-modules --out-dir webw2b target/wasm32-unknown-unknown/release/openvm_wasm.wasm
nice -n 10 "$WB" --target no-modules --out-dir webw2c target/wasm32-unknown-unknown/release/openvm_wasm.wasm
nice -n 10 python3 mkpage_w2b.py
nice -n 10 python3 mkpage_w2c.py
mkdir -p webw2d
nice -n 10 "$WB" --target no-modules --out-dir webw2d target/wasm32-unknown-unknown/release/openvm_wasm.wasm
nice -n 10 python3 mkpage_w2d.py
echo "[w2bcd] DONE $(date)"
