#!/usr/bin/env python3
# Generate webnm/index.html for --target no-modules glue with the wasm
# inlined as base64 (file:// pages cannot fetch the .wasm as a module).
# Phase 3b: adds gateC1Batched timing (atlas + batched mapAsync) vs the
# classic per-dispatch path, and the in-browser W2C grind.
import sys, base64, os

wdir = sys.argv[1]  # wasm-bindgen no-modules out dir
b64 = base64.b64encode(open(os.path.join(wdir, "openvm_wasm_bg.wasm"), "rb").read()).decode()

body = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>OpenVM WASM gate</title></head>
<body>
<h3>OpenVM wgpu/WGSL browser tier</h3>
<pre id="out"></pre>
<script>
const out = document.getElementById('out');
function log(s) { console.log(s); out.textContent += s + "\\n"; }
function b64ToBytes(b) {
  const bin = atob(b); const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}
const best = {};
async function timed(name, fn) {
  const t0 = performance.now();
  const r = await fn();
  const ms = performance.now() - t0;
  log("PERF " + name + " ms=" + ms.toFixed(1));
  if (!(name in best) || ms < best[name]) best[name] = ms;
  return r;
}
async function main() {
  log("BOOT");
  if (!navigator.gpu) { log("JS_ADAPTER: navigator.gpu MISSING"); return; }
  let adapter = null;
  for (let i = 0; i < 20 && !adapter; i++) {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) await new Promise(r => setTimeout(r, 500));
  }
  if (!adapter) { log("JS_ADAPTER: requestAdapter returned null x20"); return; }
  const ai = adapter.info && Object.keys(adapter.info).length ? adapter.info
    : (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  log("JS_ADAPTER info: " + JSON.stringify(ai));
  const lims = {};
  for (const k in adapter.limits) { const v = adapter.limits[k]; if (typeof v === 'number') lims[k] = v; }
  log("JS_ADAPTER limits: " + JSON.stringify(lims));
  const feats = []; adapter.features.forEach(f => feats.push(f));
  log("JS_ADAPTER features: " + feats.join(','));
  await wasm_bindgen(new Response(wasmB64Bytes.buffer, { headers: { 'Content-Type': 'application/wasm' } }));
  log("IMPORT_OK");
  log("--- initWgpu ---");
  log(await wasm_bindgen.initWgpu());
  log("--- u32mulSelftest ---");
  log(await wasm_bindgen.u32mulSelftest());
  log("--- gateC1 classic vs batched (atlas) ---");
  const c1 = await timed("gateC1Gpu_classic", () => wasm_bindgen.gateC1Gpu());
  const b1 = await timed("gateC1Batched", () => wasm_bindgen.gateC1Batched());
  const c2 = await timed("gateC1Gpu_classic", () => wasm_bindgen.gateC1Gpu());
  const b2 = await timed("gateC1Batched", () => wasm_bindgen.gateC1Batched());
  log("--- gateC1Gpu (classic) full output ---");
  log(c1);
  log("--- gateC1Batched full output ---");
  log(b2);
  log("JS_GATE_C1_CLASSIC_BEST_MS " + best["gateC1Gpu_classic"].toFixed(1));
  log("JS_GATE_C1_BATCHED_BEST_MS " + best["gateC1Batched"].toFixed(1));
  const strip = s => s.split("\\n").filter(l => !/^(ATLAS|GATE_C1_GPU:|GATE_C1_BATCHED:)/.test(l)).join("\\n");
  const eq = (strip(c1) === strip(b1)) && (strip(c2) === strip(b2));
  log("GATE_C1_BATCHED_EQ_CLASSIC: " + (eq ? "PASS" : "FAIL"));
  log("--- grindWgpu ---");
  log(await wasm_bindgen.grindWgpu());
  log("ALL_TESTS_RAN");
}
</script>
<script src="openvm_wasm.js"></script>
<script>
const wasmB64Bytes = b64ToBytes("__B64__");
main().catch(e => log("FATAL: " + (e && e.stack ? e.stack : e)));
</script>
</body>
</html>
"""

html = body.replace("__B64__", b64)
open(os.path.join(wdir, "index.html"), "w").write(html)
print("wrote", os.path.join(wdir, "index.html"), len(html))
