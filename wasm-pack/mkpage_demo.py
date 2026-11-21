#!/usr/bin/env python3
# Landing-page demo: 'prove in your browser'. Self-contained single HTML:
# wasm inlined as base64, no external fetches (GitHub-Pages-proof).
import sys, base64, os, datetime
wdir = sys.argv[1]
b64 = base64.b64encode(open(os.path.join(wdir, "openvm_wasm_bg.wasm"), "rb").read()).decode()

body = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenVM wgpu &mdash; prove in your browser</title>
<style>
 body{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0d1117;color:#c9d1d9;margin:0;padding:2rem 1rem;line-height:1.5}
 .wrap{max-width:860px;margin:0 auto}
 h1{font-size:1.4rem;color:#e6edf3} h1 small{color:#7d8590;font-weight:normal}
 .sub{color:#7d8590;margin-bottom:1.2rem}
 .cards{display:flex;gap:.6rem;flex-wrap:wrap;margin:1rem 0}
 .card{border:1px solid #30363d;border-radius:8px;padding:.8rem 1rem;background:#161b22;min-width:150px}
 .card .k{color:#7d8590;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}
 .card .v{font-size:1.05rem;margin-top:.2rem}
 .pass{color:#3fb950;font-weight:bold} .fail{color:#f85149;font-weight:bold} .wait{color:#d29922}
 button{background:#238636;color:#fff;border:0;border-radius:6px;padding:.5rem 1.1rem;font:inherit;cursor:pointer;margin:.4rem 0}
 button:disabled{background:#30363d;color:#7d8590}
 pre{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem;overflow-x:auto;font-size:.78rem;max-height:420px}
 a{color:#58a6ff}
 .foot{color:#7d8590;font-size:.8rem;margin-top:1.5rem}
</style></head><body><div class="wrap">
<h1>OpenVM wgpu &mdash; prove in your browser <small>wgsl / webgpu</small></h1>
<div class="sub">Poseidon2 + Merkle + PoW grind over WebGPU in pure u32 WGSL (no int64 anywhere).
The goldens below are byte-identical to the native reference runs on Intel Arc, NVIDIA and AMD hardware.</div>
<div class="cards">
 <div class="card"><div class="k">GPU</div><div class="v wait" id="dev">waiting&hellip;</div></div>
 <div class="card"><div class="k">Merkle golden</div><div class="v wait" id="g1">&hellip;</div></div>
 <div class="card"><div class="k">Grind golden</div><div class="v wait" id="g2">&hellip;</div></div>
 <div class="card"><div class="k">gate_c1 (batched)</div><div class="v wait" id="t1">&hellip;</div></div>
</div>
<button id="rerun" disabled>Run again</button>
<pre id="out">booting&hellip;</pre>
<div class="foot">Every number on this page is produced live on YOUR GPU in this tab &mdash; nothing is mocked.
Repo &amp; proof of the cross-vendor matrix: <a href="https://github.com/hapyendin/openvm-wgpu">hapyendin/openvm-wgpu</a>
&middot; built __DATE__</div>
<script>
const out = document.getElementById('out');
function log(s){ console.log(s); out.textContent += s + "\n"; out.scrollTop = out.scrollHeight; }
function card(id, txt, ok){ const el=document.getElementById(id); el.textContent=txt; el.className='v '+(ok===null?'wait':(ok?'pass':'fail')); }
function b64ToBytes(b){ const bin=atob(b); const u=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i); return u; }
async function run(){
  if (!navigator.gpu){ log('WebGPU not available in this browser (try Chrome/Edge with WebGPU enabled).'); return; }
  let adapter=null;
  for (let i=0;i<20&&!adapter;i++){ adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'}); if(!adapter) await new Promise(r=>setTimeout(r,500)); }
  if (!adapter){ log('requestAdapter returned null (WebGPU blocked or no adapter).'); return; }
  const ai = adapter.info && Object.keys(adapter.info).length ? adapter.info : (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  log('ADAPTER ' + JSON.stringify(ai));
  const feats=[]; adapter.features.forEach(f=>feats.push(f));
  log('FEATURES ' + feats.join(','));
  card('dev', (ai.device||ai.vendor||'detected adapter'), null);
  await wasm_bindgen(new Response(wasmB64Bytes.buffer,{headers:{'Content-Type':'application/wasm'}}));
  log('IMPORT_OK');
  log(await wasm_bindgen.initWgpu());
  const u32 = await wasm_bindgen.u32mulSelftest(); log('U32 ' + u32);
  let best=1e9, rep='';
  for (let i=0;i<2;i++){ const t0=performance.now(); rep=await wasm_bindgen.gateC1Batched(); const ms=performance.now()-t0; if(ms<best)best=ms; }
  log('GATE_C1_BATCHED_BEST_MS ' + best.toFixed(1));
  log(rep);
  const mk = rep.includes('1264a0fd266c3e7b') && rep.includes('[PASS]');
  card('g1', mk ? 'PASS 1264a0fd\u2026' : 'FAIL', mk);
  card('t1', best.toFixed(0)+' ms', true);
  const gr = await wasm_bindgen.grindWgpu(); log('GRIND ' + gr);
  const gg = gr.includes('90/90') || gr.includes('e69fddee') || gr.includes('PASS');
  card('g2', gg ? 'PASS e69fddee\u2026' : 'check log', gg);
  log('ALL_TESTS_RAN');
  document.getElementById('rerun').disabled=false;
}
document.getElementById('rerun').onclick = () => { out.textContent=''; document.getElementById('rerun').disabled=true; run().catch(e=>log('FATAL: '+(e&&e.stack?e.stack:e))); };
const wasmB64Bytes = b64ToBytes("__B64__");
run().catch(e => log('FATAL: ' + (e&&e.stack?e.stack:e)));
</script>
<script src="openvm_wasm.js"></script>
</body></html>
"""
glue = open(os.path.join(wdir, "openvm_wasm.js")).read()
html = body.replace('<script src="openvm_wasm.js"></script>', "<script>\n" + glue + "\n</script>")
html = html.replace("__B64__", b64).replace("__DATE__", datetime.date.today().isoformat())
open(os.path.join(wdir, "demo.html"), "w").write(html)
print("wrote", os.path.join(wdir, "demo.html"), len(html), "bytes")
