#!/usr/bin/env python3
# Build webw2d/index.html: the FULL W2D prove chain IN-BROWSER (Phase 3d).
# transcript -> leaf (w2b feeds, W2D world) -> GKR(live) -> grinds -> WHIR
# fold -> emitted stream; asserts gpu_sha256 == 3e3bfbec...64 (native golden)
# and rec sha of the recorded stream via crypto.subtle.
import base64, pathlib
root = pathlib.Path.home() / "appz/openvm-wgpu/wasm-pack"
wasm = base64.b64encode((root / "webw2d/openvm_wasm_bg.wasm").read_bytes()).decode()
js = (root / "webw2d/openvm_wasm.js").read_text()
html = """<!doctype html><html><head><meta charset=utf-8></head><body>
<pre id=out style="font-size:10px"></pre>
<script>
const B64='WASMB64';
const W='file:///home/matt/appz/openvm-sycl-skel/b9/r15/w2d-r1/';
const I='file:///home/matt/appz/openvm-wgpu/w2d/w2d-inputs/';
const S='file:///home/matt/appz/openvm-wgpu/w2d/streams-w2d/';
const TR=W+'traces/app-segments_0/';
const GOLD='3e3bfbec62b817f5f9813a326af4698816bd9f95fd4a7d83a9ea9e6f4b7a3d0d';
const outEl=document.getElementById('out');
function log(s){console.log(s);outEl.textContent+=s+'\\n';}
function b64ToBytes(b){const bin=atob(b);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
async function xhrText(url){
  const t0=performance.now();
  const txt=await new Promise((res,rej)=>{
    const x=new XMLHttpRequest();
    x.open('GET',url,true);
    x.onload=()=>res(x.status===200?x.responseText:null);
    x.onerror=()=>rej('xhr-err '+url);
    x.send();
  });
  return {ms:performance.now()-t0, txt};
}
async function xhrBuf(url){
  const t0=performance.now();
  const buf=await new Promise((res,rej)=>{
    const x=new XMLHttpRequest();
    x.open('GET',url,true); x.responseType='arraybuffer';
    x.onload=()=>res(x.status===200?x.response:null);
    x.onerror=()=>rej('xhr-err '+url);
    x.send();
  });
  return {ms:performance.now()-t0, buf};
}
async function shaHex(buf){
  const d=await crypto.subtle.digest('SHA-256',buf);
  return [...new Uint8Array(d)].map(b=>b.toString(16).padStart(2,'0')).join('');
}
</script>
<script>__JS__</script>
<script>
(async()=>{
 try{
  await wasm_bindgen(new Response(b64ToBytes(B64).buffer,{headers:{'Content-Type':'application/wasm'}}));
  log('IMPORT_OK');
  let adapter=null;
  for(let i=0;i<20 && !adapter;i++){
    adapter = await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    if(!adapter) await new Promise(r=>setTimeout(r,500));
  }
  if(!adapter){log('FATAL: no adapter');log('ALL_TESTS_RAN');return;}
  log('JS_ADAPTER: true');
  log(await wasm_bindgen.initWgpu());
  const stream=await xhrBuf(S+'app-segments_0.prove.bin');
  const flags=await xhrBuf(S+'app-segments_0.prove.bin.flags');
  const recsha=await shaHex(stream.buf);
  const psites=(await xhrText(W+'psites/pow_sites.jsonl')).txt;
  const meta=(await xhrText(W+'whir/w0-meta.txt')).txt;
  const wi=JSON.parse((await xhrText(I+'world-info.json')).txt);
  const nr=meta.match(/rounds=\\[([^\\]]*)\\]/)[1].split(',').length;
  const kw=+meta.match(/k_whir=(\\d+)/)[1];
  log(log.length?'':'');
  log('W2D_STAGE n='+stream.buf.byteLength/8+' recsha='+recsha+' nr='+nr);
  log(await wasm_bindgen.w2dInit(new Uint8Array(stream.buf),new Uint8Array(flags.buf),psites,meta));
  log(await wasm_bindgen.w2dTranscript());
  // ---- leaf build via the w2b state machine, W2D world inputs ----
  const lwt=(await xhrText(I+'leafworld.json')).txt;
  const man=JSON.parse(await wasm_bindgen.w2bPlan(lwt));
  log('W2D_PLAN airs='+man.length);
  for(const air of man){
    const ft=await xhrBuf(I+air.tables);
    await wasm_bindgen.w2bFeedTables(air.i,new Uint8Array(ft.buf));
    for(const tf of air.traces){
      const b=await xhrBuf(TR+tf);
      await wasm_bindgen.w2bFeedTrace(air.i,new Uint8Array(b.buf));
    }
    await wasm_bindgen.w2bAirRun(air.i);
  }
  const leafsha=await wasm_bindgen.w2bLeafSha();
  const leafOk=leafsha===wi.leaf_sha256_raw_dump;
  log('W2D_LEAF '+leafsha+' EXP '+wi.leaf_sha256_raw_dump+' OK '+leafOk);
  // ---- GKR (live constants) ----
  const gj=(await xhrText(I+'gkr-inputs-w2d.json')).txt;
  log(await wasm_bindgen.w2dGkr(gj));
  // ---- grinds ----
  log((await wasm_bindgen.w2dGrinds()).trim());
  // ---- fold ----
  const alphas=await xhrBuf(W+'whir/w0-alphas.bin');
  const mu=await xhrBuf(W+'whir/w0-mu.bin');
  log(await wasm_bindgen.w2dFoldBegin(new Uint8Array(alphas.buf),new Uint8Array(mu.buf)));
  const f0=await xhrBuf(W+'whir/w0-f0.bin');
  const w0=await xhrBuf(W+'whir/w0-w0.bin');
  const PC=65536; // rows per pair chunk (2 MiB/file)
  for(let r=0;r<nr;r++){
    for(let kq=0;kq<kw;kq++){
      const streamed=await wasm_bindgen.w2dFoldKqBegin()==='true';
      if(streamed){
        const rows=f0.buf.byteLength/32;
        for(let j=0;j<rows;j+=PC){
          const e=Math.min(j+PC,rows);
          await wasm_bindgen.w2dFoldPairChunk(
            new Uint8Array(f0.buf,j*32,(e-j)*32),
            new Uint8Array(w0.buf,j*32,(e-j)*32));
        }
        f0.buf=null; w0.buf=null;
      }
      log(await wasm_bindgen.w2dFoldKqEnd());
    }
    const fc=await xhrBuf(W+'whir/w0-finalcoeffs.bin');
    log(await wasm_bindgen.w2dFoldRound(new Uint8Array(fc.buf)));
  }
  const fin=await wasm_bindgen.w2dFinish();
  log(fin);
  const m=fin.match(/W2D_GPU_SHA256 ([0-9a-f]{64})/);
  const gpu=m?m[1]:'';
  const shaOk=gpu.length===64&&gpu===GOLD;
  const recOk=recsha===GOLD;
  const green=/GREEN true/.test(fin);
  const ok=shaOk&&recOk&&green&&leafOk;
  log('W2D_SHA_GPU_OK: '+(shaOk?'PASS':'FAIL ('+gpu+')'));
  log('W2D_SHA_REC: '+(recOk?'PASS':'FAIL ('+recsha+')'));
  log('W2D_BROWSER: '+(ok?'PASS':'FAIL'));
  log('ALL_TESTS_RAN');
 }catch(e){log('FATAL '+(e&&e.stack?e.stack:e));log('ALL_TESTS_RAN');}
})();
</script>
</body></html>"""
html = html.replace("__JS__", js).replace("WASMB64", wasm)
out = root / "webw2d" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print("wrote", out, len(html))
