#!/usr/bin/env python3
# Build webw2b/index.html: the W2B logup-leaf leg IN-BROWSER (Phase 3d).
# The page streams the world inputs via ASYNC file:// XHRs one file at a
# time into resumable wasm exports (sync-XHR arraybuffer is illegal and
# x-user-defined is ignored by Chromium — see WASM-REPORT §9). Two full
# passes (leaf build + 25 GKR rounds each); asserts leaf + out digests,
# per-layer/concat goldens byte-exact, determinism.
import base64, pathlib
root = pathlib.Path.home() / "appz/openvm-wgpu/wasm-pack"
wasm = base64.b64encode((root / "webw2b/openvm_wasm_bg.wasm").read_bytes()).decode()
js = (root / "webw2b/openvm_wasm.js").read_text()
html = """<!doctype html><html><head><meta charset=utf-8></head><body>
<pre id=out style="font-size:10px"></pre>
<script>
const B64='WASMB64';
const INPUTS='file:///home/matt/appz/openvm-w2b/work/w2b-inputs/';
const TRACES='file:///home/matt/appz/openvm-sycl-skel/b9/dump-cpu9/traces/app-segments_0/';
const GOLD_LEAF='f5ecc63de1646b064ff943f9bd59cbea0b2ec799a73dc5c33cafe3342d4a965f';
const GOLD_OUT='8c06f8ac2b4070d68b46f2a7d2112855d354c91584b3edeeb374e84a8e370a63';
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
  const {txt:worldTxt}=await xhrText(INPUTS+'leafworld.json');
  const {txt:shasTxt}=await xhrText(INPUTS+'layer-shas.json');
  const {txt:gkrTxt}=await xhrText(INPUTS+'gkr-inputs-w2b.json');
  if(!worldTxt||!shasTxt||!gkrTxt){log('W2B_STAGE_JSON: FAIL');log('ALL_TESTS_RAN');return;}
  log('W2B_STAGE_JSON: OK');
  let reps=[];
  for(const run of [1,2]){
    const t0=performance.now();
    const man=JSON.parse(await wasm_bindgen.w2bPlan(worldTxt));
    log('W2B_PLAN R'+run+': airs='+man.length);
    for(const air of man){
      const ft=await xhrBuf(INPUTS+air.tables);
      await wasm_bindgen.w2bFeedTables(air.i,new Uint8Array(ft.buf));
      let xms=0;
      for(const tf of air.traces){
        const b=await xhrBuf(TRACES+tf);
        await wasm_bindgen.w2bFeedTrace(air.i,new Uint8Array(b.buf));
        xms+=b.ms;
      }
      await wasm_bindgen.w2bAirRun(air.i);
      log('W2B_FEED R'+run+' air'+air.air+' files='+(1+air.traces.length)+' xhr_ms='+xms.toFixed(0));
    }
    const sha=await wasm_bindgen.w2bLeafSha();
    log('W2B_LEAFSHA R'+run+': '+sha);
    const rep=await wasm_bindgen.w2bGkr(run,gkrTxt,shasTxt);
    log(rep.trim());
    log('W2B_PASS R'+run+' wall_ms='+(performance.now()-t0).toFixed(0));
    reps.push(rep);
  }
  const all=reps.join('\\n');
  const runs=[...all.matchAll(/W2B_RUN (\\d+) LEAF_SHA256 ([0-9a-f]{64}) OUT_SHA256 ([0-9a-f]{64})/g)];
  const rounds=[...all.matchAll(/W2B_RUN (\\d+) ROUNDS (\\d+)\\/(\\d+) MISMATCHES (\\d+)/g)];
  const layers=[...all.matchAll(/W2B_RUN (\\d+) LAYERS (OK|FAIL)/g)];
  const concats=[...all.matchAll(/W2B_RUN (\\d+) CONCAT [0-9a-f]{64} CONCAT_OK (true|false)/g)];
  let ok=true;
  for(const r of runs){
    const leafOk=r[2]===GOLD_LEAF&&r[2].length===64;
    const outOk=r[3]===GOLD_OUT&&r[3].length===64;
    ok=ok&&leafOk&&outOk;
    log('W2B_LEAF_DIGEST R'+r[1]+': '+(leafOk?'PASS '+r[2]:'FAIL ('+r[2]+')'));
    log('W2B_OUT_DIGEST R'+r[1]+': '+(outOk?'PASS '+r[3]:'FAIL ('+r[3]+')'));
  }
  const roundsOk=rounds.length===2&&rounds.every(x=>x[2]===x[3]&&x[4]==='0');
  ok=ok&&roundsOk;
  log('W2B_GKR_ROUNDS: '+(roundsOk?'PASS':'FAIL'));
  const layersOk=layers.length===2&&layers.every(x=>x[2]==='OK');
  ok=ok&&layersOk;
  log('W2B_LAYERS_SHA: '+(layersOk?'PASS':'FAIL'));
  const concatOk=concats.length===2&&concats.every(x=>x[2]==='true');
  ok=ok&&concatOk;
  log('W2B_CONCAT_SHA: '+(concatOk?'PASS':'FAIL'));
  const det=runs.length===2&&runs[0][2]===runs[1][2]&&runs[0][3]===runs[1][3]&&runs[0][2]===GOLD_LEAF&&runs[0][3]===GOLD_OUT;
  ok=ok&&det;
  log('W2B_DETERMINISM: '+(det?'PASS':'FAIL'));
  log('W2B_BROWSER: '+(ok?'PASS':'FAIL'));
  log('ALL_TESTS_RAN');
 }catch(e){log('FATAL '+(e&&e.stack?e.stack:e));log('ALL_TESTS_RAN');}
})();
</script>
</body></html>"""
html = html.replace("__JS__", js).replace("WASMB64", wasm)
out = root / "webw2b" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print("wrote", out, len(html))
