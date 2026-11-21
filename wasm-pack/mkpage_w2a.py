#!/usr/bin/env python3
# Build webw2a/index.html: the W2A GKR leg IN-BROWSER (Phase 3c).
# file:// XHRs the recorded gkr-inputs.json + the agg-leaf_0 leaf dump
# (64 MiB, the golden instance) into wasm, runs the async GKR driver twice,
# asserts leaf merkle digest + folded-claim hash byte-exact + determinism.
import base64, pathlib
root = pathlib.Path.home() / "appz/openvm-wgpu/wasm-pack"
wasm = base64.b64encode((root / "webw2a/openvm_wasm_bg.wasm").read_bytes()).decode()
js = (root / "webw2a/openvm_wasm.js").read_text()
html = """<!doctype html><html><head><meta charset=utf-8></head><body>
<pre id=out style="font-size:10px"></pre>
<script>
const B64='WASMB64';
const JSON_URL='file:///DATA/gkr-inputs.json';
const LEAF_URL='file:///DATA/sycl-skel/b9/r15/x-l1a-r1/dbg_agg-leaf_0_leaves_cpu.bin';
const INST='agg-leaf_0';
const GOLD_LEAF='5c65a7bbc355594da3b4bd1866d6e128a013b5d65c1d15405e149ef0c6edb6ac';
const GOLD_CLAIM='fd53a33e82d7319b551a97c0710e5e75fac26980c0b6c1b02bd79bef0929f691';
const outEl=document.getElementById('out');
function log(s){console.log(s);outEl.textContent+=s+'\\n';}
function b64ToBytes(b){const bin=atob(b);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
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
function field(rep,key){
  const m=rep.split('\\n').find(l=>l.startsWith(key+' '));
  return m?m.split(' ')[1]:null;
}
function line(rep,key){
  const m=rep.split('\\n').find(l=>l.startsWith(key+' '));
  return m?m.slice(key.length+1):null;
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
  const {ms:jms,txt}=await xhrText(JSON_URL);
  if(!txt){log('W2A_STAGE_JSON: FAIL');log('ALL_TESTS_RAN');return;}
  log('W2A_STAGE_JSON: OK bytes='+txt.length+' ms='+jms.toFixed(1));
  const instObj=JSON.parse(txt).instances.find(i=>i.name===INST);
  const expRounds=instObj.rounds.length;
  const expSubs=instObj.rounds.reduce((a,r)=>a+r.subs.length,0);
  const {ms:lms,buf}=await xhrBuf(LEAF_URL);
  if(!buf){log('W2A_STAGE_LEAF: FAIL');log('ALL_TESTS_RAN');return;}
  log('W2A_STAGE_LEAF: OK bytes='+buf.byteLength+' ms='+lms.toFixed(1));
  const leaf=new Uint8Array(buf);
  const claims=[]; let okAll=true;
  for(const run of [1,2]){
    const rep=await wasm_bindgen.w2aRun(txt,INST,leaf);
    log('--- W2A RUN '+run+' ---');
    log(rep.trim());
    const leafSha=field(rep,'W2A_LEAF_SHA256');
    const claim=field(rep,'W2A_CLAIM_SHA256');
    const tok=line(rep,'W2A_INSTANCE');           // "agg-leaf_0 t=21 rounds=20/20 subs=210"
    const rounds=tok?tok.split(' ')[2].split('=')[1]:'?';
    const mism=parseInt(field(rep,'W2A_VALUE_MISMATCHES'));
    const leafOk=(leafSha===GOLD_LEAF);
    const roundsOk=(rounds===expRounds+'/'+expRounds && mism===0);
    const claimOk=(claim===GOLD_CLAIM && claim.length===64);
    okAll = okAll && leafOk && roundsOk && claimOk;
    log('W2A_MERKLE_LEAF_DIGEST R'+run+': '+(leafOk?'PASS':'FAIL ('+leafSha+')'));
    log('W2A_GKR_ROUNDS R'+run+': '+(roundsOk?'PASS':'FAIL rounds='+rounds+' mism='+mism));
    log('W2A_FOLDED_CLAIM R'+run+': '+(claimOk?'PASS':'FAIL ('+claim+')'));
    claims.push(claim);
  }
  const det=claims[0]===claims[1]&&claims[0]===GOLD_CLAIM;
  log('W2A_DETERMINISM: '+(det?'PASS':'FAIL '+claims.join(' vs ')));
  log('W2A_BROWSER: '+((okAll&&det)?'PASS':'FAIL'));
  log('ALL_TESTS_RAN');
 }catch(e){log('FATAL '+(e&&e.stack?e.stack:e));}
})();
</script>
</body></html>"""
html = html.replace("__JS__", js).replace("WASMB64", wasm)
out = root / "webw2a" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print("wrote", out, len(html))
