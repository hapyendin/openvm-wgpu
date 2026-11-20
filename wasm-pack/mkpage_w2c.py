#!/usr/bin/env python3
# Build webw2c/index.html: the W2C WHIR-fold leg IN-BROWSER (Phase 3d).
# 5 instances x per-round: page streams f0/alphas/gcoeffs/grs/finalcoeffs
# via async file:// XHR; wasm runs the CPU fold-plan + GPU DIT-FFT + GPU
# merkle; page verifies gcoeffs-prefix OK (wasm side), grs sha vs
# crypto.subtle of the golden dump, commit roots vs w2c-results-r1-GREEN.json,
# and determinism across 2 passes.
import pathlib
root = pathlib.Path.home() / "appz/openvm-wgpu/wasm-pack"
js = (root / "webw2c/openvm_wasm.js").read_text()
import base64
wasm = base64.b64encode((root / "webw2c/openvm_wasm_bg.wasm").read_bytes()).decode()
html = """<!doctype html><html><head><meta charset=utf-8></head><body>
<pre id=out style="font-size:10px"></pre>
<script>
const B64='WASMB64';
const WHIR='file:///home/matt/appz/openvm-sycl-skel/b9/r15/w2c-r1/whir/';
const GREENURL='file:///home/matt/appz/openvm-wgpu/w2c-results-r1-GREEN.json';
const NAMES=['app-segments_0','agg-leaf_0','agg-internal-for-leaf_0','agg-internal-recursive_0','wrap-internal-recursive.1_0'];
const GOLD={ '1-0':'5b91a9e38bfa47fb633ce4ee564a0175f6f360c1e7d8b0e3f7e607e98547777c',
            '1-1':'28948222729453e28e950f1dd4c6c2435c1d77521cd4dd88266ea521b744c1e8' };
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
  const {txt:greenTxt}=await xhrText(GREENURL);
  if(!greenTxt){log('W2C_STAGE_JSON: FAIL');log('ALL_TESTS_RAN');return;}
  const GREEN=JSON.parse(greenTxt);
  log('W2C_STAGE_JSON: OK');
  log((await wasm_bindgen.w2cSelftest()).trim());
  let ok=true;
  const passLines=[[],[]];
  for(let pass=0;pass<2;pass++){
    for(let ii=0;ii<5;ii++){
      const t0=performance.now();
      const {txt:metaTxt}=await xhrText(WHIR+'w'+ii+'-meta.txt');
      const k=+metaTxt.match(/k_whir=(\\d+)/)[1];
      const lrs=+metaTxt.match(/log_rs0=(\\d+)/)[1];
      const nr=metaTxt.match(/rounds=\\[([^\\]]*)\\]/)[1].split(',').length;
      const f0=await xhrBuf(WHIR+'w'+ii+'-f0.bin');
      const al=await xhrBuf(WHIR+'w'+ii+'-alphas.bin');
      const plan=JSON.parse(await wasm_bindgen.w2cInstBegin(k,lrs,nr,new Uint8Array(f0.buf),new Uint8Array(al.buf)));
      f0.buf=null; al.buf=null;
      const gold=(GREEN.instances[NAMES[ii]]||GREEN[NAMES[ii]]).fold.rounds;
      for(let r=0;r<nr;r++){
        const isfin=(r===nr-1);
        const gcf=isfin?('w'+ii+'-finalcoeffs.bin'):('w'+ii+'-r'+r+'-gcoeffs.bin');
        const gcb=await xhrBuf(WHIR+gcf);
        const prefix=plan.rounds[r].prefix_bytes;
        let gsha='', gbytes=new Uint8Array(0);
        if(!isfin){ const gb=await xhrBuf(WHIR+'w'+ii+'-r'+r+'-grs.bin'); gsha=await shaHex(gb.buf); gbytes=new Uint8Array(gb.buf); }
        const rep=await wasm_bindgen.w2cRound(r,new Uint8Array(gcb.buf.slice(0,prefix)),gbytes);
        gcb.buf=null;
        const m=rep.match(/GCOEFFS_OK (\\S+)/);
        const gs=rep.match(/GRSSHA ([0-9a-f]{64})/);
        const rt=rep.match(/ROOT \\[([^\\]]*)\\]/);
        const gcok=m&&m[1]==='true';
        let grsok=true, rootok=true;
        if(!isfin){
          grsok = gs && gs[1]===gsha && /GRS_OK true/.test(rep);
          if(!grsok) log('W2C_MISMATCH '+rep.replace(/ROUND_MS.*/,''));
          const exp=gold[r]&&gold[r].commit_canon;
          rootok = exp && rt && rt[1]===exp.join(',');
        }
        ok=ok&&gcok&&grsok&&rootok;
        const key=ii+'-'+r;
        if(GOLD[key]&&!isfin){ const g=GOLD[key]===gs[1]; ok=ok&&g; log('W2C_GOLD_SHA '+key+': '+(g?'PASS':'FAIL')); log('W2C_ROW_SHA '+key+' '+gs[1]+' GOLD '+GOLD[key]); }
        const line='i'+ii+' r'+r+' GC'+(gcok?'OK':'BAD')+' GRS'+(grsok?'OK':'BAD')+' ROOT'+(rootok?'OK':'BAD')+(isfin?' FIN':'');
        passLines[pass].push(line);
        log('W2C P'+pass+' '+line+' '+(gs?('GRSSHA '+gs[1].slice(0,16)):'')+' ms='+(performance.now()-t0).toFixed(0));
      }
    }
  }
  const det=JSON.stringify(passLines[0])===JSON.stringify(passLines[1]);
  ok=ok&&det;
  log('W2C_DETERMINISM: '+(det?'PASS':'FAIL'));
  log('W2C_BROWSER: '+(ok?'PASS':'FAIL'));
  log('ALL_TESTS_RAN');
 }catch(e){log('FATAL '+(e&&e.stack?e.stack:e));log('ALL_TESTS_RAN');}
})();
</script>
</body></html>"""
html = html.replace("__JS__", js).replace("WASMB64", wasm)
out = root / "webw2c" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print("wrote", out, len(html))
