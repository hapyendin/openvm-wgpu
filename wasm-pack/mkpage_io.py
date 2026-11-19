#!/usr/bin/env python3
# Build webio/index.html: w2a-stretch I/O wall probe.
import base64, pathlib
root = pathlib.Path.home() / "appz/openvm-wgpu/wasm-pack"
wasm = base64.b64encode((root / "webnm/openvm_wasm_bg.wasm").read_bytes()).decode()
js = (root / "webnm/openvm_wasm.js").read_text()
html = """<!doctype html><html><head><meta charset=utf-8></head><body>
<pre id=out>probe</pre>
<script>
const B64='WASMB64';
function log(s){const e=document.getElementById('out');e.textContent+='\\n'+s;console.log(s);}
async function xhr(file){
  const t0=performance.now();
  const r=await new Promise((res,rej)=>{
    const x=new XMLHttpRequest();
    x.open('GET',file,true); x.responseType='arraybuffer';
    x.onload=()=>res(x.status===200?x.response:null);
    x.onerror=()=>rej('xhr-err '+file);
    x.send();
  });
  const ms=performance.now()-t0;
  return {ms, bytes: r?r.byteLength:-1, buf: r};
}
</script>
<script>__JS__</script>
<script>
function b64ToBytes(b){const bin=atob(b);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
(async()=>{
 try{
  await wasm_bindgen(new Response(b64ToBytes(B64).buffer,{headers:{'Content-Type':'application/wasm'}}));
  log('INIT: OK');
  let adapter=null;
  for(let i=0;i<20 && !adapter;i++){
    adapter = await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    if(!adapter) await new Promise(r=>setTimeout(r,500));
  }
  log('JS_ADAPTER '+!!adapter);
  await wasm_bindgen.initWgpu();
  log('WGPU: OK');
  const files=['/tmp/ioprobe/f4k.bin','/tmp/ioprobe/f1m.bin','/tmp/ioprobe/f16m.bin','/tmp/ioprobe/f64m.bin','/tmp/ioprobe/f256m.bin','/tmp/ioprobe/f1024m.bin'];
  let keep=null;
  for(const f of files){
    try{
      const {ms,bytes,buf}=await xhr('file://'+f);
      log('XHR '+f+' bytes='+bytes+' ms='+ms.toFixed(1));
      if(bytes===64*1024*1024) keep=new Uint8Array(buf);
    }catch(e){log('XHR '+f+' FAIL '+e);}
  }
  if(keep){
    log('KEEP head_bytes='+[...keep.slice(0,8)].join(','));
    log('IOSMALL '+await wasm_bindgen.ioProbe(new Uint8Array([1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0])));
    log('IOPROBE '+await wasm_bindgen.ioProbe(keep));
    // second upload to see steady-state (GC'd previous buffers)
    log('IOPROBE2 '+await wasm_bindgen.ioProbe(keep));
  }else{log('IOPROBE: skipped (no 64MB buffer)');}
  log('ALL_TESTS_RAN');
 }catch(e){log('FATAL '+(e&&e.message?e.message:e));}
})();
</script></body></html>"""
html = html.replace("__JS__", js).replace("WASMB64", wasm)
out = root / "webio" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print("wrote", out, len(html))
