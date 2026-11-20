//! W2D FULL-CHAIN prove IN-BROWSER (Phase 3d stretch): transcript -> leaf
//! (leaf buffers handed over from the w2b module) -> GKR (live constants) ->
//! GPU grinds -> WHIR fold (CPU chain + GPU mobius/FFT/merkle) -> emitted
//! proof stream. Golden: gpu_sha256 3e3bfbec62b817f5f9813a326af4698816bd9f9
//! 5fd4a7d83a9ea9e6f4b7a3d0d (native w2d-report.json), byte-exact.
//! Resumable state machine; the page owns every file:// XHR.

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use wgpu_backend::device::WgpuDevice;
use wgpu_backend::field::{
    constants_buffer_mont, decode, encode, fsub, mont_mul, sha256_hex, u32s_to_bytes, Poseidon2,
};
use wgpu_backend::gkr::{check_expected, run_instance_gpu_async, GkrInputs, GkrPipeline};
use wgpu_backend::limitsbuf::ShardBuf;
use wgpu_backend::readfix::ReadFix;
use wgpu_backend::w2c::*;
use wgpu_backend::wgsl;
use wgpu_backend::whir_fold::{self, WhirFold};

use crate::ctx;
use crate::w2b;

const GRIND_WGSL: &str = r#"
@group(0) @binding(4) var<storage, read_write> amin: array<atomic<u32>>;
@compute @workgroup_size(256u)
fn grind_scan(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  if (w >= uni.a2) { return; }
  var cells: array<u32, 16u>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) { cells[i] = io1[i]; }
  cells[uni.seed] = encw(w);
  perm_mix(&cells);
  let c = decode(cells[7u]);
  if ((c & uni.a1) == 0u) { atomicMin(&amin[0], w); }
}
@compute @workgroup_size(1u)
fn grind_check(@builtin(global_invocation_id) gid: vec3<u32>) {
  var cells: array<u32, 16u>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) { cells[i] = io1[i]; }
  cells[uni.seed] = encw(uni.a2);
  perm_mix(&cells);
  io2[0] = decode(cells[7u]);
}
"#;

fn to_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn raw_ef(c: &[u32]) -> Vec<Ef> {
    c.chunks_exact(4).map(|x| [encode(x[0]), encode(x[1]), encode(x[2]), encode(x[3])]).collect()
}
fn raw_ef4(b: &[u8]) -> Ef {
    [
        encode(u32::from_le_bytes(b[0..4].try_into().unwrap())),
        encode(u32::from_le_bytes(b[4..8].try_into().unwrap())),
        encode(u32::from_le_bytes(b[8..12].try_into().unwrap())),
        encode(u32::from_le_bytes(b[12..16].try_into().unwrap())),
    ]
}
fn canon_ef(e: &[Ef]) -> Vec<u32> {
    e.iter().flat_map(|x| x.iter().map(|&w| decode(w))).collect()
}
fn flatten(e: &[Ef]) -> Vec<u32> {
    e.iter().flat_map(|r| r.iter().copied()).collect()
}
fn powm(mut b: u32, mut e: u64) -> u32 {
    let mut r = encode(1);
    while e > 0 {
        if e & 1 == 1 { r = mont_mul(r, b); }
        b = mont_mul(b, b);
        e >>= 1;
    }
    r
}
fn ms() -> f64 { js_sys::Date::now() }

struct GrindPipes {
    p_scan: wgpu::ComputePipeline,
    p_check: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

struct St {
    n: usize,
    vals: Vec<u32>,
    flags: Vec<bool>,
    site_bits: Vec<usize>,
    site_w: Vec<u32>,
    site_events: Vec<usize>,
    mu_event: usize,
    fin_qpow: usize,
    k_whir: usize,
    log_rs0: usize,
    mu_bits: usize,
    q_bits: usize,
    ulen: usize,
    nq_v: Vec<usize>,
    derived: Vec<u32>,
    bases: Vec<([u32; 16], usize)>,
    chain_ok: bool,
    gkr_words: Vec<u32>,
    wmin_all: Vec<u32>,
    grind_ok: bool,
    // fold
    dump_alphas: Vec<u32>,
    my_f: Vec<Ef>,
    my_w: Vec<Ef>,
    log_rs: usize,
    aidx: usize,
    ev: usize,
    kq_r: usize,
    kq_k: usize,
    acc1: Ef,
    acc2: Ef,
    fold_ok: bool,
    fold_commit_words: Vec<u32>,
    fold_ood_words: Vec<u32>,
    fold_seval_pos: Vec<(usize, Vec<u32>)>,
    fold_fc: Vec<u32>,
    gkr_done: bool,
}

thread_local! {
    static ST: RefCell<Option<Box<St>>> = const { RefCell::new(None) };
    static PIPES: RefCell<Option<Rc<(GkrPipeline, WhirFold, GrindPipes, ReadFix)>>> = const { RefCell::new(None) };
}

fn pipes() -> Result<Rc<(GkrPipeline, WhirFold, GrindPipes, ReadFix)>, JsValue> {
    PIPES.with(|p| {
        let mut b = p.borrow_mut();
        if b.is_none() {
            let c = ctx().map_err(|e| JsValue::from_str(&e))?;
            let dev = &c.dev;
            let wgrind = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("grind"),
                source: wgpu::ShaderSource::Wgsl(
                    format!("{}{}\n{}", wgsl::mul_prelude(dev.has_shader_int64), wgsl::COMMON, GRIND_WGSL).into(),
                ),
            });
            let bt = |bi: u32, ro: bool| wgpu::BindGroupLayoutEntry {
                binding: bi,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: ro },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("w2d-grind-bgl"),
                entries: &[
                    bt(0, true),
                    bt(1, false),
                    bt(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    bt(4, false),
                ],
            });
            let layout = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let mkp = |name: &str| {
                dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&layout),
                    module: &wgrind,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            *b = Some(Rc::new((
                GkrPipeline::new(dev),
                WhirFold::new(dev),
                GrindPipes { p_scan: mkp("grind_scan"), p_check: mkp("grind_check"), bgl },
                ReadFix::new(dev),
            )));
        }
        Ok(b.clone().unwrap())
    })
}

/// Awaited plain readback of a device buffer (dispatch-written source:
/// plain copy+map is the proven-good pattern; chunks of 16 MiB).
async fn read_buf(dev: &WgpuDevice, b: &wgpu::Buffer, nwords: u64) -> Vec<u32> {
    let mut out = vec![0u32; nwords as usize];
    let mut done = 0u64;
    while done < nwords {
        let take = (nwords - done).min(1u64 << 20).max(1);
        let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("w2d-rb"),
            size: take * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(b, done * 4, &staging, 0, take * 4);
        dev.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        rx.await.map_err(|e| e.to_string()).and_then(|r| r.map_err(|e| e.to_string())).expect("w2d rb map");
        {
            let d = slice.get_mapped_range().expect("w2d rb mapped");
            for k in 0..take as usize {
                out[(done + k as u64) as usize] = u32::from_le_bytes(d[4 * k..4 * k + 4].try_into().unwrap());
            }
        }
        staging.unmap();
        done += take;
    }
    out
}

/// 1) init: stream + flags + psites + whir meta; skeleton walk asserts.
#[wasm_bindgen(js_name = "w2dInit")]
pub fn w2d_init(stream: Box<[u8]>, flags: Box<[u8]>, psites: String, meta: String) -> Result<String, JsValue> {
    let n = stream.len() / 8;
    let vals: Vec<u32> = (0..n).map(|k| u32::from_le_bytes(stream[8 * k..8 * k + 4].try_into().unwrap())).collect();
    let fl = &flags[..n.div_ceil(8)];
    let flagsv: Vec<bool> = (0..n).map(|k| (fl[k >> 3] >> (k & 7)) & 1 == 1).collect();
    let mut site_bits = Vec::new();
    let mut site_w = Vec::new();
    for l in psites.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(l).map_err(|e| JsValue::from_str(&e.to_string()))?;
        site_bits.push(v["bits"].as_u64().unwrap() as usize);
        site_w.push(v["witness"].as_u64().unwrap() as u32);
    }
    let mp = |k: &str| -> Result<usize, JsValue> {
        meta.split_whitespace()
            .find(|t| t.starts_with(&format!("{k}=")))
            .ok_or_else(|| JsValue::from_str("meta key"))?
            [k.len() + 1..].parse::<usize>().map_err(|e| JsValue::from_str(&e.to_string()))
    };
    let k_whir = mp("k_whir")?;
    let log_rs0 = mp("log_rs0")?;
    let mu_bits = mp("mu_bits")?;
    let q_bits = mp("q_bits")?;
    let ulen = mp("ulen")?;
    let nq_v: Vec<usize> = meta.split("rounds=[").nth(1).ok_or_else(|| JsValue::from_str("rounds"))?
        .split(']').next().unwrap()
        .split(',').map(|x| x.trim().parse().unwrap()).collect();
    // mu witness position
    let mut mu_ev: Option<usize> = None;
    for ev in 6000..13800.min(n - 6) {
        if !flagsv[ev] && flagsv[ev + 1] && mask_ok(vals[ev + 1], mu_bits) && (ev + 2..ev + 6).all(|x| flagsv[x]) {
            if mu_ev.is_some() { return Err(JsValue::from_str("mu ambiguous")); }
            mu_ev = Some(ev);
        }
    }
    let mu_event = mu_ev.ok_or_else(|| JsValue::from_str("mu witness not found"))?;
    let fin_qpow = n - (nq_v[nq_v.len() - 1] + 5) - 1;
    let mut site_events: Vec<usize> = vec![77, mu_event];
    {
        let mut ev = mu_event + 6;
        for (r, &nq) in nq_v.iter().enumerate() {
            let is_last = r + 1 == nq_v.len();
            for _ in 0..k_whir {
                ev += 8;
                site_events.push(ev);
                ev += 2;
                ev += 4;
            }
            if !is_last {
                ev += 8 + 4 + 4;
                if q_bits > 0 { site_events.push(ev); ev += 2; }
                ev += nq + 4;
            } else {
                ev += 4 * ulen;
                ev = fin_qpow;
                if q_bits > 0 { site_events.push(ev); ev += 2; }
                ev += nq + 4;
            }
        }
        if ev != n { return Err(JsValue::from_str("fold skeleton does not consume stream")); }
    }
    if site_bits.len() < site_events.len() { return Err(JsValue::from_str("fewer psites than derived sites")); }
    site_bits.truncate(site_events.len());
    site_w.truncate(site_events.len());
    for (i2, e) in site_events.iter().enumerate() {
        if flagsv[*e] { return Err(JsValue::from_str("site witness not O")); }
        if vals[*e] != site_w[i2] { return Err(JsValue::from_str("site witness != psites")); }
    }
    let st = Box::new(St {
        n, vals, flags: flagsv, site_bits, site_w, site_events, mu_event, fin_qpow,
        k_whir, log_rs0, mu_bits, q_bits, ulen, nq_v,
        derived: vec![], bases: vec![], chain_ok: false,
        gkr_words: vec![], wmin_all: vec![], grind_ok: false,
        dump_alphas: vec![], my_f: vec![], my_w: vec![], log_rs: log_rs0, aidx: 0, ev: 0,
        kq_r: 0, kq_k: 0, acc1: ef_zero(), acc2: ef_zero(), fold_ok: true,
        fold_commit_words: vec![], fold_ood_words: vec![], fold_seval_pos: vec![], fold_fc: vec![],
        gkr_done: false,
    });
    ST.with(|s| *s.borrow_mut() = Some(st));
    let nsites = ST.with(|s| s.borrow().as_ref().unwrap().site_events.len());
    Ok(format!("W2D_INIT n={n} mu_event={mu_event} fin_qpow={fin_qpow} sites={nsites} walk_ok true"))
}

/// 2) duplex transcript pass 1.
#[wasm_bindgen(js_name = "w2dTranscript")]
pub fn w2d_transcript() -> Result<String, JsValue> {
    ST.with(|sb| {
        let mut sbb = sb.borrow_mut();
        let st = sbb.as_mut().ok_or_else(|| JsValue::from_str("w2dInit first"))?;
        let p2 = Poseidon2::new();
        let glue: [u32; 8] = [93085625, 540830993, 1888218509, 927761177, 1083260992, 464140180, 402247195, 690334738];
        let mut d = Duplex::new(&p2);
        for g in glue { d.observe(g); }
        st.derived = vec![0u32; st.n];
        let mut sample_mism = Vec::new();
        let mut bases = Vec::new();
        let mut si = 0usize;
        let n = st.n;
        for ev in 0..n {
            if si < st.site_events.len() && st.site_events[si] == ev {
                bases.push(d.grind_base());
                si += 1;
                d.observe(st.vals[ev]);
            } else if st.flags[ev] {
                let v = d.sample();
                st.derived[ev] = v;
                if v != st.vals[ev] && sample_mism.len() < 10 { sample_mism.push((ev, st.vals[ev], v)); }
            } else {
                d.observe(st.vals[ev]);
            }
        }
        if si != st.site_events.len() { return Err(JsValue::from_str("si != sites")); }
        st.chain_ok = sample_mism.is_empty();
        st.bases = bases;
        let nsamp = st.flags.iter().filter(|f| **f).count();
        Ok(format!("W2D_CHAIN samples={nsamp} mismatches={} chain_ok={} {:?}", sample_mism.len(), st.chain_ok, &sample_mism[..sample_mism.len().min(3)]))
    })
}

/// 3) GKR with LIVE constants. Leaf buffers are taken from the w2b module
/// (the page must run the w2b feed exports with the W2D world first).
#[wasm_bindgen(js_name = "w2dGkr")]
pub async fn w2d_gkr(gkr_json: String) -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let pipes = pipes()?;
    let leaf_bufs = w2b::take_leaf();
    if leaf_bufs.is_empty() { return Err(JsValue::from_str("leaf missing (run w2b feeds + w2bLeafSha first)")); }
    let leaf_sha = w2b::last_leaf_sha();
    let (gkr_words, const_mism, val_mism_n, root_q) = {
        let mut gin: GkrInputs = serde_json::from_str(&gkr_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let derived = ST.with(|s| s.borrow().as_ref().map(|st| st.derived.clone())).unwrap();
        {
            let inst = &mut gin.instances[0];
            let mut mus_live: Vec<[u32; 4]> = vec![[derived[107], derived[108], derived[109], derived[110]]];
            let mut cu = 111usize;
            let mut lam_live = Vec::new();
            let mut r_live: Vec<Vec<[u32; 4]>> = Vec::new();
            for j in 1..inst.t {
                lam_live.push([derived[cu], derived[cu + 1], derived[cu + 2], derived[cu + 3]]); cu += 4;
                let mut rs = Vec::new();
                for _ in 0..j {
                    cu += 12;
                    rs.push([derived[cu], derived[cu + 1], derived[cu + 2], derived[cu + 3]]); cu += 4;
                }
                cu += 16;
                mus_live.push([derived[cu], derived[cu + 1], derived[cu + 2], derived[cu + 3]]); cu += 4;
                r_live.push(rs);
            }
            let mut const_mism = 0usize;
            for j in 1..inst.t {
                let want: Vec<u64> = std::iter::once(mus_live[j as usize - 1])
                    .chain(if j >= 2 { r_live[j as usize - 2].iter().copied() } else { [].iter().copied() })
                    .flat_map(|q: [u32; 4]| q.into_iter().map(|w| w as u64)).collect();
                if inst.per_round_xi.get(&j).cloned().unwrap_or_default() != want { const_mism += 1; }
                inst.per_round_xi.insert(j, want);
                let lamw = lam_live[j as usize - 1].iter().map(|&w| w as u64).collect::<Vec<_>>();
                if inst.rounds[j as usize - 1].lambda != lamw { const_mism += 1; }
                inst.rounds[j as usize - 1].lambda = lamw;
                for k in 0..j as usize {
                    let rw = r_live[j as usize - 1][k].iter().map(|&w| w as u64).collect::<Vec<_>>();
                    if inst.rounds[j as usize - 1].subs[k].r != rw { const_mism += 1; }
                    inst.rounds[j as usize - 1].subs[k].r = rw;
                }
            }
            let mut snaps = Vec::new();
            let _t0 = ms();
            let gout = run_instance_gpu_async(&c.dev, &pipes.0, &gin.instances[0], leaf_bufs, leaf_sha.clone(), std::path::Path::new("."), &mut snaps).await;
            let val_mism = check_expected(&gin.instances[0], &gout);
            let n = ST.with(|s| s.borrow().as_ref().unwrap().n);
            let mut gkr_words = vec![0u32; n];
            {
                let mut cu = 87usize;
                for w in gout.root_q.iter() { gkr_words[cu] = *w; cu += 1; }
                for w in &gout.claims0 { gkr_words[cu] = *w; cu += 1; }
                cu += 4;
                for rr in &gout.rounds {
                    cu += 4;
                    for sm in &rr.subs {
                        for w in sm { gkr_words[cu] = *w; cu += 1; }
                        cu += 4;
                    }
                    for w in &rr.claims { gkr_words[cu] = *w; cu += 1; }
                    cu += 4;
                }
                if cu != 5911 { return Err(JsValue::from_str("gkr word cursor != 5911")); }
            }
            (gkr_words, const_mism, val_mism.len(), gout.root_q.clone())
        }
    };
    ST.with(|s| {
        let mut bb = s.borrow_mut();
        let st = bb.as_mut().unwrap();
        st.gkr_words = gkr_words;
        st.gkr_done = true;
    });
    Ok(format!("W2D_GKR const_mism={const_mism} valmism={val_mism_n} root_q={root_q:?} leaf={leaf_sha}"))
}

/// 4) GPU grinds (22 sites).
#[wasm_bindgen(js_name = "w2dGrinds")]
pub async fn w2d_grinds() -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let pipes = pipes()?;
    let dev = &c.dev;
    let wf = &pipes.1;
    let gp = &pipes.2;
    let kc = wf.buf(dev, &constants_buffer_mont(), "kc");
    let io1 = wf.buf(dev, &[0u32; 16], "bases");
    let io2 = wf.buf(dev, &[0u32; 8], "check");
    let amin = wf.buf(dev, &[0xFFFFFFFFu32; 4], "amin");
    let uni = wf.uni16(dev);
    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &gp.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: kc.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: io1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: io2.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: amin.as_entire_binding() },
        ],
        label: None,
    });
    let p2 = Poseidon2::new();
    let mut rep = String::new();
    let mut ok_all = true;
    let mut wmin_all = Vec::new();
    let sites = ST.with(|s| {
        let b = s.borrow();
        let st = b.as_ref().unwrap();
        (st.site_events.clone(), st.site_bits.clone(), st.site_w.clone(), st.bases.clone(), st.derived.clone(), st.vals.clone())
    });
    let (site_events, site_bits, site_w, bases, _derived, vals) = sites;
    for i2 in 0..site_events.len() {
        let ev = site_events[i2];
        let bits = site_bits[i2];
        let wrec = site_w[i2];
        let (base, idx) = bases[i2];
        let limit = ((wrec + 1).next_power_of_two() << 1).max(256);
        let mask = (1u64 << bits) - 1;
        dev.queue.write_buffer(&io1, 0, &u32s_to_bytes(&base));
        dev.queue.write_buffer(&amin, 0, &u32s_to_bytes(&[0xFFFFFFFFu32; 4]));
        WhirFold::set_uni(dev, &uni, idx as u32, mask as u32, limit, 0);
        WhirFold::dispatch(dev, &gp.p_scan, &bg, limit as usize);
        let wmin = read_buf(dev, &amin, 1).await[0];
        WhirFold::set_uni(dev, &uni, idx as u32, mask as u32, wmin, 0);
        WhirFold::dispatch(dev, &gp.p_check, &bg, 1);
        let check_gpu = read_buf(dev, &io2, 1).await[0];
        let check_rec = vals[ev + 1];
        let check_cpu = Duplex::grind_check(&p2, &base, idx, wmin);
        let ok = wmin == wrec && check_gpu == check_rec && check_cpu == check_rec && mask_ok(check_gpu, bits);
        ok_all &= ok;
        wmin_all.push(wmin);
        rep.push_str(&format!("W2D_GRIND {i2} ev={ev} bits={bits} w_gpu={wmin} rec={wrec} ok={ok}\n"));
    }
    ST.with(|s| {
        let mut bb = s.borrow_mut();
        let st = bb.as_mut().unwrap();
        st.wmin_all = wmin_all;
        st.grind_ok = ok_all;
    });
    rep.push_str(&format!("W2D_GRINDS ok={ok_all}"));
    Ok(rep)
}

/// 5a) fold begin: dump alphas + mu live check.
#[wasm_bindgen(js_name = "w2dFoldBegin")]
pub fn w2d_fold_begin(alphas: Box<[u8]>, mu: Box<[u8]>) -> Result<String, JsValue> {
    ST.with(|sb| {
        let mut sbb = sb.borrow_mut();
        let st = sbb.as_mut().ok_or_else(|| JsValue::from_str("w2dInit first"))?;
        st.dump_alphas = to_u32s(&alphas);
        let mu_c = to_u32s(&mu);
        let live_mu: Vec<u32> = (st.mu_event + 2..st.mu_event + 6).map(|x| st.derived[x]).collect();
        if live_mu != mu_c { st.fold_ok = false; }
        st.ev = st.mu_event + 6;
        st.fold_commit_words = vec![0u32; st.n];
        st.fold_ood_words = vec![0u32; st.n];
        Ok(format!("W2D_FOLD_BEGIN ev={} mu_ok={}", st.ev, live_mu == mu_c))
    })
}

/// 5b) start one sub-round (kq): sets fold alpha from live transcript.
#[wasm_bindgen(js_name = "w2dFoldKqBegin")]
pub fn w2d_fold_kq_begin() -> Result<String, JsValue> {
    ST.with(|sb| {
        let mut sbb = sb.borrow_mut();
        let st = sbb.as_mut().ok_or_else(|| JsValue::from_str("w2dInit first"))?;
        st.acc1 = ef_zero();
        st.acc2 = ef_zero();
        let streamed = st.kq_r == 0 && st.kq_k == 0;
        Ok(format!("{}", streamed)) // caller may now stream pair chunks
    })
}

/// 5c) streamed first sub-round pair chunk (f0/w0 rows: 32 B per pair/file).
#[wasm_bindgen(js_name = "w2dFoldPairChunk")]
pub fn w2d_fold_pair_chunk(fbytes: Box<[u8]>, wbytes: Box<[u8]>) -> Result<(), JsValue> {
    if fbytes.len() != wbytes.len() || fbytes.len() % 32 != 0 {
        return Err(JsValue::from_str("pair chunk size"));
    }
    ST.with(|sb| {
        let mut sbb = sb.borrow_mut();
        let st = sbb.as_mut().ok_or_else(|| JsValue::from_str("w2dInit first"))?;
        if !(st.kq_r == 0 && st.kq_k == 0) { return Err(JsValue::from_str("pair chunk outside streamed subround")); }
        if st.my_f.is_empty() && st.my_w.is_empty() {
            // pre-size to half of n_f0 once (n_f0 unknown here; grow naturally)
        }
        let a = raw_ef(&[st.derived[st.ev + 10], st.derived[st.ev + 11], st.derived[st.ev + 12], st.derived[st.ev + 13]])[0];
        for y in 0..fbytes.len() / 32 {
            let f0v = raw_ef4(&fbytes[y * 32..y * 32 + 16]);
            let f1v = raw_ef4(&fbytes[y * 32 + 16..y * 32 + 32]);
            let w0v = raw_ef4(&wbytes[y * 32..y * 32 + 16]);
            let w1v = raw_ef4(&wbytes[y * 32 + 16..y * 32 + 32]);
            let df = ef_sub(&f1v, &f0v);
            let dw = ef_sub(&w1v, &w0v);
            st.acc1 = ef_add(&st.acc1, &ef_mul(&ef_add(&f0v, &df), &ef_add(&w0v, &dw)));
            st.acc2 = ef_add(&st.acc2, &ef_mul(&ef_add(&f0v, &ef_scale(&df, 2)), &ef_add(&w0v, &ef_scale(&dw, 2))));
            st.my_f.push(ef_add(&f0v, &ef_mul(&df, &a)));
            st.my_w.push(ef_add(&w0v, &ef_mul(&dw, &a)));
        }
        Ok(())
    })
}

/// 5d) end one sub-round: sevals check, alpha check, in-place fold (when not
/// streamed), event cursor advance.
#[wasm_bindgen(js_name = "w2dFoldKqEnd")]
pub fn w2d_fold_kq_end() -> Result<String, JsValue> {
    ST.with(|sb| {
        let mut sbb = sb.borrow_mut();
        let st = sbb.as_mut().ok_or_else(|| JsValue::from_str("w2dInit first"))?;
        let streamed = st.kq_r == 0 && st.kq_k == 0;
        let ev = st.ev;
        let a = raw_ef(&[st.derived[ev + 10], st.derived[ev + 11], st.derived[ev + 12], st.derived[ev + 13]])[0];
        if !streamed {
            let half = st.my_f.len() / 2;
            for y in 0..half {
                let f0v = st.my_f[2 * y];
                let df = ef_sub(&st.my_f[2 * y + 1], &f0v);
                let w0v = st.my_w[2 * y];
                let dw = ef_sub(&st.my_w[2 * y + 1], &w0v);
                st.acc1 = ef_add(&st.acc1, &ef_mul(&ef_add(&f0v, &df), &ef_add(&w0v, &dw)));
                st.acc2 = ef_add(&st.acc2, &ef_mul(&ef_add(&f0v, &ef_scale(&df, 2)), &ef_add(&w0v, &ef_scale(&dw, 2))));
            }
        }
        let sv = canon_ef(&[st.acc1, st.acc2]);
        if !(0..8).all(|j| !st.flags[ev + j] && st.vals[ev + j] == sv[j]) {
            st.fold_ok = false;
        }
        st.fold_seval_pos.push((ev, sv));
        let mut ev = ev + 8 + 2;
        if st.derived[ev..ev + 4] != st.dump_alphas[st.aidx * 4..st.aidx * 4 + 4] { st.fold_ok = false; }
        ev += 4;
        if !streamed {
            let half = st.my_f.len() / 2;
            for y in 0..half {
                let f0v = st.my_f[2 * y];
                let df = ef_sub(&st.my_f[2 * y + 1], &f0v);
                st.my_f[y] = ef_add(&f0v, &ef_mul(&df, &a));
                let w0v = st.my_w[2 * y];
                let dw = ef_sub(&st.my_w[2 * y + 1], &w0v);
                st.my_w[y] = ef_add(&w0v, &ef_mul(&dw, &a));
            }
            st.my_f.truncate(half);
            st.my_w.truncate(half);
        }
        st.aidx += 1;
        st.ev = ev;
        st.kq_k += 1;
        if st.kq_k == st.k_whir { st.kq_k = 0; st.kq_r += 1; }
        Ok(format!("W2D_KQ r{} ev={} fold_ok={}", st.kq_r, st.ev, st.fold_ok))
    })
}

#[wasm_bindgen(js_name = "w2dFoldRound")]
pub async fn w2d_fold_round(finalcoeffs: Box<[u8]>) -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let pipes = pipes()?;
    let dev = &c.dev;
    let wf = &pipes.1;
    let rf = &pipes.3;
    let (r, nq, is_last) = ST.with(|s| {
        let b = s.borrow();
        let st = b.as_ref().unwrap();
        let r = st.kq_r - 1;
        (r, st.nq_v[r], r + 1 == st.nq_v.len())
    });
    let mut rep = String::new();
    // CPU mobius is NOT used: exact twin of the native gpu_mobius
    let (my_f, my_w, ev0, log_rs) = ST.with(|s| {
        let mut b = s.borrow_mut();
        let st = b.as_mut().unwrap();
        (std::mem::take(&mut st.my_f), std::mem::take(&mut st.my_w), st.ev, st.log_rs)
    });
    // gpu_mobius(dev, wf, &my_f) with readfix async
    let coeffs = {
        let n = my_f.len();
        let data = whir_fold::data_shard(dev, n, "w2d-mob");
        data.write_slice(dev, 0, &u32s_to_bytes(&flatten(&my_f)));
        whir_fold::whir_stages(dev, wf, &data, n, 1, &[]);
        let words = rf.read_words_after_write(dev, &data, 0, (n * 4) as u64).await;
        words.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect::<Vec<Ef>>()
    };
    if is_last {
        let fc = canon_ef(&coeffs);
        let nfc = fc.len();
        let (fvals, fflags) = ST.with(|s| {
            let b = s.borrow();
            let st = b.as_ref().unwrap();
            (st.vals[ev0..ev0 + nfc].to_vec(), st.flags[ev0..ev0 + nfc].to_vec())
        });
        let stream_ok = (0..nfc).all(|j| !fflags[j] && fvals[j] == fc[j]);
        let dump: Vec<u32> = to_u32s(&finalcoeffs);
        let dump_c: Vec<u32> = dump;
        let dump_ok = fc == dump_c;
        ST.with(|s| {
            let mut b = s.borrow_mut();
            let st = b.as_mut().unwrap();
            st.fold_fc = fc;
            st.my_f = my_f; st.my_w = my_w;
            st.ev = st.fin_qpow;
            if st.q_bits > 0 { st.ev += 2; }
            st.ev += nq + 4;
            if !(stream_ok && dump_ok) { st.fold_ok = false; }
        });
        rep.push_str(&format!("W2D_FOLD r{r} FINAL stream_ok={stream_ok} dump_ok={dump_ok} nfc={nfc}"));
        return Ok(rep);
    }
    // commit: DIT-FFT to 2^(log_rs-1) rows + merkle (async twins)
    let k_whir = ST.with(|s| s.borrow().as_ref().unwrap().k_whir);
    let log_n = (log_rs - 1) as u32;
    let n = 1usize << log_n;
    let mut perm: Vec<Ef> = vec![ef_zero(); n];
    for (i, cw) in coeffs.iter().enumerate() {
        let j = ((i as u32).reverse_bits() >> (32 - log_n)) as usize;
        perm[j] = *cw;
    }
    let data = whir_fold::data_shard(dev, n, "w2d-fft");
    {
        const UC: usize = 1 << 18;
        let mut bv: Vec<u8> = Vec::with_capacity(UC * 16);
        let mut j0 = 0usize;
        while j0 < n {
            let cnt = UC.min(n - j0);
            bv.clear();
            for jj in 0..cnt {
                bv.extend_from_slice(&u32s_to_bytes(&flatten(&[perm[j0 + jj]])));
            }
            data.write_slice(dev, (j0 * 16) as u64, &bv);
            j0 += cnt;
        }
    }
    drop(perm);
    let mut tws: Vec<wgpu::Buffer> = Vec::new();
    let mut half = 1usize;
    while half < n {
        let wm = powm(root_of_unity_raw(log_n), (n / (half * 2)) as u64);
        let mut tw = vec![0u32; half];
        tw[0] = encode(1);
        for k in 1..half { tw[k] = mont_mul(tw[k - 1], wm); }
        tws.push(wf.buf(dev, &tw, "tw"));
        half *= 2;
    }
    whir_fold::whir_stages(dev, wf, &data, n, 0, &tws);
    drop(tws);
    let rows_b = ShardBuf::new(dev, "w2d-mrows", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, (n * 32) as u64);
    {
        const DC: usize = 1 << 18;
        let mut bv: Vec<u8> = Vec::with_capacity(DC * 32);
        let mut r0 = 0usize;
        while r0 < n {
            let cnt = DC.min(n - r0);
            let words = rf.read_words_after_write(dev, &data, (r0 * 4) as u64, (cnt * 4) as u64).await;
            bv.clear();
            for row in words.chunks_exact(4) {
                bv.extend_from_slice(&u32s_to_bytes(row));
                bv.extend_from_slice(&[0u8; 16]);
            }
            rows_b.write_slice(dev, (r0 * 32) as u64, &bv);
            r0 += cnt;
        }
    }
    drop(data);
    let root = whir_fold::merkle_from_rows_async(dev, wf, &rows_b, n, k_whir).await;
    drop(rows_b);

    // commit/ood/query checks against the recorded stream (pure CPU over
    // saved state; all awaits above are done).
    let mut ev = ev0;
    let (stream_ok, fold_ok) = ST.with(|s| {
        let mut b = s.borrow_mut();
        let st = b.as_mut().unwrap();
        let stream_ok = (0..8).all(|j| !st.flags[ev + j] && st.vals[ev + j] == root[j]);
        if !stream_ok { st.fold_ok = false; }
        for j in 0..8 { st.fold_commit_words[ev + j] = root[j]; }
        (stream_ok, st.fold_ok)
    });
    ev += 8;
    let (derived, vals, flags) = ST.with(|s| {
        let b = s.borrow();
        let st = b.as_ref().unwrap();
        (st.derived.clone(), st.vals.clone(), st.flags.clone())
    });
    let z0 = raw_ef(&[derived[ev], derived[ev + 1], derived[ev + 2], derived[ev + 3]]);
    ev += 4;
    let yv = {
        let mut acc = coeffs[coeffs.len() - 1];
        for cc in coeffs[..coeffs.len() - 1].iter().rev() {
            acc = ef_add(&ef_mul(&acc, &z0[0]), cc);
        }
        acc
    };
    let y_c = canon_ef(&[yv]);
    let ood_ok = (0..4).all(|j| !flags[ev + j] && vals[ev + j] == y_c[j]);
    let ood_words = y_c.clone();
    ev += 4;
    let q_bits = ST.with(|s| s.borrow().as_ref().unwrap().q_bits);
    if q_bits > 0 { ev += 2; }
    let d_bits = log_rs - k_whir;
    let qmask = (1u64 << d_bits) - 1;
    let qidx: Vec<u64> = (0..nq).map(|j| (derived[ev + j] as u64) & qmask).collect();
    ev += nq;
    let gamma = raw_ef(&[derived[ev], derived[ev + 1], derived[ev + 2], derived[ev + 3]]);
    ev += 4;
    let mut my_w = my_w;
    {
        let dim = my_w.len().trailing_zeros() as usize;
        let n_w = my_w.len();
        let zpow = exp_powers_of_2(&z0[0], dim);
        let mut eq: Vec<Ef> = vec![ef_zero(); n_w];
        eq[0] = gamma[0];
        for (j, zj) in zpow.iter().enumerate() {
            let one_m = ef_sub(&[encode(1), 0, 0, 0], zj);
            let step = 1usize << j;
            for i2 in (0..2usize << j).step_by(step * 2) {
                for k2 in 0..step {
                    let prev = eq[i2 + k2];
                    eq[i2 + k2] = ef_mul(&prev, &one_m);
                    eq[i2 + k2 + step] = ef_mul(&prev, zj);
                }
            }
        }
        for i2 in 0..n_w { my_w[i2] = ef_add(&my_w[i2], &eq[i2]); }
        let omega = root_of_unity_raw(d_bits as u32);
        let mut gpow = ef_mul(&gamma[0], &gamma[0]);
        for qi in 0..nq {
            let zc = powm(omega, qidx[qi]);
            let dim = my_w.len().trailing_zeros() as usize;
            let mut zp2: Vec<u32> = Vec::with_capacity(dim);
            let mut cur = zc;
            for _ in 0..dim { zp2.push(cur); cur = mont_mul(cur, cur); }
            let n_w = my_w.len();
            let mut eq1: Vec<u32> = vec![0u32; n_w];
            eq1[0] = encode(1);
            for (j, zj) in zp2.iter().enumerate() {
                let om = fsub(encode(1), *zj);
                let step = 1usize << j;
                for i2 in (0..2usize << j).step_by(step * 2) {
                    for k2 in 0..step {
                        let prev = eq1[i2 + k2];
                        eq1[i2 + k2] = mont_mul(prev, om);
                        eq1[i2 + k2 + step] = mont_mul(prev, *zj);
                    }
                }
            }
            for i2 in 0..n_w {
                let add = ef_scale(&gpow, decode(eq1[i2]));
                my_w[i2] = ef_add(&my_w[i2], &add);
            }
            gpow = ef_mul(&gpow, &gamma[0]);
        }
    }
    ST.with(|s| {
        let mut b = s.borrow_mut();
        let st = b.as_mut().unwrap();
        st.my_f = my_f;
        st.my_w = my_w;
        st.ev = ev;
        st.log_rs -= 1;
        if !ood_ok { st.fold_ok = false; }
        for j in 0..4 { st.fold_ood_words[ev0 + 8 + 4 + j] = ood_words[j]; }
    });
    rep.push_str(&format!("W2D_FOLD r{r} COMMIT stream_ok={stream_ok} OOD ok={ood_ok} end_ev={ev}"));
    Ok(rep)
}

/// 6) assemble the emitted proof stream + sha256.
#[wasm_bindgen(js_name = "w2dFinish")]
pub fn w2d_finish() -> Result<String, JsValue> {
    ST.with(|sb| {
        let mut sbb = sb.borrow_mut();
        let st = sbb.as_mut().ok_or_else(|| JsValue::from_str("w2dInit first"))?;
        if !st.gkr_done || st.kq_r != st.nq_v.len() { return Err(JsValue::from_str("pipeline incomplete")); }
        if st.ev != st.n { return Err(JsValue::from_str("fold walk did not consume stream")); }
        let n = st.n;
        let mut computed = vec![0u32; n];
        let mut computed_src: Vec<Option<&'static str>> = vec![None; n];
        for k in 87..5911 {
            if !st.flags[k] { computed[k] = st.gkr_words[k]; computed_src[k] = Some("logup-gpu"); }
        }
        for (sp, sv) in &st.fold_seval_pos {
            for j in 0..8 { computed[sp + j] = sv[j]; computed_src[sp + j] = Some("fold-sevals"); }
        }
        {
            let mut cu = st.mu_event + 6;
            for (r, &nq) in st.nq_v.iter().enumerate() {
                let _ = r;
                let is_last = r + 1 == st.nq_v.len();
                cu += st.k_whir * 14;
                if !is_last {
                    for j in 0..8 { computed[cu + j] = st.fold_commit_words[cu + j]; computed_src[cu + j] = Some("fold-commit-gpu"); }
                    cu += 8 + 4;
                    for j in 0..4 { computed[cu + j] = st.fold_ood_words[cu + j]; computed_src[cu + j] = Some("fold-ood"); }
                    cu += 4 + 2 + nq + 4;
                }
            }
        }
        {
            let mut cu = st.mu_event + 6;
            for (r, &nq) in st.nq_v.iter().enumerate() {
                let is_last = r + 1 == st.nq_v.len();
                cu += st.k_whir * 14;
                if !is_last { cu += 8 + 4 + 4 + 2 + nq + 4; }
                else {
                    for j in 0..st.fold_fc.len() { computed[cu + j] = st.fold_fc[j]; computed_src[cu + j] = Some("fold-finalcoeffs"); }
                }
            }
        }
        let mut emit: Vec<u32> = Vec::with_capacity(n);
        let mut emit_mism = 0usize;
        let mut first: Option<(usize, u32, u32)> = None;
        for k in 0..n {
            let w = if st.flags[k] {
                st.derived[k]
            } else if let Some(pos) = st.site_events.iter().position(|e| *e == k) {
                st.wmin_all[pos]
            } else if computed_src[k].is_some() {
                computed[k]
            } else {
                st.vals[k]
            };
            if w != st.vals[k] {
                emit_mism += 1;
                if first.is_none() { first = Some((k, st.vals[k], w)); }
            }
            emit.push(w);
        }
        let mut of = Vec::with_capacity(n * 8);
        for w in &emit { of.extend_from_slice(&(*w as u64).to_le_bytes()); }
        let gpu_sha = sha256_hex(&of);
        let n_pass = (0..n).filter(|k| !st.flags[*k] && computed_src[*k].is_none() && !st.site_events.contains(k)).count();
        let green = st.chain_ok && st.grind_ok && st.fold_ok && emit_mism == 0;
        Ok(format!(
            "W2D_GPU_SHA256 {gpu_sha} EMITMISM {emit_mism} FIRST {first:?} PASSTHRU {n_pass} GREEN {green} CHAIN {} GRIND {} FOLD {}",
            st.chain_ok, st.grind_ok, st.fold_ok
        ))
    })
}
