//! openvm-wasm: browser tier (Phase 3) for the OpenVM wgpu/WGSL backend.
//!
//! Exposes async init(), u32mul_selftest(), gate_c1_gpu() to JS.
//! Reuses the native lib's WGSL sources, pipeline/bind machinery and
//! ShardBuf windowing verbatim; the ONLY browser-specific machinery is
//! await-based readback (map_async callback -> oneshot -> await), since
//! poll()/pollster don't exist on the JS event loop.
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

mod atlas;
mod gatefast;
mod grind;
mod w2a;
mod w2b;
mod w2c_fold;
mod w2d;

use wgpu_backend::device::WgpuDevice;
use crate::atlas::ReadBatch;
use wgpu::BufferUsages as BU;
use wgpu_backend::field::{
    decode, encode, parse_vectors, sha256_hex, to_hex, u32s_to_bytes, Poseidon2, MERKLE_LEAVES,
    MERKLE_ROOT_HEX, MERKLE_ROOT_SHA256, MERKLE_SEED, P,
};
use wgpu_backend::limitsbuf::ShardBuf;
use wgpu_backend::pipeline::{dispatch, dispatch_ntt, KernelCtx, NttPipelineSet, PipelineSet};

/// golden perm vectors: the SAME file the native gate_c1 bin embeds.
const VEC_SRC: &str = include_str!("../../wgpu-backend/goldens/vectors.inc");

pub(crate) struct Ctx {
    dev: WgpuDevice,
    ps: PipelineSet,
    kctx: KernelCtx,
    ntt_ps: NttPipelineSet,
    dummy: wgpu::Buffer,
}

thread_local! {
    static CTX: RefCell<Option<Rc<Ctx>>> = const { RefCell::new(None) };
}

pub(crate) fn ctx() -> Result<Rc<Ctx>, String> {
    CTX.with(|c| c.borrow().clone()).ok_or_else(|| "call init() first".to_string())
}

// ---- async readback (the browser-only leg) ----

/// Copy [off, off+len) of `src` to a MAP_READ staging buffer and await the
/// map_async callback (JS promise) before returning words.
async fn read_words(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    off: u64,
    nbytes: u64,
) -> Result<Vec<u32>, String> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wasm-read"),
        size: nbytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, off, &staging, 0, nbytes);
        queue.submit(Some(enc.finish()));
    }
    let slice = staging.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    rx.await
        .map_err(|e| format!("map oneshot dropped: {e}"))?
        .map_err(|_| "map_async failed".to_string())?;
    let words: Vec<u32> = {
        let data = slice.get_mapped_range().map_err(|e| format!("get_mapped_range: {e:?}"))?;
        data.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    staging.unmap();
    Ok(words)
}

fn flat_bytes(v: &[u32]) -> Vec<u8> {
    u32s_to_bytes(v)
}

// ---- exported API ----

/// Init device via lib's browser path (no extra features, Limits::default()
/// = WebGPU spec floors on wasm). Builds the u32-limb pipeline set
/// (has_shader_int64 is false in-browser, so WGSL MUL_U32 prelude).
/// Returns the adapter/limits report for the console + DOM.
#[wasm_bindgen(js_name = "initWgpu")]
pub async fn init() -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    if CTX.with(|c| c.borrow().is_some()) {
        return Ok("INIT: already initialized".to_string());
    }
    let dev = WgpuDevice::init_browser().await;
    let report = format!(
        "ADAPTER name={} vendor=0x{:x} device=0x{:x} driver={} driver_info={} type={}\n\
         LIMITS storage_binding_max={} buffer_max={} wg_smem={} wg_inv={} wg_per_dim={} bindings_per_bg={}\n\
         FEATURES_ENABLED: (none)\nINIT: OK",
        dev.report.name,
        dev.report.vendor_id,
        dev.report.device_id,
        dev.report.driver,
        dev.report.driver_info,
        dev.report.device_type,
        dev.report.limits.max_storage_buffer_binding_size,
        dev.report.limits.max_buffer_size,
        dev.report.limits.max_compute_workgroup_storage_size,
        dev.report.limits.max_compute_invocations_per_workgroup,
        dev.report.limits.max_compute_workgroups_per_dimension,
        dev.report.limits.max_bindings_per_bind_group,
    );
    let u64_mul = dev.has_shader_int64; // false in the browser
    let ps = PipelineSet::new(&dev, u64_mul);
    let kctx = KernelCtx::new(&dev, &wgpu_backend::field::constants_buffer_mont());
    let ntt_ps = NttPipelineSet::new(&dev, u64_mul);
    let dummy = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    CTX.with(|c| *c.borrow_mut() = Some(Rc::new(Ctx { dev, ps, kctx, ntt_ps, dummy })));
    Ok(report)
}

// ---- u32mul selftest (port of src/bin/u32mul_test.rs + GPU sample leg) ----

const P32: u32 = 0x78000001; // 2013265921
const INV: u32 = 2013265919; // -P^{-1} mod 2^32

struct W64 {
    lo: u32,
    hi: u32,
}

// verbatim mirror of MUL_U32::mul64
fn mul64(a: u32, b: u32) -> W64 {
    let al = a & 65535u32;
    let ah = a >> 16u32;
    let bl = b & 65535u32;
    let bh = b >> 16u32;
    let p0 = al.wrapping_mul(bl);
    let p1 = ah.wrapping_mul(bl);
    let p2 = al.wrapping_mul(bh);
    let p3 = ah.wrapping_mul(bh);
    let mid = (p0 >> 16u32).wrapping_add(p1 & 65535u32).wrapping_add(p2 & 65535u32);
    W64 {
        lo: (p0 & 65535u32) | mid.wrapping_shl(16u32),
        hi: p3.wrapping_add(p1 >> 16u32).wrapping_add(p2 >> 16u32).wrapping_add(mid >> 16u32),
    }
}

// verbatim mirror of MUL_U32::mul
fn mul_u32_limbs(a: u32, b: u32) -> u32 {
    let t = mul64(a, b);
    let m = t.lo.wrapping_mul(INV);
    let u = mul64(m, P32);
    let co: u32 = if t.lo != 0u32 { 1u32 } else { 0u32 };
    let mut r = t.hi.wrapping_add(u.hi).wrapping_add(co);
    if r >= P32 {
        r -= P32;
    }
    r
}

// mirror of MUL_U64::mul (the golden path) on u64
fn mul_u64(a: u32, b: u32) -> u32 {
    let t = (a as u64) * (b as u64);
    let m = (t as u32).wrapping_mul(INV);
    let u = (m as u64) * (P32 as u64);
    let mut r = ((t + u) >> 32u32) as u32;
    if r >= P32 {
        r -= P32;
    }
    r
}

fn check_mul64() -> u32 {
    let mut n = 0u64;
    let edges = [0u32, 1, 2, 0xFFFF, 0x10000, 0xFFFFFFFF, 1 << 16, P32 - 1, P32, P32 + 1];
    for &a in edges.iter() {
        for &b in edges.iter() {
            let w = mul64(a, b);
            assert_eq!(((w.hi as u64) << 32) | w.lo as u64, (a as u64) * (b as u64), "mul64 {} {}", a, b);
            n += 1;
        }
    }
    let mut s: u64 = 0x243F6A8885A308D3;
    for _ in 0..(1u64 << 20) {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = (s >> 32) as u32;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let b = (s >> 32) as u32;
        let w = mul64(a, b);
        assert_eq!(((w.hi as u64) << 32) | w.lo as u64, (a as u64) * (b as u64), "mul64 rnd {} {} {}", a, b, "rnd");
        n += 1;
    }
    n as u32
}

#[wasm_bindgen(js_name = "u32mulSelftest")]
pub async fn u32mul_selftest() -> Result<String, JsValue> {
    run_u32mul().await.map_err(|e| JsValue::from_str(&e))
}

async fn run_u32mul() -> Result<String, String> {
    let mut out = String::new();
    // INV check: INV*P == -1 mod 2^32
    assert_eq!(((INV as u64) * (P32 as u64)) as u32, u32::MAX, "INV constant");
    let n64 = check_mul64();
    out.push_str(&format!("MUL64_EXACT: PASS ({} pairs, full u32 domain)\n", n64));

    // REDC compare over a,b < P: extremes + 2^24 pseudorandom pairs
    let mut cases: Vec<(u32, u32)> = vec![
        (0, 0), (0, 1), (1, 0), (1, 1), (P32 - 1, P32 - 1), (P32 - 1, 1), (1, P32 - 1),
        (2, P32 - 2), (P32 - 2, 2), (1 << 30, 1 << 30), (P32 - 1, P32 >> 1),
        (0x40000000 - 1, 0x40000000 - 1), (P32 - 1, 0x7FFFFFFE & (P32 - 1)),
    ];
    let mut s: u64 = 0x9E3779B97F4A7C15;
    const N: usize = 1 << 24; // same 16_777_216 random pairs as native u32mul_test
    for _ in 0..N {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = ((s >> 32) as u32) % P32;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let b = ((s >> 32) as u32) % P32;
        cases.push((a, b));
    }
    let mut bad = 0usize;
    let mut first: Option<(u32, u32, u32, u32)> = None;
    for &(a, b) in cases.iter() {
        let e = mul_u64(a, b);
        let g = mul_u32_limbs(a, b);
        if e != g {
            bad += 1;
            if first.is_none() {
                first = Some((a, b, e, g));
            }
        }
    }
    match first {
        Some((a, b, e, g)) => out.push_str(&format!(
            "U32_REDC_COMPARE: FAIL count={}/{} first a={} b={} expect_u64={} got_u32={}\n",
            bad, cases.len(), a, b, e, g
        )),
        None => out.push_str(&format!(
            "U32_REDC_COMPARE: PASS {}/{} equal\n",
            cases.len(),
            cases.len()
        )),
    }
    if bad != 0 {
        return Ok(out);
    }

    // GPU leg: run the EXACT wgsl::MUL_U32 shader over a sample corpus
    // (13 corners + 2^12 LCG pairs) and compare with the u64 mirror.
    let ctx = ctx()?;
    let dev = &ctx.dev;
    let mut gpu_cases: Vec<(u32, u32)> = vec![
        (0, 0), (0, 1), (1, 0), (1, 1), (P32 - 1, P32 - 1), (P32 - 1, 1), (1, P32 - 1),
        (2, P32 - 2), (P32 - 2, 2), (1 << 30, 1 << 30), (P32 - 1, P32 >> 1),
        (0x40000000 - 1, 0x40000000 - 1), (P32 - 1, 0x7FFFFFFE & (P32 - 1)),
    ];
    let mut s2: u64 = 0x1234567890ABCDE;
    for _ in 0..(1u32 << 12) {
        s2 = s2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = ((s2 >> 32) as u32) % P32;
        s2 = s2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let b = ((s2 >> 32) as u32) % P32;
        gpu_cases.push((a, b));
    }
    let n = gpu_cases.len();
    let mut pairs: Vec<u32> = Vec::with_capacity(n * 2);
    for (a, b) in &gpu_cases {
        pairs.push(*a);
        pairs.push(*b);
    }
    let mk = |size: u64, usages: wgpu::BufferUsages| {
        dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mul-sample"),
            size,
            usage: usages,
            mapped_at_creation: false,
        })
    };
    let pair_buf = mk((n * 2 * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let out_buf = mk((n * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    dev.queue.write_buffer(&pair_buf, 0, &u32s_to_bytes(&pairs));
    let wgsl = format!(
        "{}\n\
         @group(0) @binding(0) var<storage, read> pairs: array<u32>;\n\
         @group(0) @binding(1) var<storage, read_write> outw: array<u32>;\n\
         @compute @workgroup_size(64u)\n\
         fn mul_sample(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
           let i = gid.x;\n\
           if (2u * i >= arrayLength(&pairs)) {{ return; }}\n\
           outw[i] = mul(pairs[2u * i], pairs[2u * i + 1u]);\n\
         }}\n",
        wgpu_backend::wgsl::MUL_U32
    );
    let module = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mul-sample"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mul-sample"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("mul_sample"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pair_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        ],
    });
    wgpu_backend::pipeline::dispatch(dev, &pipe, &bg, n);
    let got = read_words(&dev.device, &dev.queue, &out_buf, 0, (n * 4) as u64).await?;
    let mut gbad = 0usize;
    let mut gfirst: Option<(u32, u32, u32, u32)> = None;
    for (i, &(a, b)) in gpu_cases.iter().enumerate() {
        let e = mul_u64(a, b);
        if got[i] != e {
            gbad += 1;
            if gfirst.is_none() {
                gfirst = Some((a, b, e, got[i]));
            }
        }
    }
    match gfirst {
        Some((a, b, e, g)) => out.push_str(&format!(
            "U32_REDC_COMPARE_GPU: FAIL count={}/{} first a={} b={} expect={} got={}\n",
            gbad, n, a, b, e, g
        )),
        None => out.push_str(&format!("U32_REDC_COMPARE_GPU: PASS {}/{} equal\n", n, n)),
    }
    Ok(out)
}

// ---- gate_c1 GPU legs ----

pub(crate) const N_NTT: usize = 4096;

pub(crate) fn gen_lcg(n: usize, seed: u64) -> Vec<u32> {
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        v.push((s >> 32) as u32);
    }
    v
}

fn br_bits(x: u32, bits: u32) -> u32 {
    let mut r = 0u32;
    for i in 0..bits {
        if (x >> i) & 1 == 1 {
            r |= 1 << (bits - 1 - i);
        }
    }
    r
}

fn fsub_l(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + (2013265921u32 - b)
    }
}

// CPU DIT NTT mirroring the WGSL kernel (verbatim port of ntt_test.rs)
pub(crate) fn ntt_dit(inp: &[u32], tw: &[u32]) -> Vec<u32> {
    let n = inp.len();
    let lg = n.trailing_zeros() as usize;
    let mut data = vec![0u32; n];
    for k in 0..n {
        let brk = br_bits(k as u32, lg as u32);
        data[brk as usize] = encode(inp[k]);
    }
    let mut off = 0usize;
    for st in 1..=lg {
        let half = 1usize << st;
        let q1 = half >> 1;
        let cnt = 1usize << (st - 1);
        for i in 0..(n / 2) {
            let b = i & (q1 - 1);
            let blk = i >> (st - 1);
            let pp = blk * half + b;
            let qq = pp + q1;
            let w = tw[off + b];
            let t = wgpu_backend::field::mont_mul(data[qq], w);
            let u = data[pp];
            data[pp] = wgpu_backend::field::fadd(u, t);
            data[qq] = fsub_l(u, t);
        }
        off += cnt;
    }
    data.iter().map(|&x| decode(x)).collect()
}

pub(crate) fn twiddles(n: usize, omega_canon: u32) -> Vec<u32> {
    let lg = n.trailing_zeros() as usize;
    let w_m = encode(omega_canon);
    let mut out = Vec::with_capacity(n - 1);
    for st in 1..=lg {
        let step = n >> st;
        let mut wpow = vec![encode(1); step * (1 << (st - 1))];
        for k in 1..wpow.len() {
            wpow[k] = wgpu_backend::field::mont_mul(wpow[k - 1], w_m);
        }
        for b in 0..(1usize << (st - 1)) {
            out.push(wpow[b * step]);
        }
    }
    out
}

#[wasm_bindgen(js_name = "gateC1Gpu")]
pub async fn gate_c1_gpu() -> Result<String, JsValue> {
    run_gate_c1().await.map_err(|e| JsValue::from_str(&e))
}

async fn run_gate_c1() -> Result<String, String> {
    let ctx = ctx()?;
    let dev = &ctx.dev;
    let mut out = String::new();
    let mut pass = true;

    // ---- 0) CPU reference self-check against goldens ----
    let vectors = parse_vectors(VEC_SRC);
    let p2 = Poseidon2::new();
    let mut cpu_ok = true;
    for (name, inp, exp) in &vectors {
        let got = p2.perm_canon(inp);
        if &to_hex(&got) != &to_hex(exp) {
            out.push_str(&format!("CPU_PERM {name}: FAIL\n"));
            cpu_ok = false;
        }
    }
    out.push_str(&format!(
        "CPU_REFERENCE_GOLDEN: {}\n",
        if cpu_ok { "PASS" } else { "FAIL" }
    ));
    if !cpu_ok {
        return Ok(out);
    }

    // ---- 3) GPU golden perms through the same PipelineSet/KernelCtx ----
    let mut flat: Vec<u32> = Vec::new();
    for (_, inp, _) in &vectors {
        for i in 0..16 {
            flat.push(encode(inp[i]));
        }
    }
    let n = vectors.len();
    let mkbuf = |size: u64, usages: wgpu::BufferUsages| {
        dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perm"),
            size,
            usage: usages,
            mapped_at_creation: false,
        })
    };
    let in_buf = mkbuf(
        (n * 16 * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    dev.queue.write_buffer(&in_buf, 0, &u32s_to_bytes(&flat));
    // perm_batch permutes io1 IN-PLACE (native WgpuBackend::perm_batch binds
    // io2=dummy); results come back in in_buf.
    let bg = ctx.kctx.bind(dev, &ctx.ps, &in_buf, &ctx.dummy);
    dispatch(dev, &ctx.ps.perm_batch, &bg, n); // perm_batch: 16 words/thread
    let got_words = read_words(&dev.device, &dev.queue, &in_buf, 0, (n * 16 * 4) as u64).await?;
    let mut out_canon: Vec<u32> = Vec::new();
    for (vi, (_, _, exp)) in vectors.iter().enumerate() {
        let mut got = [0u32; 16];
        for j in 0..16 {
            got[j] = decode(got_words[vi * 16 + j]);
        }
        let ok = to_hex(&got) == to_hex(exp);
        out.push_str(&format!("GPU_PERM {}: {}\n", vectors[vi].0, if ok { "PASS" } else { "FAIL" }));
        pass &= ok;
        out_canon.extend_from_slice(&got);
    }
    out.push_str(&format!("GPU_PERM_OUTPUTS_SHA256: {}\n", sha256_hex(&flat_bytes(&out_canon))));

    // ---- 4) 4096-leaf merkle root (same windowed loop as lib.rs, async tail) ----
    let seed_f = (MERKLE_SEED % P as u64) as u32;
    let nl = MERKLE_LEAVES;
    let us = wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC;
    let rows = ShardBuf::new(dev, "merkle-rows", us, (nl * 8 * 4) as u64);
    let wr = (dev.win() / 32).min(nl as u64).max(1);
    for w0 in (0..nl as u64).step_by(wr as usize) {
        let cnt = wr.min(nl as u64 - w0);
        ctx.kctx.set_seed_row(dev, seed_f, w0 as u32);
        let bgr = ctx.kctx.bind_res(
            dev,
            &ctx.ps,
            rows.bind(dev, w0 * 32, cnt * 32),
            ctx.dummy.as_entire_binding(),
        );
        dispatch(dev, &ctx.ps.derive_row, &bgr, cnt as usize);
    }
    let mut cur = ShardBuf::new(dev, "merkle-leaves", us, (nl * 8 * 4) as u64);
    for w0 in (0..nl as u64).step_by(wr as usize) {
        let cnt = wr.min(nl as u64 - w0);
        let bgr = ctx.kctx.bind_res(
            dev,
            &ctx.ps,
            rows.bind(dev, w0 * 32, cnt * 32),
            cur.bind(dev, w0 * 32, cnt * 32),
        );
        dispatch(dev, &ctx.ps.merkle_init, &bgr, cnt as usize);
    }
    let mut cur_n = nl;
    while cur_n > 1 {
        let out_nodes = cur_n / 2;
        let nxt = ShardBuf::new(dev, "merkle-nxt", us, (out_nodes * 8 * 4) as u64);
        let wo = (dev.win() / 64).min(out_nodes as u64).max(1);
        for w0 in (0..out_nodes as u64).step_by(wo as usize) {
            let cnt = wo.min(out_nodes as u64 - w0);
            let bgr = ctx.kctx.bind_res(
                dev,
                &ctx.ps,
                cur.bind(dev, w0 * 64, cnt * 64),
                nxt.bind(dev, w0 * 32, cnt * 32),
            );
            dispatch(dev, &ctx.ps.merkle_compress, &bgr, cnt as usize);
        }
        cur = nxt;
        cur_n = out_nodes;
    }
    let root_mont = read_words(&dev.device, &dev.queue, &cur.shards[0], 0, 32).await?;
    let mut root = [0u32; 8];
    for i in 0..8 {
        root[i] = decode(root_mont[i]);
    }
    let root_hex = to_hex(&root);
    let root_sha = sha256_hex(&flat_bytes(&root));
    let ok_root = root_hex == MERKLE_ROOT_HEX;
    let ok_sha = root_sha == MERKLE_ROOT_SHA256;
    out.push_str(&format!(
        "GPU_MERKLE root={root_hex} exp={MERKLE_ROOT_HEX} [{}]\n",
        if ok_root { "PASS" } else { "FAIL" }
    ));
    out.push_str(&format!(
        "GPU_MERKLE sha256={root_sha} exp={MERKLE_ROOT_SHA256} [{}]\n",
        if ok_sha { "PASS" } else { "FAIL" }
    ));
    pass &= ok_root && ok_sha;

    // ---- 5) NTT 4096 golden leg (port of ntt_test.rs GPU path) ----
    const NTT_FIRST8: &str = "08407c0c33095dae29413c166d2c846a41695163328bd2fd00f11c08568aebab";
    const NTT_LAST8: &str = "481851ba74ad8f6f516167e7414acf100ca25fad5cd153ce3e6583f51a4f3fa0";
    const NTT_GOLDEN_SHA: &str = "a11b282970362f31b0d16a443e93a9171d5744ef657358da628b902e9835be0e";
    let inp = gen_lcg(N_NTT, 0x5eed1234abcd0001u64);
    let inp_mont: Vec<u32> = inp.iter().map(|&x| encode(x)).collect();
    let omega = 0x4c734715u32; // TWO_ADIC_GENERATORS[12]
    let tw = twiddles(N_NTT, omega);
    let out_cpu = ntt_dit(&inp, &tw);
    let cpu_first = to_hex(&out_cpu[0..8]);
    let cpu_last = to_hex(&out_cpu[N_NTT - 8..N_NTT]);
    let cpu_ok_ntt = cpu_first == NTT_FIRST8 && cpu_last == NTT_LAST8;
    out.push_str(&format!(
        "CPU_NTT first8={} match={} \n",
        cpu_first,
        if cpu_ok_ntt { "PASS" } else { "FAIL" }
    ));
    pass &= cpu_ok_ntt;

    let n_buf = mkbuf((N_NTT * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let tw_buf = mkbuf((tw.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let ntt_out = mkbuf((N_NTT * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    dev.queue.write_buffer(&n_buf, 0, &u32s_to_bytes(&inp_mont));
    dev.queue.write_buffer(&tw_buf, 0, &u32s_to_bytes(&tw));
    let ntbg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &ctx.ntt_ps.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: n_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tw_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ntt_out.as_entire_binding() },
        ],
    });
    dispatch_ntt(dev, &ctx.ntt_ps.fwd, &ntbg);
    let got_mont = read_words(&dev.device, &dev.queue, &ntt_out, 0, (N_NTT * 4) as u64).await?;
    let got_canon: Vec<u32> = got_mont.iter().map(|&x| decode(x)).collect();
    let gpu_first = to_hex(&got_canon[0..8]);
    let gpu_last = to_hex(&got_canon[N_NTT - 8..N_NTT]);
    let mut nbad = 0usize;
    for i in 0..N_NTT {
        if got_canon[i] != out_cpu[i] {
            nbad += 1;
        }
    }
    let full_sha = sha256_hex(&flat_bytes(&got_canon));
    let sha_ok = full_sha == NTT_GOLDEN_SHA;
    out.push_str(&format!("GPU_NTT first8={} last8={}\n", gpu_first, gpu_last));
    out.push_str(&format!(
        "GPU_NTT sha256={full_sha} exp={NTT_GOLDEN_SHA} [{}]\n",
        if sha_ok { "PASS" } else { "FAIL" }
    ));
    out.push_str(&format!(
        "GPU_NTT full correctness: {} ({} bad words vs CPU)\n",
        if nbad == 0 { "PASS" } else { "FAIL" },
        nbad
    ));
    pass &= sha_ok && nbad == 0;

    out.push_str(&format!("GATE_C1_GPU: {}\n", if pass { "PASS" } else { "FAIL" }));
    Ok(out)
}

// ---- Phase 3b exports ----

/// gate_c1 with bind-group atlas + ONE batched mapAsync (vs classic's 3).
#[wasm_bindgen(js_name = "gateC1Batched")]
pub async fn gate_c1_batched() -> Result<String, JsValue> {
    gatefast::run_fast().await.map_err(|e| JsValue::from_str(&e))
}

/// W2C PoW grind in-browser: 90 sites, u32 atomicMin design, golden
/// run-hash e69fddee611e56ab (per-site + batched modes + determinism).
#[wasm_bindgen(js_name = "grindWgpu")]
pub async fn grind_wgpu() -> Result<String, JsValue> {
    grind::run_all().await.map_err(|e| JsValue::from_str(&e))
}

/// w2a-stretch I/O wall probe (Phase 3c): upload `bytes` JS->wasm->GPU,
/// then read the first 4 MiB back through the batched async path. Measures
/// the JS-fed I/O wall (W2) that a real w2a leaf would hit.
#[wasm_bindgen(js_name = "ioProbe")]
pub async fn io_probe(bytes: Box<[u8]>) -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let dev = &c.dev;
    let size = bytes.len() as u64;
    let t0 = js_sys::Date::now();
    let buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe-src"),
        size: size.max(16),
        usage: BU::COPY_DST | BU::COPY_SRC,
        mapped_at_creation: false,
    });
    dev.queue.write_buffer(&buf, 0, &bytes);
    // wgpu 30.0.2 web quirk (measured io-10..14): a copy-only encoder
    // submitted right after queue.write_buffer sees PRE-WRITE contents
    // (zeros); reading the written buffer directly via ReadBatch (whose
    // own staging copy lands in a later submit) returns the real data.
    // Real kernels are unaffected: their first GPU op is a dispatch.
    dev.queue.submit([]);
    let n = (size.min(4 * 1024 * 1024)).max(4);
    let t1 = js_sys::Date::now();
    let mut rb = ReadBatch::new();
    rb.push(&buf, 0, n / 4); // read SOURCE directly: is the write landing?
    let got = rb.resolve(&dev.device, &dev.queue).await.map_err(|e| JsValue::from_str(&e))?;
    
    let head = got[0].first().copied().unwrap_or(0);
    let w1 = got[0].get(1).copied().unwrap_or(0);
    // sanity: did the JS->wasm payload actually arrive?
    let cks: u64 = bytes.iter().map(|b| *b as u64).sum();
    let words = got.first().map(|v| v.len()).unwrap_or(999);
    let upload_ms = t1 - t0; // t1 is taken right after upload/submit
    let readback_ms = js_sys::Date::now() - t1;
    Ok(format!(
        "{{\"size\":{size},\"upload_ms\":{upload_ms:.1},\"readback_ms\":{readback_ms:.1},\"head_word\":{head},\"w1\":{w1},\"words\":{words},\"cks\":{cks}}}"
    ))
}
