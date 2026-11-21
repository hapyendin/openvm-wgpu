// W2a: GKR/zerocheck GPU driver — builds the Frac segment tree, then runs the
// T-1 GKR sumcheck rounds (eq build on host, unpack/seval/fold on GPU),
// comparing every observed value against the recorded transcript
// (gkr-inputs.json: streams are authoritative for this dump).
//
// Determinism law: fixed-index kernels only, u32 Montgomery integer math, no
// atomics, no floats. 5 consecutive runs must be bit-identical.
//
// API style mirrors the fork-C tree (wgpu 30.0.2: ShaderSource, entry_point:
// Some, compilation_options, PollType::Wait, submit(Some), mapped_at_creation).

use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::path::Path;

use crate::limitsbuf::ShardBuf;

use wgpu::BufferUsages as BU;

use crate::device::WgpuDevice;
use crate::field::{decode, encode, fadd, fsub, mont_mul, u32s_to_bytes};
use crate::gkr_wgsl::GKR_WGSL;

// element widths (u32) for tree_add
const W_FRAC: u32 = 8;
const W_STRIPLE: u32 = 12;
const ROW10: u32 = 40; // 10 EF columns

// ---------------- JSON input model (gkr-inputs.json) ----------------

#[derive(serde::Deserialize)]
pub struct GkrInputs {
    pub p: u64,
    pub ext: u32,
    pub instances: Vec<GkrInstance>,
}

#[derive(serde::Deserialize)]
pub struct GkrInstance {
    pub name: String,
    pub t: u32,
    pub alpha: Vec<u64>,
    pub beta: Vec<u64>,
    pub root_q: Vec<u64>,
    pub root_p: Vec<u64>,
    pub claims0: Vec<u64>,
    pub blocks: Vec<u64>,
    pub per_round_xi: BTreeMap<u32, Vec<u64>>,
    pub rounds: Vec<GkrRound>,
    pub xi_full: Vec<u64>,
}

#[derive(serde::Deserialize)]
pub struct GkrRound {
    pub j: u32,
    pub lambda: Vec<u64>,
    pub subs: Vec<GkrSub>,
    pub claims: Vec<u64>,
    pub mu: Vec<u64>,
}

#[derive(serde::Deserialize)]
pub struct GkrSub {
    pub s_evals: Vec<u64>,
    pub r: Vec<u64>,
}

/// Per-instance GPU outputs (canonical u32 limbs) + hashes for determinism.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GkrOutputs {
    pub name: String,
    pub t: u32,
    pub root_p: Vec<u32>,
    pub root_q: Vec<u32>,
    pub claims0: Vec<u32>,
    pub levels: Vec<LevelSnap>,
    pub rounds: Vec<RoundOut>,
    pub leaf_sha256: String,
    pub out_sha256: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LevelSnap {
    pub lvl: u32,
    pub tree_level: u32,
    pub tag: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RoundOut {
    pub j: u32,
    pub subs: Vec<Vec<u32>>, // 3 EFs each (12 u32)
    pub claims: Vec<u32>,     // 16 u32
}

// ---------------- pipeline ----------------

pub struct GkrPipeline {
    pub module: wgpu::ShaderModule,
    pub bgl: wgpu::BindGroupLayout,
    pub layout: wgpu::PipelineLayout,
    pub pipe_tree: wgpu::ComputePipeline,
    pub pipe_unpack: wgpu::ComputePipeline,
    pub pipe_seval: wgpu::ComputePipeline,
    pub pipe_fold: wgpu::ComputePipeline,
    pub uni: wgpu::Buffer,
    pub dummy: wgpu::Buffer, // 16-byte read buffer for unused read slots
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GkrPipeline {
    pub fn new(dev: &WgpuDevice) -> GkrPipeline {
        let module = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gkr-w2a"),
            source: wgpu::ShaderSource::Wgsl(
                crate::wgsl::with_prelude(dev.has_shader_int64, GKR_WGSL).into(),
            ),
        });
        let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gkr-4slot"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(64),
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
        let mk = |entry: &str| {
            dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipe_tree = mk("tree_add");
        let pipe_unpack = mk("unpack");
        let pipe_seval = mk("seval");
        let pipe_fold = mk("fold");
        let uni = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gkr-uni"),
            size: 64,
            usage: BU::UNIFORM | BU::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gkr-dummy"),
            size: 16,
            usage: BU::STORAGE | BU::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&dummy, 0, &[0u8; 16]);
        GkrPipeline {
            module,
            bgl,
            layout,
            pipe_tree,
            pipe_unpack,
            pipe_seval,
            pipe_fold,
            uni,
            dummy,
        }
    }

    // WGSL struct U { w: u32; h: u32; base: u32; pad: u32; lam: vec4; r: vec4 }
    // offsets: w@0, h@4, base@8, pad@12, lam@16, r@32  (48 bytes, buffer 64)
    fn set_uniform(&self, dev: &WgpuDevice, w: u32, lam: [u32; 4], r: [u32; 4], h: u32, base: u32, sbase: u32) {
        let mut b = [0u8; 64];
        b[0..4].copy_from_slice(&w.to_le_bytes());
        b[4..8].copy_from_slice(&h.to_le_bytes());
        b[8..12].copy_from_slice(&base.to_le_bytes());
        b[12..16].copy_from_slice(&sbase.to_le_bytes());
        b[16..20].copy_from_slice(&lam[0].to_le_bytes());
        b[20..24].copy_from_slice(&lam[1].to_le_bytes());
        b[24..28].copy_from_slice(&lam[2].to_le_bytes());
        b[28..32].copy_from_slice(&lam[3].to_le_bytes());
        b[32..36].copy_from_slice(&r[0].to_le_bytes());
        b[36..40].copy_from_slice(&r[1].to_le_bytes());
        b[40..44].copy_from_slice(&r[2].to_le_bytes());
        b[44..48].copy_from_slice(&r[3].to_le_bytes());
        dev.queue.write_buffer(&self.uni, 0, &b);
    }

    /// bind from pre-built BindingResources (Phase-2 sliced/windowed binds).
    fn bind3(
        &self,
        dev: &WgpuDevice,
        r0: wgpu::BindingResource<'_>,
        r1: wgpu::BindingResource<'_>,
        r2: wgpu::BindingResource<'_>,
    ) -> wgpu::BindGroup {
        dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: r0 },
                wgpu::BindGroupEntry { binding: 1, resource: r1 },
                wgpu::BindGroupEntry { binding: 2, resource: r2 },
                wgpu::BindGroupEntry { binding: 3, resource: self.uni.as_entire_binding() },
            ],
        })
    }

    fn run(dev: &WgpuDevice, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, wgs: u32) {
        let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(wgs.max(1), 1, 1);
        }
        dev.queue.submit(Some(enc.finish()));
    }
}

// ---------------- host helpers ----------------

/// evals_eq_hypercube mirror (reference poly.rs:141, raw-Montgomery domain).
/// out[y] = prod_i (xis[i] if y bit i set else 1 - xis[i]) as an EF product
/// (x^4=11, emul_host), NOT limbwise; xis in Mont domain, LSB-first
/// (x_i <-> bit i). Init per point = Mont EF (1,0,0,0).
pub fn eq_vec_host(xis: &[[u32; 4]]) -> Vec<u32> {
    let n = xis.len() as u32;
    let one = encode(1u32);
    let mut out: Vec<u32> = vec![0u32; (1usize << n) * 4];
    for p in 0..(1usize << n) {
        out[p * 4] = one;
    }
    for (i, x) in xis.iter().enumerate() {
        let half = 1usize << i;
        // 1 - x is EF subtraction from the EF element (1,0,0,0): lane0 gets
        // 1-x0, lanes 1..3 get 0-xi (NOT limbwise 1-xi — AT5 bug: that made
        // eqe off by +1 per lane 1..3, corrupting the X2/X3 lerps only).
        let one_ef = [encode(1u32), 0u32, 0u32, 0u32];
        let omx = [
            fsub(one_ef[0], x[0]),
            fsub(one_ef[1], x[1]),
            fsub(one_ef[2], x[2]),
            fsub(one_ef[3], x[3]),
        ];
        for y in 0..half {
            let lo = y * 4;
            let hi = (half + y) * 4;
            let l = [out[lo], out[lo + 1], out[lo + 2], out[lo + 3]];
            // Standard hypercube doubling: new[hi] = old[lo] * x and
            // new[lo] = old[lo] * (1-x). Using the stale out[hi] (=1) for the
            // high half (AT5 bug) dropped all earlier challenges from eq[k>=2].
            let nh = emul_host(&l, x);
            let nl = emul_host(&l, &omx);
            out[hi..hi + 4].copy_from_slice(&nh);
            out[lo..lo + 4].copy_from_slice(&nl);
        }
    }
    out
}

/// W2d lowmem eq builder: for |xis| > 20 avoids materializing the full 2^j
/// table (512 MiB at j=25) by splitting y = (p << LOW) | e into
/// eq[y] = eq_low[e] * eq_high[p], with LOW = 20 (16 MiB tables at most),
/// uploading one 2^LOW-entry block (16 MiB) at a time.
fn eq_upload_split(dev: &WgpuDevice, eqv: &ShardBuf, xis: &[[u32; 4]]) {
    const LOW: usize = 20;
    if xis.len() <= LOW {
        let eq = eq_vec_host(xis);
        let mut bo = 0usize;
        while bo < eq.len() {
            let be = (bo + (1usize << 22)).min(eq.len());
            eqv.write_slice(dev, (bo * 4) as u64, &u32s_to_bytes(&eq[bo..be]));
            bo = be;
        }
        return;
    }
    let hi = xis.len() - LOW;
    let eqlow = eq_vec_host(&xis[..LOW]);
    let eqhi = eq_vec_host(&xis[LOW..]);
    let mut bytes: Vec<u8> = Vec::with_capacity((1usize << LOW) * 16);
    for p in 0..(1usize << hi) {
        let h = [eqhi[p * 4], eqhi[p * 4 + 1], eqhi[p * 4 + 2], eqhi[p * 4 + 3]];
        bytes.clear();
        for e in 0..(1usize << LOW) {
            let l = [eqlow[e * 4], eqlow[e * 4 + 1], eqlow[e * 4 + 2], eqlow[e * 4 + 3]];
            for w in emul_host(&l, &h) {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
        }
        eqv.write_slice(dev, (p << LOW) as u64 * 16, &bytes);
    }
}

/// Binomial EF multiply (x^4 = 11) in raw-Montgomery domain (mirrors WGSL emul).
pub fn emul_host(a: &[u32; 4], b: &[u32; 4]) -> [u32; 4] {
    let w11 = encode(11u32);
    let t4 = fadd(fadd(mont_mul(a[1], b[3]), mont_mul(a[2], b[2])), mont_mul(a[3], b[1]));
    let t5 = fadd(mont_mul(a[2], b[3]), mont_mul(a[3], b[2]));
    let t6 = mont_mul(a[3], b[3]);
    let r0 = fadd(mont_mul(a[0], b[0]), mont_mul(w11, t4));
    let r1 = fadd(fadd(mont_mul(a[0], b[1]), mont_mul(a[1], b[0])), mont_mul(w11, t5));
    let r2 = fadd(
        fadd(fadd(mont_mul(a[0], b[2]), mont_mul(a[1], b[1])), mont_mul(a[2], b[0])),
        mont_mul(w11, t6),
    );
    let r3 = fadd(
        fadd(fadd(mont_mul(a[0], b[3]), mont_mul(a[1], b[2])), mont_mul(a[2], b[1])),
        mont_mul(a[3], b[0]),
    );
    [r0, r1, r2, r3]
}

fn fadd4(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    [
        fadd(a[0], b[0]),
        fadd(a[1], b[1]),
        fadd(a[2], b[2]),
        fadd(a[3], b[3]),
    ]
}

fn ef_decode4(canon: &[u64]) -> [u32; 4] {
    [
        canon[0] as u32,
        canon[1] as u32,
        canon[2] as u32,
        canon[3] as u32,
    ]
}

/// canonical EF (from JSON) -> raw-Montgomery EF for the kernels.
fn ef_encode4(canon: &[u64]) -> [u32; 4] {
    [
        encode(canon[0] as u32),
        encode(canon[1] as u32),
        encode(canon[2] as u32),
        encode(canon[3] as u32),
    ]
}

/// fold endpoint pair at Mont EF point r (raw domain): e + (o - e) * r.
fn fold_pair(e: &[u32], o: &[u32], r: &[u32; 4]) -> [u32; 4] {
    let d = [
        fsub(o[0], e[0]),
        fsub(o[1], e[1]),
        fsub(o[2], e[2]),
        fsub(o[3], e[3]),
    ];
    let m = emul_host(&d, r);
    [
        fadd(e[0], m[0]),
        fadd(e[1], m[1]),
        fadd(e[2], m[2]),
        fadd(e[3], m[3]),
    ]
}

fn raw_to_canon(v: &[u32]) -> Vec<u32> {
    v.iter().map(|x| decode(*x)).collect()
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

#[allow(dead_code)]
fn download(dev: &WgpuDevice, buf: &wgpu::Buffer) -> Vec<u8> {
    download_n(dev, buf, buf.size())
}

// ---------------- async readback port (W2A in-browser) ----------------
//
// W2A-BROWSER: on wasm32 `device.poll(PollType::Wait)` is a NO-OP in the
// WebGPU backend (returns Ok(QueueEmpty), map resolves from the JS
// mapAsync promise), so the sync poll-loop readbacks below silently return
// empty in the browser. The *_async variants keep the NATIVE branch exactly
// as the sync code (same map_async + poll(Wait) + get_mapped_range tail,
// same call order) and only swap the wait for a oneshot await of the
// mapAsync callback on wasm32 (the wasm-pack oneshot pattern). The driver
// (`run_instance_async` / `run_instance_gpu_async`) awaits these; the native
// sync API is preserved via pollster::block_on wrappers at the bottom.

/// Copy `n` bytes of `buf` starting at `off` to the host (awaitable).
pub async fn download_off_async(
    dev: &WgpuDevice,
    buf: &wgpu::Buffer,
    off: u64,
    n: u64,
) -> Vec<u8> {
    let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gkr-stage-off"),
        size: n,
        usage: BU::COPY_DST | BU::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: None,
    });
    enc.copy_buffer_to_buffer(buf, off, &staging, 0, n);
    dev.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    #[cfg(not(target_arch = "wasm32"))]
    {
        slice.map_async(wgpu::MapMode::Read, |_| ());
        dev.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
    }
    #[cfg(target_arch = "wasm32")]
    {
        let (tx, rx) = futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Native never lands here; on wasm the callback fires from the JS
        // mapAsync promise. A dropped channel / map error yields empty data
        // exactly like the native `.ok()`/`unwrap_or_default()` posture.
        let _ = rx.await;
    }
    let data: Vec<u8> = slice
        .get_mapped_range()
        .map(|d| d.to_vec())
        .unwrap_or_default();
    drop(slice);
    staging.unmap();
    data
}

/// Copy the first `n` bytes of `buf` to the host (awaitable).
pub async fn download_n_async(dev: &WgpuDevice, buf: &wgpu::Buffer, n: u64) -> Vec<u8> {
    download_off_async(dev, buf, 0, n).await
}

/// Copy the first `n` bytes of `buf` to the host.
/// Copy `n` bytes of `buf` starting at byte offset `off` to the host.
#[allow(dead_code)]
fn download_off(dev: &WgpuDevice, buf: &wgpu::Buffer, off: u64, n: u64) -> Vec<u8> {
    let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gkr-stage-off"),
        size: n,
        usage: BU::COPY_DST | BU::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: None,
    });
    enc.copy_buffer_to_buffer(buf, off, &staging, 0, n);
    dev.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    dev.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .ok();
    let data: Vec<u8> = slice
        .get_mapped_range()
        .map(|d| d.to_vec())
        .unwrap_or_default();
    drop(slice);
    staging.unmap();
    data
}

#[allow(dead_code)]
fn download_n(dev: &WgpuDevice, buf: &wgpu::Buffer, n: u64) -> Vec<u8> {
    let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gkr-stage"),
        size: n,
        usage: BU::COPY_DST | BU::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: None,
    });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, n);
    dev.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    dev.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .ok();
    let data: Vec<u8> = slice
        .get_mapped_range()
        .map(|d| d.to_vec())
        .unwrap_or_default();
    drop(slice);
    staging.unmap();
    data
}

// ---------------- instance driver ----------------

/// sliced BindingResource with EXPLICIT length (Phase-2 binds: never bind
/// past the device's storage-binding cap).
fn bres(b: &wgpu::Buffer, off: u64, len: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: b,
        offset: off,
        size: Some(unsafe { NonZeroU64::new_unchecked(len.max(4)) }),
    })
}

/// (nchunks, chunk_bytes) for a 2^t-row leaf: ONE GKR unpack window per
/// chunk (chunk = 4*wsz rows x 32 B). ANV default: 4 x 512 MiB;
/// WebGPU-floor device (wsz=2^19, t=26): 32 x 64 MiB.
pub fn leaf_chunks(dev: &WgpuDevice, t: u32) -> (u32, u64) {
    let wsz = dev.gkr_wsz() as u64;
    let nchunks = ((1u64 << (t - 2)) / wsz).max(1);
    let chunk_rows = (1u64 << t) / nchunks;
    (nchunks as u32, chunk_rows * 32)
}

/// Run one full GKR instance on the GPU; returns canonical outputs.
/// All buffers are dropped on return (memory law: free between runs).
/// Native sync API (unchanged signature): pollster wrapper over the async
/// core. On native the async core never yields (the readback cfg branch is
/// the same poll(Wait) tail), so behavior is identical to the historical
/// sync body.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_instance(
    dev: &WgpuDevice,
    pipe: &GkrPipeline,
    inst: &GkrInstance,
    leaf_bytes: &[u8],
    inputs_dir: &Path,
    snap_shas: &mut Vec<LevelSnap>,
) -> GkrOutputs {
    pollster::block_on(run_instance_async(
        dev, pipe, inst, leaf_bytes, inputs_dir, snap_shas,
    ))
}

pub async fn run_instance_async(
    dev: &WgpuDevice,
    pipe: &GkrPipeline,
    inst: &GkrInstance,
    leaf_bytes: &[u8],
    inputs_dir: &Path,
    snap_shas: &mut Vec<LevelSnap>,
) -> GkrOutputs {
    let t = inst.t;
    let n = 1u32 << t; // leaf count
    let leaf_sha = sha256_hex(leaf_bytes);

    // 1) encode leaves (canonical -> Montgomery raw)
    let mut raw: Vec<u32> = Vec::with_capacity(leaf_bytes.len() / 4);
    for chunk in leaf_bytes.chunks(4) {
        let v = u32::from_le_bytes(chunk.try_into().unwrap());
        raw.push(encode(v));
    }
    let raw_bytes = u32s_to_bytes(&raw);

    // 2) leaf level buffer(s): one unpack window per chunk, sized so the
    // whole chunk binds legally under the device's binding cap.
    let (nchunks, per_chunk) = leaf_chunks(dev, t);
    let mut leaf_bufs: Vec<wgpu::Buffer> = Vec::new();
    for c in 0..nchunks {
        let b = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("gkr-leaf{}", c)),
            size: per_chunk,
            usage: BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
            mapped_at_creation: false,
        });
        let start = c as u64 * per_chunk;
        dev.queue
            .write_buffer(&b, 0, &raw_bytes[start as usize..(start + per_chunk) as usize]);
        leaf_bufs.push(b);
    }
    run_instance_gpu_async(dev, pipe, inst, leaf_bufs, leaf_sha, inputs_dir, snap_shas).await
}

/// W2b phase-2 entry: run rounds on PREBUILT raw-Montgomery leaf chunk
/// buffers (produced by the WGSL leaf builder in logup_leaf.rs). Same 512 MiB
/// chunk geometry as run_instance (nchunks = leaf_size / 2^29).
/// Native sync API (unchanged signature): pollster wrapper over the async
/// core (see run_instance note).
#[cfg(not(target_arch = "wasm32"))]
pub fn run_instance_gpu(
    dev: &WgpuDevice,
    pipe: &GkrPipeline,
    inst: &GkrInstance,
    leaf_bufs: Vec<wgpu::Buffer>,
    leaf_sha: String,
    inputs_dir: &Path,
    snap_shas: &mut Vec<LevelSnap>,
) -> GkrOutputs {
    pollster::block_on(run_instance_gpu_async(
        dev, pipe, inst, leaf_bufs, leaf_sha, inputs_dir, snap_shas,
    ))
}

pub async fn run_instance_gpu_async(
    dev: &WgpuDevice,
    pipe: &GkrPipeline,
    inst: &GkrInstance,
    leaf_bufs: Vec<wgpu::Buffer>,
    leaf_sha: String,
    inputs_dir: &Path,
    snap_shas: &mut Vec<LevelSnap>,
) -> GkrOutputs {
    let t = inst.t;
    let nchunks = leaf_bufs.len() as u32;

    // 3) level buffers 1..=t-1 (level L = 2^L Fracs, node-index order)
    let mut levels: Vec<ShardBuf> = (1..t)
        .map(|l| {
            ShardBuf::new(
                dev,
                &format!("gkr-lvl{}", l),
                BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
                (1u64 << l) * 32,
            )
        })
        .collect();

    // 4) tree build: level L from level L+1 (L = t .. 2)
    // Window size per dispatch: pairs <= win/64B (src = 64 B/pair) and
    // <= 65535 workgroups. All binds are sliced pow2 windows with the
    // kernel counting pairs from 0 (u.base = u.pad = 0).
    let pmax: u32 = ((dev.win() / 64) as u32).min(65535 * 256).max(256);
    for l in (2..=t).rev() {
        let dst = &levels[(l - 1) as usize - 1];
        let th_full = 1u32 << (l - 1);
        if l == t {
            // source = leaves (chunked); pairs per chunk feed the dst at base
            let per = th_full / nchunks;
            for c in 0..nchunks {
                let mut done = 0u32;
                while done < per {
                    let pairs = (per - done).min(pmax);
                    pipe.set_uniform(dev, W_FRAC, [0; 4], [0; 4], pairs, 0, 0);
                    let bg = pipe.bind3(
                        dev,
                        bres(
                            &leaf_bufs[c as usize],
                            done as u64 * 64,
                            pairs as u64 * 64,
                        ),
                        pipe.dummy.as_entire_binding(),
                        dst.bind(
                            dev,
                            (c as u64 * per as u64 + done as u64) * 32,
                            pairs as u64 * 32,
                        ),
                    );
                    GkrPipeline::run(dev, &pipe.pipe_tree, &bg, pairs.div_ceil(256));
                    done += pairs;
                }
            }
        } else {
            let src = &levels[(l - 1) as usize];
            let mut done = 0u32;
            while done < th_full {
                let pairs = (th_full - done).min(pmax);
                pipe.set_uniform(dev, W_FRAC, [0; 4], [0; 4], pairs, 0, 0);
                let bg = pipe.bind3(
                    dev,
                    src.bind(dev, done as u64 * 64, pairs as u64 * 64),
                    pipe.dummy.as_entire_binding(),
                    dst.bind(dev, done as u64 * 32, pairs as u64 * 32),
                );
                GkrPipeline::run(dev, &pipe.pipe_tree, &bg, pairs.div_ceil(256));
                done += pairs;
            }
        }
    }

    // 5) root + claims0 (level 1 = 2 Fracs = tree[2], tree[3]; projective Frac add)
    let lvl1 = &levels[0];
    let u32s = lvl1.read_words_async(dev, 0, lvl1.size / 4).await;
    let p2 = [u32s[0], u32s[1], u32s[2], u32s[3]];
    let q2 = [u32s[4], u32s[5], u32s[6], u32s[7]];
    let p3 = [u32s[8], u32s[9], u32s[10], u32s[11]];
    let q3 = [u32s[12], u32s[13], u32s[14], u32s[15]];
    let root_p = raw_to_canon(&fadd4(emul_host(&p2, &q3), emul_host(&p3, &q2)));
    let root_q = raw_to_canon(&emul_host(&q2, &q3));
    let claims0 = raw_to_canon(&u32s);

    // debug: dump small GPU levels to W2A_DBG_DIR for golden diffing
    if let Ok(dir) = std::env::var("W2A_DBG_DIR") {
        let dump_lvls: Vec<u32> = if inst.name.contains("app-segments") {
            vec![2, 13, 14, 16, 18, 20, 22, 23, 24, 25]
        } else {
            vec![2, 13]
        };
        for lvl in dump_lvls {
            if (lvl as usize) <= levels.len() {
                let words = levels[(lvl as usize) - 1].read_all_async(dev).await;
                let canon = raw_to_canon(&words);
                let cbytes: Vec<u8> = canon.iter().flat_map(|w| w.to_le_bytes()).collect();
                let _ = std::fs::create_dir_all(&dir);
                let p = std::path::Path::new(&dir).join(format!("dbg-{}-lvl{}-gpu.bin", inst.name, lvl));
                let _ = std::fs::write(&p, &cbytes);
            }
        }
        for (c, lb) in leaf_bufs.iter().enumerate() {
            let bytes = download_n_async(dev, lb, lb.size()).await;
            let p = std::path::Path::new(&dir).join(format!("dbg-{}-leaf{}-gpu.bin", inst.name, c));
            let _ = std::fs::write(&p, &bytes);
        }
    }
    // 6) golden level snapshots (recorded files: lvl 13,14,16,18,20[,22])
    // W2d lowmem: stream the level download through 1 MiB slices and hash
    // incrementally — pre-patch this held bytes/words/canon/cbytes of the
    // 128 MiB level-22 buffer all at once (~4x128 MiB host peak).
    for lvl in [13u32, 14, 16, 18, 20, 22] {
        if lvl >= t {
            continue; // level must exist in-tree (levels 1..=t-1)
        }
        let tag = format!("dbg_{}_lvl{}{}_cpu.bin", inst.name, lvl, "");
        let p = inputs_dir.join(&tag);
        let meta = std::fs::metadata(&p).ok();
        if meta.map(|m| m.len() == 0).unwrap_or(true) {
            continue; // golden absent/empty for this t
        }
        use sha2::Digest as _;
        let mut hasher = sha2::Sha256::new();
        let lvl_bytes = (1u64 << lvl) * 32;
        let mut total_cbytes = 0u64;
        let mut off = 0u64;
        while off < lvl_bytes {
            let n = (1u64 << 20).min(lvl_bytes - off);
            let cbytes: Vec<u8> = levels[(lvl as usize) - 1]
                .read_words_async(dev, off / 4, n / 4)
                .await
                .into_iter()
                .flat_map(|w| decode(w).to_le_bytes())
                .collect();
            hasher.update(&cbytes);
            total_cbytes += cbytes.len() as u64;
            off += n;
        }
        snap_shas.push(LevelSnap {
            lvl,
            tree_level: lvl,
            tag,
            sha256: format!("{:x}", hasher.finalize()),
            bytes: total_cbytes,
        });
    }

    // 7) window pool: one wsz-row window (160 B/row) per slot. wsz = 2^22
    // (640 MiB slots) on ANV, 2^19 (80 MiB) under WebGPU-floor limits.
    // Slot count covers the largest round window count (nwin max =
    // 2^(t-2)/wsz; ANV: 4, floors t=26: 32 x 80 MiB = 2.5 GiB < ANV's 4 x
    // 640 MiB). + eqv (sharded) + spart.
    let wsz: u32 = dev.gkr_wsz();
    let pool_size = wsz as u64 * (ROW10 as u64) * 4;
    // Slot allocator at each fold stage needs ALL nwin input windows live
    // simultaneously + 1 free (pending set excludes future inputs). Max
    // nwin occurs at the first fold (rows = 2^(t-1)): nslots = nwin + 2.
    let nslots: u32 = (((1u64 << (t - 1)) / wsz as u64) + 2).max(6) as u32;
    let pool: Vec<wgpu::Buffer> = (0..nslots)
        .map(|c| {
            dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("gkr-win{}", c)),
                size: pool_size,
                usage: BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();
    let eqv = ShardBuf::new(
        dev,
        "gkr-eqv",
        BU::STORAGE | BU::COPY_DST,
        (1u64 << (t - 1)) * 16,
    );
    let spart_max = if t - 2 >= 9 {
        1u64 << (t - 2 - 9)
    } else {
        1
    };
    let spart_a = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gkr-spart-a"),
        size: spart_max * 48,
        usage: BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
        mapped_at_creation: false,
    });
    let spart_b = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gkr-spart-b"),
        size: (spart_max / 2).max(1) * 48,
        usage: BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut round_outs: Vec<RoundOut> = Vec::new();

    // 8) rounds j = 1..=t-1; windowed (2^22-row) unpack/seval/fold over the 4-slot pool
    for (ri, round) in inst.rounds.iter().enumerate() {
        let j = round.j;
        debug_assert_eq!(j as usize, ri + 1);

        // eq vector from recorded xi (canonical)
        let xi_c: Vec<u64> = inst.per_round_xi[&j].clone();
        let mut xis = Vec::with_capacity(j as usize);
        for x in xi_c.chunks(4) {
            xis.push(ef_encode4(x));
        }
        // recorded CPU prover lays eq hypercube bit0 = OLDEST challenge:
        // per_round_xi is chronological, and eq_vec_host binds xis[i] to bit i,
        // so feed it un-reversed (AT5 bug: reversing made bit0 = newest,
        // scrambling eq<->row pairing for every round j >= 2).
        // W2d lowmem: eq hypercube for j=25 was 512 MiB host-resident pre-patch.
        // For j > 20 build eq as (low-20-bits table) x (high-bits coefficients)
        // and upload block by block — identical field values (products are
        // associative/commutative mod p; Mont encoding is canonical).
        eq_upload_split(&dev, &eqv, &xis);
        let lam = ef_encode4(&round.lambda);
        let mut dbg_lines: Vec<String> = Vec::new();
        let dbg_on = (j == 1 || j == 2 || j == 24 || j == 25) && std::env::var("W2A_DBG_DIR").is_ok();
        if dbg_on {
            let eqc = raw_to_canon(&eq_vec_host(&xis));
            dbg_lines.push(format!("eq_first8 {:?}", &eqc[..8.min(eqc.len())]));
            dbg_lines.push(format!("lam_raw {:?}", lam));
            dbg_lines.push(format!("xis_raw {:?}", xis));
            if j + 1 < t {
                let lw = levels[1].read_all_async(dev).await;
                dbg_lines.push(format!("lvl{}{:?}", j + 1, raw_to_canon(&lw)));
            }
        }

        // window slots: current window index -> pool slot
        // 6 slots: a 4->2 window fold needs 4 live inputs + 2 fresh
        // outputs (AT5 bug: with 4 slots the w=0 output clobbered w=1's
        // still-pending input window, corrupting round-25 s_evals).
        let mut slots: Vec<u32> = (0..nslots).collect();
        let mut r_cur = 1u32 << (j - 1); // current row count
        // AT5 bug: nwin=1 hardcoded overflowed the 2^22-row pool slots once
        // rows (2^(j-1)) exceeded wsz (t=26 circuits, rounds 24-25).
        let mut nwin: u32 = (r_cur / wsz).max(1);
        let mut subs_out: Vec<Vec<u32>> = Vec::with_capacity(round.subs.len());

        // unpack: level-(j+1) segment -> windows
        {
            let rows_w = r_cur / nwin;
            for w in 0..nwin {
                let eq_base = w as u64 * rows_w as u64 * 32; // 8 u32/row
                let dst = &pool[slots[w as usize] as usize];
                let mut done = 0u32;
                while done < rows_w {
                    let wgs = ((rows_w - done + 255) / 256).min(65535);
                    let rows_disp = ((rows_w - done) as u64).min(wgs as u64 * 256);
                    // level-(j+1) always sits at index 1 after earlier removes
                    let r0 = if j + 1 == t {
                        bres(&leaf_bufs[w as usize], done as u64 * 128, rows_disp * 128)
                    } else {
                        levels[1].bind(
                            dev,
                            w as u64 * rows_w as u64 * 128 + done as u64 * 128,
                            rows_disp * 128,
                        )
                    };
                    let r1 = eqv.bind(dev, eq_base + done as u64 * 32, rows_disp * 32);
                    pipe.set_uniform(dev, W_FRAC, lam, [0; 4], rows_w, (done * 40) as u32, 0);
                    let bg = pipe.bind3(dev, r0, r1, dst.as_entire_binding());
                    GkrPipeline::run(dev, &pipe.pipe_unpack, &bg, wgs);
                    done += wgs * 256;
                }
                if dbg_on && w == 0 {
                    let rb = download_n_async(dev, dst, 320).await;
                    let rw: Vec<u32> = rb
                        .chunks(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    dbg_lines.push(format!("row01 {:?}", raw_to_canon(&rw)));
                }
            }
        }

        for (k, sub) in round.subs.iter().enumerate() {
            let h = j - 1 - k as u32;
            let r = ef_encode4(&sub.r);

            let mut s_acc: Vec<u32> = vec![0u32; 12];
            // seval over each window; host accumulates window partials
            for w in 0..nwin {
                let rows_w = r_cur / nwin;
                let src = &pool[slots[w as usize] as usize];
                let blocks = if rows_w >= 512 { rows_w / 512 } else { 1 };
                let sbase_dbg: u32 = if dbg_on && k == 0 && w == 0 { 43981 } else { 0 };
                pipe.set_uniform(dev, W_STRIPLE, lam, r, rows_w, 0, sbase_dbg);
                let bg = pipe.bind3(
                    dev,
                    src.as_entire_binding(),
                    pipe.dummy.as_entire_binding(),
                    spart_a.as_entire_binding(),
                );
                GkrPipeline::run(dev, &pipe.pipe_seval, &bg, blocks);
                if dbg_on && k == 0 && w == 0 {
                    let rb = download_n_async(dev, &spart_a, (156 * 4).min(spart_a.size())).await;
                    let rw: Vec<u32> = rb
                        .chunks(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    if rw.len() >= 60 {
                        dbg_lines.push(format!("vraw_mont {:?}", &rw[48..60]));
                    }
                    if rw.len() >= 100 {
                        dbg_lines.push(format!("x2dbg_mont {:?}", raw_to_canon(&rw[60..100])));
                    }
                    if rw.len() >= 156 {
                        dbg_lines.push(format!("treedbg_mont {:?}", raw_to_canon(&rw[140..156])));
                    }
                }
                let mut nblk = blocks;
                let (mut sa, mut sb) = (&spart_a, &spart_b);
                while nblk > 1 {
                    pipe.set_uniform(dev, W_STRIPLE, lam, r, rows_w, 0, 0);
                    let bg = pipe.bind3(
                        dev,
                        sa.as_entire_binding(),
                        pipe.dummy.as_entire_binding(),
                        sb.as_entire_binding(),
                    );
                    GkrPipeline::run(dev, &pipe.pipe_tree, &bg, ((nblk / 2) as u32 + 255) / 256);
                    std::mem::swap(&mut sa, &mut sb);
                    nblk /= 2;
                }
                let sbytes = download_n_async(dev, sa, 48).await;
                let su32: Vec<u32> = sbytes
                    .chunks(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                if dbg_on && k == 0 && w == 0 {
                    dbg_lines.push(format!("spart0 {:?}", raw_to_canon(&su32)));
                }
                if std::env::var("W2A_TRACE").is_ok() && j <= 4 {
                    eprintln!(
                        "TRACE j={j} k={k} rows_w={rows_w} nwin={nwin} r_cur={r_cur} blocks={blocks} su32={:?} seval0={:?}",
                        &su32[..8.min(su32.len())],
                        &raw_to_canon(&s_acc)[..4.min(s_acc.len())]
                    );
                }
                for (a, b) in s_acc.iter_mut().zip(su32.iter()) {
                    *a = fadd(*a, *b);
                }
            }

            if std::env::var("W2A_TRACE2").is_ok() && j == 2 && k == 0 {
                let pr = download_off_async(dev, &pool[slots[0] as usize], 0, 320).await;
                let pw: Vec<u32> = pr.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                eprintln!("TRACE2 poolrow0_1={:?}", &raw_to_canon(&pw)[..20.min(pw.len())]);
                eprintln!("TRACE2 lvl_head={:?}", &levels[1].read_words_async(dev, 0, 8).await);
            }
            let s_evals = raw_to_canon(&s_acc);
            if dbg_on && k == 0 {
                dbg_lines.push(format!("s_acc {:?}", s_evals));
            }
            subs_out.push(s_evals);

            // fold: r_cur -> r_cur/2 (output windows draw on 2 input windows each)
            if h >= 1 {
                let r_out = r_cur / 2;
                let nwin_out = if r_out >= wsz { r_out / wsz } else { 1 };
                let rows_w_out = r_out / nwin_out;
                let k = nwin / nwin_out; // input windows per output window (2 or 1)
                let mut new_slots = Vec::with_capacity(nwin_out as usize);
                for w in 0..nwin_out {
                    let in0 = slots[(k * w) as usize];
                    let in1 = if k == 2 { slots[(k * w + 1) as usize] } else { in0 };
                    let pending: Vec<u32> = slots[((k * w) as usize + k as usize)..(nwin as usize)].to_vec();
                    let out = (0..nslots).find(|i| *i != in0 && *i != in1 && !new_slots.contains(i) && !pending.contains(i)).unwrap();
                    new_slots.push(out);
                    // dispatch m: input window (k*w + m) -> output rows [m*rows_w_out/2 ...]
                    // each input window has 2*rows_w_out rows (k=2) -> rows_w_out out rows
                    // input rows per input window; each fold-halves its own
                    // window (AT5 bug: 2*rows_w_out overran the window when
                    // k == 2 with nwin > 1, folding past the window boundary).
                    let in_rows = r_cur / nwin;
                    let mut off = 0u64;
                    for m in 0..k {
                        let inslot = slots[((k * w + m) as usize)];
                        let src = &pool[inslot as usize];
                        let dst = &pool[out as usize];
                        let half = in_rows / 2;
                        let mut done = 0u32;
                        while done < half {
                            let wgs = ((half - done + 255) / 256).min(65535);
                            pipe.set_uniform(dev, W_FRAC, lam, r, half, ((off + done as u64 * 40) as u32), 0);
                            let bg = pipe.bind3(
                                dev,
                                src.as_entire_binding(),
                                pipe.dummy.as_entire_binding(),
                                dst.as_entire_binding(),
                            );
                            GkrPipeline::run(dev, &pipe.pipe_fold, &bg, wgs);
                            done += wgs * 256;
                        }
                        off += half as u64 * 40; // u32 units (40 per row)
                    }
                }
                slots = new_slots;
                nwin = nwin_out;
                r_cur = r_out;
                if dbg_on && subs_out.len() == 1 {
                    let wmax = slots.len().min(2);
                    for (wi, ridx) in [(0usize, 0usize), (0, (1 << 21) - 1), (0, 1 << 21), (1, 0), (1, (1 << 21) - 1), (1, 1 << 21)] {
                        if wi >= wmax { continue; }
                        let rb = download_off_async(dev, &pool[slots[wi] as usize], (ridx * 160) as u64, 160).await;
                        let rw: Vec<u32> = rb
                            .chunks(4)
                            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        dbg_lines.push(format!("frow w{} r{} {:?},", wi, ridx, raw_to_canon(&rw)));
                    }
                }
            }
        }

        // claims = final single row (p0e, q0e, p1e, q1e) in the final pool slot
        let final_slot = slots[0];
        // claims = final row e/o dim evaluated at the LAST sub's r (the final
        // in-row fold the kernel loop skips when h==0): (p0,q0,p1,q1).
        let cbytes = download_n_async(dev, &pool[final_slot as usize], 160).await;
        let cu32: Vec<u32> = cbytes
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let rl = ef_encode4(&round.subs[round.subs.len() - 1].r);
        let mut claims = Vec::with_capacity(16);
        let mut push_pair = |off_e: usize, off_o: usize, claims: &mut Vec<u32>| {
            let v = fold_pair(&cu32[off_e..off_e + 4], &cu32[off_o..off_o + 4], &rl);
            claims.extend_from_slice(&v);
        };
        push_pair(8, 16, &mut claims);  // p0
        push_pair(24, 28, &mut claims); // q0
        push_pair(12, 20, &mut claims); // p1
        push_pair(32, 36, &mut claims); // q1
        let claims = raw_to_canon(&claims);
        if dbg_on {
            dbg_lines.push(format!("claims_row40 {:?}", raw_to_canon(&cu32)));
            if let Ok(dir) = std::env::var("W2A_DBG_DIR") {
                let _ = std::fs::create_dir_all(&dir);
                let _ = std::fs::write(
                    Path::new(&dir).join(format!("dbg-{}-r{}.txt", inst.name, j)),
                    dbg_lines.join("\n"),
                );
            }
        }
        round_outs.push(RoundOut {
            j,
            subs: subs_out,
            claims,
        });

        // free the now-unneeded level-(j+1) buffer (memory law)
        if j + 1 < t {
            levels.remove(1);
        }
    }

    let mut all = Vec::new();
    all.extend_from_slice(leaf_sha.as_bytes());
    for v in &root_p {
        all.extend_from_slice(&v.to_le_bytes());
    }
    for v in &root_q {
        all.extend_from_slice(&v.to_le_bytes());
    }
    for v in &claims0 {
        all.extend_from_slice(&v.to_le_bytes());
    }
    for r in &round_outs {
        for s in &r.subs {
            for v in s {
                all.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in &r.claims {
            all.extend_from_slice(&v.to_le_bytes());
        }
    }

    let _ = (leaf_bufs, levels, pool, spart_a, spart_b, eqv);

    GkrOutputs {
        name: inst.name.clone(),
        t,
        root_p: root_p.to_vec(),
        root_q: root_q.to_vec(),
        claims0,
        levels: std::mem::take(snap_shas),
        rounds: round_outs,
        leaf_sha256: leaf_sha,
        out_sha256: sha256_hex(&all),
    }
}

/// Compare a GPU run against the recorded expected values; returns mismatch list.
pub fn check_expected(inst: &GkrInstance, out: &GkrOutputs) -> Vec<String> {
    let mut m = Vec::new();
    let mut eq4 = |tag: &str, got: &[u32], want: &[u64]| {
        if got.len() != want.len() {
            m.push(format!("{tag}: len {} != {}", got.len(), want.len()));
            return;
        }
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if *g as u64 != *w {
                m.push(format!("{tag}[{}]: gpu {} != cpu {}", i, g, w));
            }
        }
    };
    eq4("root_p", &out.root_p, &inst.root_p);
    eq4("root_q", &out.root_q, &inst.root_q);
    eq4("claims0", &out.claims0, &inst.claims0);
    for (ri, round) in inst.rounds.iter().enumerate() {
        let got = &out.rounds[ri];
        for (si, (g, w)) in got.subs.iter().zip(round.subs.iter()).enumerate() {
            eq4(&format!("r{} s{}", round.j, si), g, &w.s_evals);
        }
        eq4(&format!("r{} claims", round.j), &got.claims, &round.claims);
    }
    m
}

/// Compare snapshot shas against files on disk; returns mismatch list.
pub fn check_snapshots(_inst: &GkrInstance, out: &GkrOutputs, inputs_dir: &Path) -> Vec<String> {
    let mut m = Vec::new();
    for snap in &out.levels {
        let p = inputs_dir.join(&snap.tag);
        match std::fs::read(&p) {
            Ok(data) => {
                let h = sha256_hex(&data);
                if h != snap.sha256 {
                    m.push(format!(
                        "{}: gpu {} != cpu {} ({}B)",
                        p.display(),
                        snap.sha256,
                        h,
                        data.len()
                    ));
                }
            }
            Err(e) => m.push(format!("{}: {}", p.display(), e)),
        }
    }
    m
}
#[cfg(test)]
mod seval_micro_tests {
    use super::*;

    /// Replay agg-leaf r2 (rows 0-1, lam from dbg) directly through
    /// pipe_seval; expected spart = golden [790040390, 430551731, ...].
    #[test]
    fn seval_pair_replay() {
        let row01: [u32; 80] = [595834354, 732908907, 684368567, 1930617988, 1159597387, 1425279555, 612323965, 834472346, 936940859, 939081072, 134873979, 922914657, 1227610087, 135176543, 1849176746, 1190405903, 257346266, 1441918344, 225573001, 991005037, 1432915987, 1756303729, 579993599, 1546432696, 1942537645, 776071076, 1323791147, 442811544, 495282974, 1189887971, 1544107344, 524028928, 143653583, 752259352, 1477218815, 674342250, 6285683, 455970147, 1599723998, 1367144960, 231556760, 1328778703, 1007685622, 1485122408, 26277421, 539564677, 1722153688, 1789585021, 490002816, 1452212714, 1680721424, 1294375567, 822140766, 1444066503, 1544086663, 1790668459, 1997027410, 318724680, 919576661, 105971627, 0, 0, 0, 0, 1905717363, 829453714, 184413351, 1741975069, 203606886, 274642633, 1387889115, 1361056572, 1588111336, 1416348283, 161659536, 40406250, 229552718, 1314133020, 1976436645, 447596186];
        let lam = [69905339u32, 683653480, 1098732639, 801423575];
        let dev = crate::device::WgpuDevice::init(crate::device::InitOpts {
            require_intel_arc: true,
            allow_u32_only: true,
        });
        let pipe = GkrPipeline::new(&dev);
        let rows = 512usize;
        let mut raw = vec![0u32; rows * 40];
        for i in 0..80 {
            raw[i] = crate::field::encode(row01[i]);
        }
        let poolbuf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("micro-pool"),
            size: (rows * 160) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&poolbuf, 0, &u32s_to_bytes(&raw));
        let spart = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("micro-spart"),
            size: 4096,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        pipe.set_uniform(&dev, W_STRIPLE, lam, [0u32; 4], 2, 0, 0);
        let bg = pipe.bind3(
            &dev,
            poolbuf.as_entire_binding(),
            pipe.dummy.as_entire_binding(),
            spart.as_entire_binding(),
        );
        let dispatch = |rawv: &Vec<u32>, nrows: u32, dev: &crate::device::WgpuDevice| -> Vec<u32> {
            dev.queue.write_buffer(&poolbuf, 0, &u32s_to_bytes(rawv));
            pipe.set_uniform(&dev, W_STRIPLE, lam, [0u32; 4], nrows, 0, 0);
            let bg = pipe.bind3(
                &dev,
                poolbuf.as_entire_binding(),
                pipe.dummy.as_entire_binding(),
                spart.as_entire_binding(),
            );
            GkrPipeline::run(&dev, &pipe.pipe_seval, &bg, 1);
            let out = download_n(&dev, &spart, 4096);
            let w: Vec<u32> = out.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let mut c = raw_to_canon(&w);
            c.truncate(4);
            let lane1 = raw_to_canon(&w[160..164]);
            println!("SEVALTEST   lane1_v1={:?} probe=[{} {} {} {}] canary=[{} {} {} {}] blk0={:?} blk1={:?} blk2={:?} id0={} id1={}", &lane1[..1], w[164], w[165], w[166], w[167], w[176], w[177], w[178], w[179], &w[0..12], &w[12..24], &w[24..36], w[184], w[185]);
            c
        };
        let mut r0only = vec![0u32; rows * 40];
        r0only[..40].copy_from_slice(&raw[..40]);
        let mut r1only = vec![0u32; rows * 40];
        r1only[..40].copy_from_slice(&raw[40..80]);
        println!("SEVALTEST r0-only n1 = {:?}", &dispatch(&r0only, 1, &dev)[..3]);
        println!("SEVALTEST r1-only n1 = {:?}", &dispatch(&r1only, 1, &dev)[..3]);
        println!("SEVALTEST r0row-as-r1 n2 = {:?}", &dispatch(&{ let mut v = raw.clone(); v[40..80].copy_from_slice(&raw[..40]); v[0..40].copy_from_slice(&raw[..40]); v }, 2, &dev)[..3]);
        let c = dispatch(&raw, 2, &dev);
        println!("SEVALTEST canon={:?}", &c[..3]);
        assert_eq!(&c[..3], &[790040390u32, 430551731, 201269746], "seval pair replay mismatch");
    }
}
