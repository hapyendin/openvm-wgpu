// W2b phase-2: WGSL logup LEAF BUILD (b9drv2 semantics, cpu9 world).
//
// Leaf = 2^log_stack rows of Frac {p: EF4, q: EF4} in Montgomery-RAW u32.
// Fill law (b9drv2.cpp / replay.py, byte-exact-verified in phase 1):
//   default: p = (0,0,0,0); q = alpha                       (all rows)
//   per slot (air pos, interaction j, row_idx, log_ht):
//     len = 2^log_ht; reps = len/h; norm = reps^{-1} (raw)
//     for d in 0..len:  r = d % h
//       p = (mont(pnum[j][r], norm), 0, 0, 0)
//       q = eadd(den[j][r], alpha)
// pnum/den come from the per-air DAG eval kernel (dag_rows), a 1:1 port of
// replay.py's node machine (kinds 0..8 over per-row data: sels + trace parts
// with optional rot row + pv refs; nodes are scalar BB; den = sum bp[k]*node +
// bp[ml]*raw(bus+1), pnum = node[cnt]).
//
// Determinism law: fixed-index, integer-only, no atomics, no floats.
// Buffer law: GPU buffers <= 1 GiB (leaf chunked in 512 MiB pieces).

use std::collections::HashMap;
use std::path::Path;

use wgpu::BufferUsages as BU;

use crate::device::WgpuDevice;
use crate::field::{decode, encode};
use crate::limitsbuf::ShardBuf;

pub const LOGUP_WGSL: &str = r#"
const PP: u32 = 2013265921u;

// `mul` supplied by wgsl::mul_prelude (u64-REDC or pure-u32 limb path)
fn addm(a: u32, b: u32) -> u32 {
  var s = a + b;
  if (s >= PP) { s = s - PP; }
  return s;
}
fn sub1(a: u32, b: u32) -> u32 {
  var s = a + (PP - b);
  if (s >= PP) { s = s - PP; }
  if (s >= PP) { s = s - PP; }
  return s;
}
// two-level reduced add/sub (phase-1 law: single-level leaves >=P words)
fn addp4(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
  let s = a + b;
  var r = select(s, s - vec4<u32>(PP), s >= vec4<u32>(PP));
  r = select(r, r - vec4<u32>(PP), r >= vec4<u32>(PP));
  return r;
}
// EF (x^4 = 11) ops, Montgomery-raw limbs (identical to gkr_wgsl.rs)
const W11: u32 = 939524073u;
fn emul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
  let s0 = addm(mul(a.y, b.w), addm(mul(a.z, b.z), mul(a.w, b.y)));
  let r0 = addm(mul(a.x, b.x), mul(W11, s0));
  let s1 = addm(mul(a.z, b.w), mul(a.w, b.z));
  let r1 = addm(addm(mul(a.x, b.y), mul(a.y, b.x)), mul(W11, s1));
  let s2 = mul(a.w, b.w);
  let r2 = addm(addm(addm(mul(a.x, b.z), mul(a.y, b.y)), mul(a.z, b.x)), mul(W11, s2));
  let r3 = addm(addm(addm(mul(a.x, b.w), mul(a.y, b.z)), mul(a.z, b.y)), mul(a.w, b.x));
  return vec4<u32>(r0, r1, r2, r3);
}
// den += bp * scalar: scalar operand is (s,0,0,0) -> per-limb mont mul
fn scal_ef(b: vec4<u32>, s: u32) -> vec4<u32> {
  return vec4<u32>(mul(b.x, s), mul(b.y, s), mul(b.z, s), mul(b.w, s));
}

struct LU {
  w0: u32, w1: u32, w2: u32, w3: u32,
  w4: u32, w5: u32, w6: u32, w7: u32,
  lam: vec4<u32>,   // alpha (leaf kernels)
  r:   vec4<u32>,   // r.x = chunk stride words (leaf_fill)
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> src2: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> u: LU;

// fill_default: dst[i] = (0,0,0,0, alpha) at base u.w0, count u.w1
@compute @workgroup_size(256u)
fn fill_default(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.w1) { return; }
  let o = u.w0 + i * 8u;
  dst[o] = 0u; dst[o + 1u] = 0u; dst[o + 2u] = 0u; dst[o + 3u] = 0u;
  dst[o + 4u] = u.lam.x; dst[o + 5u] = u.lam.y;
  dst[o + 6u] = u.lam.z; dst[o + 7u] = u.lam.w;
}

// leaf_fill: one slot region (row range of one leaf chunk).
// w0=dst_base(w) w1=cnt w2=r0mod w3=h w4=norm w5=pnum_base' w6=den_base'
// w7=rows_per_chunk; lam=alpha; r.x=chunk_stride_words (n_int*rows_per_chunk).
// pnum_base' = pbase + j*w7, den_base' = dbase + j*w7*4 (region [chunk][j][row]).
@compute @workgroup_size(256u)
fn leaf_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.w1) { return; }
  let hr = (u.w2 + i) % u.w3;
  let c = hr / u.w7;
  let ri = hr % u.w7;
  let off = c * u.r.x + ri;
  let pv = mul(src[u.w5 + off], u.w4);
  let db = u.w6 + off * 4u;
  let d = vec4<u32>(src[db], src[db + 1u], src[db + 2u], src[db + 3u]);
  let qv = addp4(d, u.lam);
  let o = u.w0 + i * 8u;
  dst[o] = pv; dst[o + 1u] = 0u; dst[o + 2u] = 0u; dst[o + 3u] = 0u;
  dst[o + 4u] = qv.x; dst[o + 5u] = qv.y; dst[o + 6u] = qv.z; dst[o + 7u] = qv.w;
}

// dag_rows: eval the air DAG for a row-chunk.
// w0=chunk_row0 w1=valid_rows w2=scratch_base(0) w3=pnum_base(chunk)
// w4=den_base(chunk) w5=intstride w6=stride_rows(=full chunk_rows)
// src = air blob: [header|partmeta|kinds4|ints|pv|bp|trace].
@compute @workgroup_size(256u)
fn dag_rows(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_id) lid: vec3<u32>) {
  // gid.x ALREADY = wg*256 + local (AT: lr = gid.x*256 + lid.x double-counted)
  let lr = gid.x;
  if (lr >= u.w1) { return; }
  let r = u.w0 + lr;
  let h = src[2];
  let npart = src[3];
  let need_rot = src[4];
  let kinds4 = src[5];
  let intb = src[6];
  let pvb = src[7];
  let bpb = src[8];
  let pmeta = src[12];
  let sel1 = 268435454u;  // [1]_Mont
  let sel_f: u32 = select(0u, sel1, r == 0u);
  let sel_l: u32 = select(0u, sel1, r == h - 1u);
  let sel_t: u32 = select(sel1, 0u, r == h - 1u);
  let r2 = select(r + 1u, 0u, r + 1u == h);
  for (var n: u32 = 0u; n < src[0]; n = n + 1u) {
    let kb = kinds4 + n * 4u;
    let kind = src[kb];
    let ia = src[kb + 1u];
    let ic = src[kb + 3u];
    var v: u32 = 0u;
    if (kind == 0u) {
      if (ia == 1u) {
        v = src[pvb + ic];
      } else {
        let ib = src[kb + 2u];
        let part = ib >> 1u;
        let rot = ib & 1u;
        let rr = select(r, r2, rot == 1u && need_rot == 1u);
        // trace cells come from the per-chunk window bound at src2:
        // window layout = [0..64) per-part row bases (words), rows from 64.
        // part p rows (cr+1 of them, starting at row c0 = u.w0) sit at
        // 64 + winhdr[part] + (rr - c0)*stride_p + ic.
        let sp = src[pmeta + 2u * npart + part];
        v = src2[64u + src2[part] + (rr - u.w0) * sp + ic];
      }
    } else if (kind == 1u) { v = sel_f;
    } else if (kind == 2u) { v = sel_l;
    } else if (kind == 3u) { v = sel_t;
    } else if (kind == 4u) { v = ia;
    } else {
      let ib2 = src[kb + 2u];
      let a = dst[u.w2 + ia * u.w6 + lr];
      if (kind == 5u) {
        let b = dst[u.w2 + ib2 * u.w6 + lr];
        v = addm(a, b);
      } else if (kind == 6u) {
        let b = dst[u.w2 + ib2 * u.w6 + lr];
        v = sub1(a, b);
      } else if (kind == 7u) {
        v = select(PP - a, 0u, a == 0u);
      } else {
        let b = dst[u.w2 + ib2 * u.w6 + lr];
        v = mul(a, b);
      }
    }
    dst[u.w2 + n * u.w6 + lr] = v;
  }
  let nint = src[1];
  for (var j: u32 = 0u; j < nint; j = j + 1u) {
    let rowb = intb + j * u.w5;
    let ml = src[rowb + 1u];
    let b1r = src[rowb + 2u];
    var den = vec4<u32>(0u, 0u, 0u, 0u);
    for (var k2: u32 = 0u; k2 < ml; k2 = k2 + 1u) {
      let nd = src[rowb + 4u + k2];
      let nv = dst[u.w2 + nd * u.w6 + lr];
      let bpv = vec4<u32>(src[bpb + k2 * 4u], src[bpb + k2 * 4u + 1u],
                          src[bpb + k2 * 4u + 2u], src[bpb + k2 * 4u + 3u]);
      den = addp4(den, scal_ef(bpv, nv));
    }
    den = addp4(den, scal_ef(
      vec4<u32>(src[bpb + ml * 4u], src[bpb + ml * 4u + 1u],
                src[bpb + ml * 4u + 2u], src[bpb + ml * 4u + 3u]), b1r));
    let pnum = dst[u.w2 + src[rowb] * u.w6 + lr];
    dst[u.w3 + j * u.w6 + lr] = pnum;
    let doff = u.w4 + (j * u.w6 + lr) * 4u;
    dst[doff] = den.x; dst[doff + 1u] = den.y;
    dst[doff + 2u] = den.z; dst[doff + 3u] = den.w;
  }
}
"#;

pub struct LeafPipeline {
    pub bgl: wgpu::BindGroupLayout,
    pub p_fill_default: wgpu::ComputePipeline,
    pub p_leaf_fill: wgpu::ComputePipeline,
    pub p_dag: wgpu::ComputePipeline,
    uni: wgpu::Buffer,
    dummy: wgpu::Buffer,
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

impl LeafPipeline {
    pub fn new(dev: &WgpuDevice) -> LeafPipeline {
        let module = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("logup-w2b"),
            source: wgpu::ShaderSource::Wgsl(
                crate::wgsl::with_prelude(dev.has_shader_int64, LOGUP_WGSL).into(),
            ),
        });
        let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("logup-4slot"),
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
        let uni = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logup-uni"),
            size: 64,
            usage: BU::UNIFORM | BU::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logup-dummy"),
            size: 16,
            usage: BU::STORAGE | BU::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&dummy, 0, &[0u8; 16]);
        LeafPipeline {
            bgl,
            p_fill_default: mk("fill_default"),
            p_leaf_fill: mk("leaf_fill"),
            p_dag: mk("dag_rows"),
            uni,
            dummy,
        }
    }

    pub fn set_uniform(&self, dev: &WgpuDevice, w: [u32; 8], lam: [u32; 4], rx: u32, ry: u32) {
        let mut b = [0u8; 64];
        for (i, x) in w.iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
        for (i, x) in lam.iter().enumerate() {
            b[32 + i * 4..36 + i * 4].copy_from_slice(&x.to_le_bytes());
        }
        b[48..52].copy_from_slice(&rx.to_le_bytes());
        b[52..56].copy_from_slice(&ry.to_le_bytes());
        dev.queue.write_buffer(&self.uni, 0, &b);
    }

    fn bg_res<'a>(
        &self,
        dev: &WgpuDevice,
        r0: wgpu::BindingResource<'a>,
        r1: wgpu::BindingResource<'a>,
        r2: wgpu::BindingResource<'a>,
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

    fn bg(&self, dev: &WgpuDevice, b0: &wgpu::Buffer, b2: &wgpu::Buffer) -> wgpu::BindGroup {
        dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: b0, offset: 0, size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.dummy, offset: 0, size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: b2, offset: 0, size: None,
                    }),
                },
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

    fn download(dev: &WgpuDevice, buf: &wgpu::Buffer) -> Vec<u8> {
        let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logup-stage"),
            size: buf.size(),
            usage: BU::COPY_DST | BU::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, buf.size());
        dev.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        dev.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .ok();
        let data = slice.get_mapped_range().map(|d| d.to_vec()).unwrap_or_default();
        drop(slice);
        staging.unmap();
        data
    }
}

// ---------------- leaf-world model ----------------

#[derive(serde::Deserialize, Clone)]
pub struct AirSpec {
    pub air_id: u32,
    #[allow(dead_code)]
    pub pos: u32,
    pub h: u32,
    pub n_int: u32,
    pub n_nodes: u32,
    #[allow(dead_code)]
    pub need_rot: u32,
    #[allow(dead_code)]
    pub widths: Vec<u32>,
    pub trace_files: Vec<String>,
    pub tables: String,
    pub trace_words: u64,
    pub tblen: u32,
}

#[derive(serde::Deserialize, Clone)]
pub struct SlotSpec {
    #[allow(dead_code)]
    pub pos: u32,
    pub air_id: u32,
    pub j: u32,
    pub row_idx: u64,
    pub log_ht: u32,
    pub h: u32,
    pub norm_raw: u32,
}

#[derive(serde::Deserialize)]
pub struct LeafWorld {
    pub log_stack: u32,
    pub alpha_canon: Vec<u64>,
    pub airs: Vec<AirSpec>,
    pub slots: Vec<SlotSpec>,
}

pub type TraceCache = HashMap<u32, Vec<u32>>;

fn load_trace(cache: &mut TraceCache, inputs_dir: &Path, traces_dir: &Path, air: &AirSpec) -> Vec<u32> {
    if let Some(v) = cache.get(&air.air_id) {
        return v.clone();
    }
    let mut raw: Vec<u32> = Vec::with_capacity(air.trace_words as usize);
    for tf in &air.trace_files {
        let bytes = std::fs::read(traces_dir.join(tf))
            .unwrap_or_else(|e| panic!("trace {}: {e}", tf));
        assert_eq!(bytes.len() % 8, 0);
        for c in bytes.chunks(8) {
            let w = u64::from_le_bytes(c.try_into().unwrap());
            raw.push(encode(w as u32));
        }
    }
    assert_eq!(raw.len() as u64, air.trace_words, "trace size air{}", air.air_id);
    let _ = inputs_dir;
    cache.insert(air.air_id, raw.clone());
    raw
}

/// Build the full 2^log_stack leaf as raw-Mont chunk buffers on the GPU.
/// Returns (leaf chunks [chunk_bytes], canonical-form sha256 of the leaf).
pub fn build_leaf(
    dev: &WgpuDevice,
    lp: &LeafPipeline,
    world: &LeafWorld,
    inputs_dir: &Path,
    traces_dir: &Path,
    cache: &mut TraceCache,
    chunk_bytes: u64,
) -> (Vec<wgpu::Buffer>, String) {
    let n = 1u64 << world.log_stack; // rows
    let alpha: Vec<u32> = world.alpha_canon.iter().map(|v| encode(*v as u32)).collect();
    let alpha4 = [alpha[0], alpha[1], alpha[2], alpha[3]];

    let per = chunk_bytes;
    assert_eq!(n * 32 % per, 0);
    let nchunks = (n * 32 / per) as usize;
    let leaf: Vec<wgpu::Buffer> = (0..nchunks)
        .map(|c| {
            dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("w2b-leaf{c}")),
                size: per,
                usage: BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();
    let rpc_leaf = per / 32; // leaf rows per chunk
    // 1) default fill: p=0, q=alpha
    for cb in &leaf {
        let mut done = 0u64;
        while done < rpc_leaf {
            let cnt = (rpc_leaf - done).min(65535 * 256);
            lp.set_uniform(dev, [(done * 8) as u32, cnt as u32, 0, 0, 0, 0, 0, 0], alpha4, 0, 0);
            let bg = lp.bg(dev, &lp.dummy, cb);
            LeafPipeline::run(dev, &lp.p_fill_default, &bg, ((cnt + 255) / 256) as u32);
            done += cnt;
        }
    }

    // 2) per-air DAG + slot fills (sorted order as listed in world.airs)
    eprintln!("LEAFB dagslots start");
    for air in &world.airs {
        let tables = std::fs::read(inputs_dir.join(&air.tables))
            .unwrap_or_else(|e| panic!("tables {}: {e}", air.tables));
        assert_eq!(tables.len() / 4, air.tblen as usize);
        let intstride = u32::from_le_bytes(tables[40..44].try_into().unwrap()) + 4; // 4 + maxmsg (hdr w10)
        // W2d lowmem: stream tables + trace words straight into the blob buffer
        // in 4 MiB-encoded chunks (peak host cost == chunk, not the full air
        // blob: the largest air's blob was 344 MiB resident pre-patch).
        let blob_len = tables.len() as u64 + air.trace_words * 4;
        // Phase-2: the air blob can exceed both the 256 MiB buffer cap and
        // the 128 MiB binding cap -> sharded; kernels read only a per-chunk
        // trace window (copied device-side) plus the small tables slice.
        let blob_buf = ShardBuf::new(
            dev,
            &format!("w2b-blob{}", air.air_id),
            BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
            blob_len,
        );
        {
            let mut off = 0usize;
            while off < tables.len() {
                let e = (off + (1 << 22)).min(tables.len());
                blob_buf.write_slice(dev, off as u64, &tables[off..e]);
                off = e;
            }
        }
        let mut off: u64 = tables.len() as u64;
        let mut tw: u64 = 0;
        {
            use std::io::Read;
            const CU: usize = 1 << 20; // u64 trace words per chunk (8 MiB raw / 4 MiB encoded)
            let mut raw = vec![0u8; CU * 8];
            let mut enc = vec![0u8; CU * 4];
            for tf in &air.trace_files {
                let mut f = std::io::BufReader::with_capacity(
                    1 << 20,
                    std::fs::File::open(traces_dir.join(tf))
                        .unwrap_or_else(|e| panic!("trace {}: {e}", tf)),
                );
                loop {
                    let mut filled = 0usize;
                    while filled < raw.len() {
                        match f.read(&mut raw[filled..]) {
                            Ok(0) => break,
                            Ok(n) => filled += n,
                            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                            Err(e) => panic!("trace {tf}: {e}"),
                        }
                    }
                    if filled == 0 {
                        break;
                    }
                    assert_eq!(filled % 8, 0, "trace chunk {tf}");
                    for (i, c) in raw[..filled].chunks(8).enumerate() {
                        let w = u64::from_le_bytes(c.try_into().unwrap());
                        enc[i * 4..i * 4 + 4].copy_from_slice(&encode(w as u32).to_le_bytes());
                    }
                    let el = filled / 2;
                    blob_buf.write_slice(dev, off, &enc[..el]);
                    off += el as u64;
                    tw += (filled / 8) as u64;
                }
            }
            assert_eq!(tw, air.trace_words, "trace size air{}", air.air_id);
            assert_eq!(off, blob_len, "blob size air{}", air.air_id);
        }
        let _ = (&mut *cache, inputs_dir);

        // Phase-2 out layout: one ShardBuf of per-chunk REGIONS, each
        // region = [scratch n_nodes*rpc | pnum n_int*rpc | den n_int*rpc*4]
        // at pow2-padded stride (so regions never straddle a shard). The
        // dag kernel binds exactly one region window; leaf_fill re-reads
        // the same region window. Every bind <= device binding cap.
        // per-part row strides (words/row) + count, from the header:
        // part p base word at [pmeta+npart+p], stride at [pmeta+2npart+p]
        let mut part_strides: Vec<u64> = Vec::new();
        {
            let mut hd = vec![0u8; (air.tblen as usize * 4).min(1 << 16).min(tables.len())];
            // header (incl. pmeta) is the FIRST tblen words of the tables
            // file itself — read directly from the tables bytes.
            let n = hd.len().min(tables.len());
            hd[..n].copy_from_slice(&tables[..n]);
            let hw: Vec<u32> = hd.chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let npart = hw[3] as usize;
            assert!(npart <= 64, "npart {npart} > 64");
            let pmeta = hw[12] as usize;
            for p in 0..npart {
                part_strides.push(hw[pmeta + 2 * npart + p] as u64);
            }
        }
        let budget = 16u64 << 20; // 64 MiB scratch
        let wrow = air.trace_words / (air.h as u64).max(1); // trace words/row
        let mut rpc = (budget / (air.n_nodes as u64 + 1))
            .min(dev.win() / (4 * (air.n_nodes as u64 + 5 * air.n_int as u64 + 1)))
            .min(if wrow > 0 { (dev.win() / 4 - 64) / wrow } else { 1 << 30 })
            & !255;
        if rpc < 256 { rpc = 256; }
        if rpc > air.h as u64 { rpc = air.h as u64; }
        let nch = (air.h as u64 + rpc - 1) / rpc;
        eprintln!("LEAFB air{} h={} rpc={} nch={} win={}", air.air_id, air.h, rpc, nch, dev.win());
        let scratch_w = air.n_nodes as u64 * rpc;
        let region_w = (air.n_nodes as u64 + 5 * air.n_int as u64) * rpc;
        let region_bytes = region_w * 4;
        let region_pad = crate::device::pow2_floor(region_bytes - 1) * 2; // ceil pow2
        assert!(region_pad <= dev.win(), "air{} region {region_bytes} exceeds window", air.air_id);
        let out_shard = ShardBuf::new(
            dev,
            &format!("w2b-dag{}", air.air_id),
            BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
            (nch * region_pad).max(16),
        );
        // per-chunk trace window buffer: rows [c0, c0+cr] (rot +1 row)
        let win_w = 64 + (rpc + 1) * wrow.max(part_strides.iter().sum::<u64>());
        let win_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("w2b-win{}", air.air_id)),
            size: (win_w * 4).max(16),
            usage: BU::STORAGE | BU::COPY_DST,
            mapped_at_creation: false,
        });
        let pbase = scratch_w as u32;
        let dbase = (scratch_w + air.n_int as u64 * rpc) as u32;
        let tblen_w = air.tblen as u64;
        let mut c0 = 0u64;
        let mut cidx = 0u64;
        while c0 < air.h as u64 {
            let cr = rpc.min(air.h as u64 - c0) as u32;
            // per-part device copy of trace rows [c0, c0+cr] into win_buf
            // (row data starts at word 64; words [0..64) = part row bases)
            {
                let mut hdr = vec![0u32; 64];
                let np = (air.widths.len() + air.trace_files.len()) * 0
                    + air.trace_files.len(); // parts = trace files (pmeta count)
                let _ = np;
                // part p blob-trace-relative row start = cumulative h*width
                let mut src_row0 = 0u64;
                let mut win_word0 = 0u64;
                let mut phdr: Vec<u32> = Vec::new();
                for (pi, stride) in part_strides.iter().enumerate() {
                    phdr.push(win_word0 as u32);
                    let s0 = (tblen_w + src_row0 + c0 * stride) * 4;
                    let len = ((cr as u64) + 1) * stride * 4;
                    let mut d0 = 0u64;
                    while d0 < len {
                        let (si, so) = blob_buf.loc(s0 + d0);
                        let mut take = (len - d0)
                            .min(blob_buf.shards[si].size() - so)
                            .min(win_buf.size() - (64 * 4 + win_word0 * 4 + d0));
                        if take == 0 {
                            // read landed on shard end (rot +1 row past the
                            // final row of the trace) or window is full: the
                            // unread window tail is zero padding by design
                            // (rotate source absent for the final chunk).
                            break;
                        }
                        if win_word0 + d0 / 4 + take / 4 + 65 > win_buf.size() / 4 {
                            take = win_buf.size() - (64 * 4 + win_word0 * 4 + d0);
                            if take == 0 { break; }
                        }
                        let mut enc = dev.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: Some("w2b-win-copy") });
                        enc.copy_buffer_to_buffer(
                            &blob_buf.shards[si], so,
                            &win_buf, 64 * 4 + win_word0 * 4 + d0, take);
                        dev.queue.submit(Some(enc.finish()));
                        d0 += take;
                    }
                    let _ = pi;
                    src_row0 += air.h as u64 * stride;
                    win_word0 += (cr as u64 + 1) * stride;
                }
                hdr[..phdr.len()].copy_from_slice(&phdr);
                dev.queue.write_buffer(&win_buf, 0, &crate::field::u32s_to_bytes(&hdr));
            }
            lp.set_uniform(
                dev,
                [
                    c0 as u32,
                    cr,
                    0,
                    pbase,
                    dbase,
                    intstride,
                    rpc as u32,
                    0,
                ],
                [0; 4],
                0,
                (tblen_w + c0 * wrow) as u32,
            );
            let bg = lp.bg_res(
                dev,
                blob_buf.bind(dev, 0, tblen_w * 4),
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &win_buf,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new((win_w * 4).max(16)).unwrap()),
                }),
                out_shard.bind(dev, cidx * region_pad, region_bytes),
            );
            LeafPipeline::run(dev, &lp.p_dag, &bg, (cr + 255) / 256);
            c0 += rpc;
            cidx += 1;
            if cidx % ((nch / 20).max(1)) == 0 { eprintln!("LEAFB air{} dag {}/{}", air.air_id, cidx, nch); }
        }

        // slot fills for this air (windows: one leaf chunk AND one dag chunk)
        eprintln!("LEAFB air{} slots", air.air_id);
        for slot in world.slots.iter().filter(|s| s.air_id == air.air_id) {
            let len = 1u64 << slot.log_ht;
            let mut done = 0u64;
            while done < len {
                let g = slot.row_idx + done; // global leaf row
                let chunk = (g * 32 / per) as usize;
                let crow0 = chunk as u64 * rpc_leaf;
                let hr = ((g - slot.row_idx) % slot.h as u64) as u32;
                let dchunk = (hr as u64) / rpc; // dag chunk of this row
                // stay inside leaf chunk AND dag chunk
                let cap_leaf = (crow0 + rpc_leaf - g).min(len - done);
                let cap_dag = (rpc - (hr as u64 % rpc)).min(cap_leaf);
                let cnt = cap_dag.min(65535 * 256);
                // kernel recomputes (hr+i) per row incl. chunk + wrap math,
                // so bases are REGION-relative and exclude the hr offset:
                let pb0 = pbase as u64 + (slot.j as u64) * rpc;
                let db0 = dbase as u64 + (slot.j as u64) * rpc * 4;
                lp.set_uniform(
                    dev,
                    [
                        ((g - crow0) * 8) as u32,
                        cnt as u32,
                        hr,
                        slot.h,
                        slot.norm_raw,
                        pb0 as u32,
                        db0 as u32,
                        rpc as u32,
                    ],
                    alpha4,
                    0,
                    0,
                );
                let bg = lp.bg_res(
                    dev,
                    out_shard.bind(dev, dchunk * region_pad, region_bytes),
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &lp.dummy,
                        offset: 0,
                        size: Some(std::num::NonZeroU64::new(16).unwrap()),
                    }),
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &leaf[chunk],
                        offset: 0,
                        size: Some(std::num::NonZeroU64::new(per).unwrap()),
                    }),
                );
                LeafPipeline::run(dev, &lp.p_leaf_fill, &bg, ((cnt + 255) / 256) as u32);
                done += cnt;
            }
        }
    }

    // 3) canonical-form sha over the full leaf
    eprintln!("LEAFB sha");
    let mut hasher = sha2::Sha256::new();
    let dump_dir = std::env::var("W2B_LEAF_DUMP").ok();
    for (ci, cb) in leaf.iter().enumerate() {
        let mut canon: Vec<u8> = Vec::with_capacity(if dump_dir.is_some() { cb.size() as usize } else { 0 });
        {
            let mut coff = 0u64;
            let cstep: u64 = 1u64 << 22; // 4 MiB slices (was 16 MiB)
            while coff < cb.size() {
                let clen = cstep.min(cb.size() - coff);
                let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("leaf-sha-stage"),
                    size: clen,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                enc.copy_buffer_to_buffer(cb, coff, &staging, 0, clen);
                dev.queue.submit(Some(enc.finish()));
                let sl = staging.slice(..);
                sl.map_async(wgpu::MapMode::Read, |_| ());
                dev.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
                // W2d lowmem: decode+hash straight out of the mapping in 64 KiB
                // blocks instead of to_vec + full-slice canonical copy.
                if let Ok(d) = sl.get_mapped_range() {
                    let mut blk: Vec<u8> = Vec::with_capacity(1 << 16);
                    for wchunk in d.chunks(1 << 16) {
                        blk.clear();
                        for c in wchunk.chunks(4) {
                            let w = u32::from_le_bytes(c.try_into().unwrap());
                            blk.extend_from_slice(&decode(w).to_le_bytes());
                        }
                        hasher.update(&blk);
                        if dump_dir.is_some() { canon.extend_from_slice(&blk); }
                    }
                }
                drop(sl);
                staging.unmap();
                coff += clen;
            }
        }
        if let Some(d) = &dump_dir {
            std::fs::create_dir_all(d).ok();
            std::fs::write(
                Path::new(d).join(format!("leafgpu-chunk{ci}.bin")),
                &canon,
            )
            .unwrap();
        }
    }
    use sha2::Digest;
    (leaf, format!("{:x}", hasher.finalize()))
}

// ============ W2B-BROWSER async port (Phase 3d) ============
//
// The same per-air pipeline as `build_leaf`, factored into resumable pieces
// so the browser driver (wasm-pack/src/w2b.rs) can stream inputs one file at
// a time (JS file:// XHR -> per-file wasm calls) instead of one monolithic
// call. Op order, windows and kernels are byte-identical to the sync path.
// The ONLY awaited part is the leaf sha readback (device.poll is a NO-OP on
// the web backend — §7c/§8; download_off_async's wasm branch oneshot-awaits
// the mapAsync callback).

/// Per-air streaming context (blob upload -> windowed DAG -> slot fills).
pub struct AirCtx {
    pub air_idx: usize,
    blob_buf: ShardBuf,
    tables: Vec<u8>,
    out_shard: Option<ShardBuf>,
    win_buf: Option<wgpu::Buffer>,
    off: u64,
    tw: u64,
    intstride: u32,
    part_strides: Vec<u64>,
    rpc: u64,
    nch: u64,
    region_pad: u64,
    pbase: u32,
    dbase: u32,
    tblen_w: u64,
}

pub struct LeafBuildState {
    pub leaf: Vec<wgpu::Buffer>,
    rpc_leaf: u64,
    alpha4: [u32; 4],
    ctx: Option<AirCtx>,
}

/// 1) allocate the leaf chunks + default fill (p=0, q=alpha). No readback.
pub fn leaf_begin(
    dev: &WgpuDevice,
    lp: &LeafPipeline,
    world: &LeafWorld,
    chunk_bytes: u64,
) -> LeafBuildState {
    let n = 1u64 << world.log_stack;
    let alpha: Vec<u32> = world.alpha_canon.iter().map(|v| encode(*v as u32)).collect();
    let alpha4 = [alpha[0], alpha[1], alpha[2], alpha[3]];
    let per = chunk_bytes;
    assert_eq!(n * 32 % per, 0);
    let nchunks = (n * 32 / per) as usize;
    let leaf: Vec<wgpu::Buffer> = (0..nchunks)
        .map(|c| {
            dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("w2b-leaf{c}")),
                size: per,
                usage: BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();
    let rpc_leaf = per / 32;
    for cb in &leaf {
        let mut done = 0u64;
        while done < rpc_leaf {
            let cnt = (rpc_leaf - done).min(65535 * 256);
            lp.set_uniform(dev, [(done * 8) as u32, cnt as u32, 0, 0, 0, 0, 0, 0], alpha4, 0, 0);
            let bg = lp.bg(dev, &lp.dummy, cb);
            LeafPipeline::run(dev, &lp.p_fill_default, &bg, ((cnt + 255) / 256) as u32);
            done += cnt;
        }
    }
    LeafBuildState { leaf, rpc_leaf, alpha4, ctx: None }
}

/// 2a) open air `air_idx`: allocate its blob ShardBuf (tables + encoded
/// traces). Exactly one air may be open at a time.
pub fn air_start(st: &mut LeafBuildState, dev: &WgpuDevice, world: &LeafWorld, air_idx: usize) {
    assert!(st.ctx.is_none(), "air_start with air already open");
    let air = &world.airs[air_idx];
    let _ = air; // blob size is known only after tables arrive (feed_tables)
    st.ctx = Some(AirCtx {
        air_idx,
        blob_buf: ShardBuf::new(dev, &format!("w2b-blobt{}", air.air_id), BU::STORAGE | BU::COPY_DST | BU::COPY_SRC, 16),
        tables: Vec::new(),
        out_shard: None,
        win_buf: None,
        off: 0,
        tw: 0,
        intstride: 0,
        part_strides: Vec::new(),
        rpc: 0,
        nch: 0,
        region_pad: 0,
        pbase: 0,
        dbase: 0,
        tblen_w: air.tblen as u64,
    });
    // pre-size: tables bytes unknown until feed_tables; recreate blob there.
}

/// 2b) feed the air's tables file (verbatim bytes). Uploads + parses the
/// part-stride header + allocates the out/region shards and trace window.
pub fn air_feed_tables(
    st: &mut LeafBuildState,
    dev: &WgpuDevice,
    world: &LeafWorld,
    tables: Vec<u8>,
) {
    let ctx = st.ctx.as_mut().expect("air_feed_tables without air_start");
    let air = &world.airs[ctx.air_idx];
    assert_eq!(tables.len() / 4, air.tblen as usize);
    let blob_len = tables.len() as u64 + air.trace_words * 4;
    ctx.blob_buf = ShardBuf::new(
        dev,
        &format!("w2b-blob{}", air.air_id),
        BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
        blob_len,
    );
    let mut off = 0usize;
    while off < tables.len() {
        let e = (off + (1 << 22)).min(tables.len());
        ctx.blob_buf.write_slice(dev, off as u64, &tables[off..e]);
        off = e;
    }
    ctx.off = tables.len() as u64;
    ctx.intstride = u32::from_le_bytes(tables[40..44].try_into().unwrap()) + 4;
    let mut part_strides: Vec<u64> = Vec::new();
    {
        let mut hd = vec![0u8; (air.tblen as usize * 4).min(1 << 16).min(tables.len())];
        let m = hd.len().min(tables.len());
        hd[..m].copy_from_slice(&tables[..m]);
        let hw: Vec<u32> = hd
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let npart = hw[3] as usize;
        assert!(npart <= 64, "npart {npart} > 64");
        let pmeta = hw[12] as usize;
        for p in 0..npart {
            part_strides.push(hw[pmeta + 2 * npart + p] as u64);
        }
    }
    let budget = 16u64 << 20; // 64 MiB scratch
    let wrow = air.trace_words / (air.h as u64).max(1);
    let mut rpc = (budget / (air.n_nodes as u64 + 1))
        .min(dev.win() / (4 * (air.n_nodes as u64 + 5 * air.n_int as u64 + 1)))
        .min(if wrow > 0 { (dev.win() / 4 - 64) / wrow } else { 1 << 30 })
        & !255;
    if rpc < 256 {
        rpc = 256;
    }
    if rpc > air.h as u64 {
        rpc = air.h as u64;
    }
    let nch = (air.h as u64 + rpc - 1) / rpc;
    eprintln!("LEAFB air{} h={} rpc={} nch={} win={}", air.air_id, air.h, rpc, nch, dev.win());
    let scratch_w = air.n_nodes as u64 * rpc;
    let region_w = (air.n_nodes as u64 + 5 * air.n_int as u64) * rpc;
    let region_bytes = region_w * 4;
    let region_pad = crate::device::pow2_floor(region_bytes - 1) * 2;
    assert!(region_pad <= dev.win(), "air{} region {region_bytes} exceeds window", air.air_id);
    let wrow_ = wrow;
    let out_shard = ShardBuf::new(
        dev,
        &format!("w2b-dag{}", air.air_id),
        BU::STORAGE | BU::COPY_DST | BU::COPY_SRC,
        (nch * region_pad).max(16),
    );
    let win_w = 64 + (rpc + 1) * wrow_.max(part_strides.iter().sum::<u64>());
    let win_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("w2b-win{}", air.air_id)),
        size: (win_w * 4).max(16),
        usage: BU::STORAGE | BU::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.part_strides = part_strides;
    ctx.rpc = rpc;
    ctx.nch = nch;
    ctx.region_pad = region_pad;
    ctx.pbase = scratch_w as u32;
    ctx.dbase = (scratch_w + air.n_int as u64 * rpc) as u32;
    ctx.win_buf = Some(win_buf);
    ctx.out_shard = Some(out_shard);
    ctx.tables = tables;
}

/// 2c) feed one trace file of the open air (raw u64-LE bytes). Mont-encodes
/// in 8 MiB slices and appends to the blob (same byte stream as sync path).
pub fn air_feed_trace(st: &mut LeafBuildState, dev: &WgpuDevice, world: &LeafWorld, raw: Vec<u8>) {
    let ctx = st.ctx.as_mut().expect("air_feed_trace without air_start");
    assert!(!ctx.tables.is_empty(), "trace before tables");
    let air = &world.airs[ctx.air_idx];
    const CU: usize = 1 << 20; // u64 words per chunk (8 MiB raw / 4 MiB encoded)
    let mut enc = vec![0u8; CU * 4];
    let mut ro = 0usize;
    while ro < raw.len() {
        let e = (ro + CU * 8).min(raw.len());
        let sl = &raw[ro..e];
        assert_eq!(sl.len() % 8, 0, "trace chunk air{}", air.air_id);
        for (i, c) in sl.chunks(8).enumerate() {
            let w = u64::from_le_bytes(c.try_into().unwrap());
            enc[i * 4..i * 4 + 4].copy_from_slice(&encode(w as u32).to_le_bytes());
        }
        let el = sl.len() / 2;
        ctx.blob_buf.write_slice(dev, ctx.off, &enc[..el]);
        ctx.off += el as u64;
        ctx.tw += (sl.len() / 8) as u64;
        ro = e;
    }
    let _ = air;
}

/// 2d) run the open air's windowed DAG + slot fills, then close the air
/// (drop blob/out/window).
pub fn air_run(st: &mut LeafBuildState, dev: &WgpuDevice, lp: &LeafPipeline, world: &LeafWorld) {
    let ctx = st.ctx.take().expect("air_run without air_start");
    let air = &world.airs[ctx.air_idx];
    assert_eq!(ctx.tw, air.trace_words, "trace size air{}", air.air_id);
    let blob_len = ctx.tables.len() as u64 + air.trace_words * 4;
    assert_eq!(ctx.off, blob_len, "blob size air{}", air.air_id);
    let blob_buf = ctx.blob_buf;
    let out_shard = ctx.out_shard.as_ref().unwrap();
    let win_buf = ctx.win_buf.as_ref().unwrap();
    let rpc = ctx.rpc;
    let region_pad = ctx.region_pad;
    let tblen_w = ctx.tblen_w;
    let wrow = air.trace_words / (air.h as u64).max(1);
    let region_bytes = region_w_bytes(air, rpc);
    let win_bytes =
        (64 + (rpc + 1) * wrow.max(ctx.part_strides.iter().sum::<u64>())) * 4;
    eprintln!("LEAFB air{} dag", air.air_id);
    let mut c0 = 0u64;
    let mut cidx = 0u64;
    while c0 < air.h as u64 {
        let cr = rpc.min(air.h as u64 - c0) as u32;
        {
            let mut hdr = vec![0u32; 64];
            let mut src_row0 = 0u64;
            let mut win_word0 = 0u64;
            let mut phdr: Vec<u32> = Vec::new();
            for stride in ctx.part_strides.iter() {
                phdr.push(win_word0 as u32);
                let s0 = (tblen_w + src_row0 + c0 * stride) * 4;
                let len = ((cr as u64) + 1) * stride * 4;
                let mut d0 = 0u64;
                while d0 < len {
                    let (si, so) = blob_buf.loc(s0 + d0);
                    let mut take = (len - d0)
                        .min(blob_buf.shards[si].size() - so)
                        .min(win_buf.size() - (64 * 4 + win_word0 * 4 + d0));
                    if take == 0 {
                        break;
                    }
                    if win_word0 + d0 / 4 + take / 4 + 65 > win_buf.size() / 4 {
                        take = win_buf.size() - (64 * 4 + win_word0 * 4 + d0);
                        if take == 0 {
                            break;
                        }
                    }
                    let mut enc = dev.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor { label: Some("w2b-win-copy") });
                    enc.copy_buffer_to_buffer(
                        &blob_buf.shards[si], so,
                        &win_buf, 64 * 4 + win_word0 * 4 + d0, take);
                    dev.queue.submit(Some(enc.finish()));
                    d0 += take;
                }
                src_row0 += air.h as u64 * stride;
                win_word0 += (cr as u64 + 1) * stride;
            }
            hdr[..phdr.len()].copy_from_slice(&phdr);
            dev.queue.write_buffer(win_buf, 0, &crate::field::u32s_to_bytes(&hdr));
        }
        lp.set_uniform(
            dev,
            [c0 as u32, cr, 0, ctx.pbase, ctx.dbase, ctx.intstride, rpc as u32, 0],
            [0; 4],
            0,
            (tblen_w + c0 * wrow) as u32,
        );
        let bg = lp.bg_res(
            dev,
            blob_buf.bind(dev, 0, tblen_w * 4),
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: win_buf,
                offset: 0,
                size: Some(std::num::NonZeroU64::new(win_bytes.max(16)).unwrap()),
            }),
            out_shard.bind(dev, cidx * region_pad, region_bytes),
        );
        LeafPipeline::run(dev, &lp.p_dag, &bg, (cr + 255) / 256);
        c0 += rpc;
        cidx += 1;
        if cidx % ((ctx.nch / 20).max(1)) == 0 {
            eprintln!("LEAFB air{} dag {}/{}", air.air_id, cidx, ctx.nch);
        }
    }

    eprintln!("LEAFB air{} slots", air.air_id);
    let alpha4 = st.alpha4;
    let rpc_leaf = st.rpc_leaf;
    let per = rpc_leaf * 32;
    let leaf = &st.leaf;
    for slot in world.slots.iter().filter(|s| s.air_id == air.air_id) {
        let len = 1u64 << slot.log_ht;
        let mut done = 0u64;
        while done < len {
            let g = slot.row_idx + done;
            let chunk = (g * 32 / per) as usize;
            let crow0 = chunk as u64 * rpc_leaf;
            let hr = ((g - slot.row_idx) % slot.h as u64) as u32;
            let dchunk = (hr as u64) / rpc;
            let cap_leaf = (crow0 + rpc_leaf - g).min(len - done);
            let cap_dag = (rpc - (hr as u64 % rpc)).min(cap_leaf);
            let cnt = cap_dag.min(65535 * 256);
            let pb0 = ctx.pbase as u64 + (slot.j as u64) * rpc;
            let db0 = ctx.dbase as u64 + (slot.j as u64) * rpc * 4;
            lp.set_uniform(
                dev,
                [
                    ((g - crow0) * 8) as u32,
                    cnt as u32,
                    hr,
                    slot.h,
                    slot.norm_raw,
                    pb0 as u32,
                    db0 as u32,
                    rpc as u32,
                ],
                alpha4,
                0,
                0,
            );
            let bg = lp.bg_res(
                dev,
                out_shard.bind(dev, dchunk * region_pad, region_w_bytes(air, rpc)),
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &lp.dummy,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(16).unwrap()),
                }),
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &leaf[chunk],
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(per).unwrap()),
                }),
            );
            LeafPipeline::run(dev, &lp.p_leaf_fill, &bg, ((cnt + 255) / 256) as u32);
            done += cnt;
        }
    }
    // ctx (blob/out_shard/win_buf/tables) dropped here — per-air GPU memory
    // is freed before the next air (same memory law as the sync path).
}

fn region_w_bytes(air: &AirSpec, rpc: u64) -> u64 {
    (air.n_nodes as u64 + 5 * air.n_int as u64) * rpc * 4
}

/// 3) canonical-form sha256 of the leaf, AWAITED readback (§7c).
pub async fn leaf_sha_async(dev: &WgpuDevice, st: &LeafBuildState) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    let mut blk: Vec<u8> = Vec::with_capacity(1 << 22);
    for (ci, cb) in st.leaf.iter().enumerate() {
        let mut coff = 0u64;
        let cstep: u64 = 1u64 << 22;
        while coff < cb.size() {
            let clen = cstep.min(cb.size() - coff);
            let d = crate::gkr::download_off_async(dev, cb, coff, clen).await;
            blk.clear();
            for c in d.chunks(4) {
                let w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                blk.extend_from_slice(&decode(w).to_le_bytes());
            }
            hasher.update(&blk);
            coff += clen;
        }
        eprintln!("LEAFB sha chunk {ci}");
    }
    format!("{:x}", hasher.finalize())
}

/// Convenience one-shot build (feeds via a `LeafSource`): composition of the
/// resumable pieces above — single source of truth for the pipeline.
pub trait LeafSource {
    fn tables(&mut self, air: &AirSpec) -> impl std::future::Future<Output = Result<Vec<u8>, String>>;
    fn trace(&mut self, air: &AirSpec, idx: usize) -> impl std::future::Future<Output = Result<Vec<u8>, String>>;
}

pub struct FsLeafSource {
    pub inputs: std::path::PathBuf,
    pub traces: std::path::PathBuf,
}

impl LeafSource for FsLeafSource {
    async fn tables(&mut self, air: &AirSpec) -> Result<Vec<u8>, String> {
        std::fs::read(self.inputs.join(&air.tables)).map_err(|e| format!("tables {}: {e}", air.tables))
    }
    async fn trace(&mut self, air: &AirSpec, idx: usize) -> Result<Vec<u8>, String> {
        let tf = &air.trace_files[idx];
        std::fs::read(self.traces.join(tf)).map_err(|e| format!("trace {tf}: {e}"))
    }
}

pub async fn build_leaf_async<S: LeafSource>(
    dev: &WgpuDevice,
    lp: &LeafPipeline,
    world: &LeafWorld,
    chunk_bytes: u64,
    src: &mut S,
) -> (Vec<wgpu::Buffer>, String) {
    let mut st = leaf_begin(dev, lp, world, chunk_bytes);
    eprintln!("LEAFB dagslots start");
    for i in 0..world.airs.len() {
        let air = &world.airs[i];
        let tables = src.tables(air).await.unwrap_or_else(|e| panic!("w2b-async: {e}"));
        let ntr = air.trace_files.len();
        air_start(&mut st, dev, world, i);
        air_feed_tables(&mut st, dev, world, tables);
        for ti in 0..ntr {
            let raw = src.trace(air, ti).await.unwrap_or_else(|e| panic!("w2b-async: {e}"));
            air_feed_trace(&mut st, dev, world, raw);
        }
        air_run(&mut st, dev, lp, world);
    }
    let sha = leaf_sha_async(dev, &st).await;
    (st.leaf, sha)
}
