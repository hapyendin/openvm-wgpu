//! GATE W2C PHASE 2: WGSL WHIR fold — mobius/DFT butterfly stages + data-row
//! merkle (init4 / strided-pair / adjacent-pair layers), Poseidon2 kernels
//! copied VERBATIM from the golden wgsl.rs perm path.
use crate::device::WgpuDevice;
use crate::field::{constants_buffer_mont, decode, u32s_to_bytes};
use crate::limitsbuf::ShardBuf;

const WGSL: &str = r#"
// `mul` supplied by wgsl::mul_prelude (u64-REDC or pure-u32 limb path)

@group(0) @binding(0) var<storage, read> kc: array<u32>;
@group(0) @binding(1) var<storage, read_write> io1: array<u32>;
@group(0) @binding(2) var<storage, read_write> io2: array<u32>;
struct Uni { a: u32, b: u32, c: u32, d: u32, e: u32 };
@group(0) @binding(3) var<uniform> uni: Uni;
@group(0) @binding(4) var<storage, read> tw: array<u32>;
@group(0) @binding(5) var<storage, read_write> io3: array<u32>;

const PP: u32 = 2013265921u;
const HALF_P1_C: u32 = 1006632961u;
const IC_OFF: u32 = 0u;
const TC_OFF: u32 = 64u;
const IRC_OFF: u32 = 128u;
const DG_OFF: u32 = 141u;

fn addm(a: u32, b: u32) -> u32 {
  var s = a + b;
  if (s >= PP) { s = s - PP; }
  return s;
}
fn subm(a: u32, b: u32) -> u32 {
  var s: u32;
  if (a >= b) { s = a - b; } else { s = a + (PP - b); }
  return s;
}
fn halvef(x: u32) -> u32 { return (x >> 1u) + ((x & 1u) * HALF_P1_C); }
fn dbLf(x: u32) -> u32 { return addm(x, x); }
fn d7f(x: u32) -> u32 {
  let x2 = mul(x, x);
  let x4 = mul(x2, x2);
  let x6 = mul(x4, x2);
  return mul(x6, x);
}
fn m_ext(cells: ptr<function, array<u32, 16u>>) {
  var ts: array<u32, 4u>;
  for (var j: u32 = 0u; j < 4u; j = j + 1u) { ts[j] = 0u; }
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    let b = i * 4u;
    let x0 = (*cells)[b]; let x1 = (*cells)[b + 1u];
    let x2 = (*cells)[b + 2u]; let x3 = (*cells)[b + 3u];
    let t01 = addm(x0, x1);
    let t23 = addm(x2, x3);
    let t0123 = addm(t01, t23);
    let t01123 = addm(t0123, x1);
    let t01233 = addm(t0123, x3);
    (*cells)[b + 3u] = addm(t01233, dbLf(x0));
    (*cells)[b + 1u] = addm(t01123, dbLf(x2));
    (*cells)[b] = addm(t01123, t01);
    (*cells)[b + 2u] = addm(t01233, t23);
    for (var j2: u32 = 0u; j2 < 4u; j2 = j2 + 1u) {
      ts[j2] = addm(ts[j2], (*cells)[b + j2]);
    }
  }
  for (var i2: u32 = 0u; i2 < 16u; i2 = i2 + 1u) {
    (*cells)[i2] = addm((*cells)[i2], ts[i2 % 4u]);
  }
}
fn internal_layer(cells: ptr<function, array<u32, 16u>>) {
  var part_sum = (*cells)[1u];
  for (var i: u32 = 2u; i < 16u; i = i + 1u) {
    part_sum = addm(part_sum, (*cells)[i]);
  }
  let sum = addm(part_sum, (*cells)[0u]);
  (*cells)[0u] = subm(part_sum, (*cells)[0u]);
  (*cells)[1u] = addm(sum, (*cells)[1u]);
  (*cells)[2u] = addm(sum, dbLf((*cells)[2u]));
  (*cells)[3u] = addm(sum, halvef((*cells)[3u]));
  (*cells)[4u] = addm(sum, addm(dbLf((*cells)[4u]), (*cells)[4u]));
  (*cells)[5u] = addm(sum, dbLf(dbLf((*cells)[5u])));
  (*cells)[6u] = subm(sum, halvef((*cells)[6u]));
  (*cells)[7u] = subm(sum, addm(dbLf((*cells)[7u]), (*cells)[7u]));
  (*cells)[8u] = subm(sum, dbLf(dbLf((*cells)[8u])));
  (*cells)[9u] = addm(sum, mul((*cells)[9u], kc[DG_OFF + 9u]));
  (*cells)[10u] = addm(sum, halvef(halvef((*cells)[10u])));
  (*cells)[11u] = addm(sum, mul((*cells)[11u], kc[DG_OFF + 11u]));
  (*cells)[12u] = addm(sum, mul((*cells)[12u], kc[DG_OFF + 12u]));
  (*cells)[13u] = addm(sum, mul((*cells)[13u], kc[DG_OFF + 13u]));
  (*cells)[14u] = addm(sum, mul((*cells)[14u], kc[DG_OFF + 14u]));
  (*cells)[15u] = addm(sum, mul((*cells)[15u], kc[DG_OFF + 15u]));
}
fn full_round(cells: ptr<function, array<u32, 16u>>, koff: u32, r: u32) {
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    (*cells)[i] = addm((*cells)[i], kc[koff + r * 16u + i]);
  }
  for (var i2: u32 = 0u; i2 < 16u; i2 = i2 + 1u) {
    (*cells)[i2] = d7f((*cells)[i2]);
  }
  m_ext(cells);
}
fn partial_round(cells: ptr<function, array<u32, 16u>>, r: u32) {
  (*cells)[0u] = addm((*cells)[0u], kc[IRC_OFF + r]);
  (*cells)[0u] = d7f((*cells)[0u]);
  internal_layer(cells);
}
fn perm_mix(cells: ptr<function, array<u32, 16u>>) {
  m_ext(cells);
  for (var r: u32 = 0u; r < 4u; r = r + 1u) { full_round(cells, IC_OFF, r); }
  for (var r2: u32 = 0u; r2 < 13u; r2 = r2 + 1u) { partial_round(cells, r2); }
  for (var r3: u32 = 0u; r3 < 4u; r3 = r3 + 1u) { full_round(cells, TC_OFF, r3); }
}

// one butterfly pair (EF row-major layout: rows of 4 limbs)
// uni.a = n/2 butterflies, uni.b = half (stage), uni.d = butterfly base
// uni.c = mode: 0=fft*tw local, 1=mobius local, 2=fft*tw FAR (A=io1, B=io2
// windows of one block pair), 3=mobius FAR. uni.e = window base ROW for
// the local modes (sliced binds of >128MiB data buffers); pairs are
// block-aligned and windows are pow2-aligned multiples of 2*half, so
// j - e*4 is exact window-local addressing with identical arithmetic.
@compute @workgroup_size(256u)
fn whir_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = uni.b;
  if (uni.c < 2u) {
    let b = gid.x + uni.d;
    if (b >= uni.a) { return; }
    let block = b / half;
    let k = b % half;
    let j = (block * half * 2u + k - uni.e) * 4u;
    let jp = j + half * 4u;
    if (uni.c == 0u) {
      let w = tw[k];
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let u = io1[j + l];
        let t = mul(io1[jp + l], w);
        io1[j + l] = addm(u, t);
        io1[jp + l] = subm(u, t);
      }
    } else {
      for (var l2: u32 = 0u; l2 < 4u; l2 = l2 + 1u) {
        let lo = io1[j + l2];
        let hi = io1[jp + l2];
        io1[jp + l2] = subm(hi, lo);
      }
    }
  } else {
    // FAR: one block pair per dispatch; uni.e = block-start row
    let k = gid.x + uni.d;
    if (k >= half) { return; }
    let j = k * 4u;
    if (uni.c == 2u) {
      let w = tw[k];
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let u = io1[j + l];
        let t = mul(io2[j + l], w);
        io1[j + l] = addm(u, t);
        io2[j + l] = subm(u, t);
      }
    } else {
      for (var l2: u32 = 0u; l2 < 4u; l2 = l2 + 1u) {
        io2[j + l2] = subm(io2[j + l2], io1[j + l2]);
      }
    }
  }
}

// merkle leaf init over data rows: state = [row8 | 0^8] (raw mont words),
// mirrors CPU merkle_leaf on 4-word rows zero-padded to 8.
@compute @workgroup_size(256u)
fn whir_init4(@builtin(global_invocation_id) gid: vec3<u32>) {
  // window-local: io1/io2 are bound at the same row base; uni.d unused.
  let i = gid.x;
  if (i * 8u >= arrayLength(&io2)) { return; }
  var cells: array<u32, 16u>;
  for (var z: u32 = 0u; z < 16u; z = z + 1u) { cells[z] = 0u; }
  for (var j: u32 = 0u; j < 8u; j = j + 1u) { cells[j] = io1[i * 8u + j]; }
  perm_mix(&cells);
  for (var j2: u32 = 0u; j2 < 8u; j2 = j2 + 1u) { io2[i * 8u + j2] = cells[j2]; }
}

// phase-1 strided pairing (windowed): uni.a = nsz (global bound), uni.d =
// window output-row base i0 (a multiple of qs with window <= qs rows, so
// x = i0/qs is CONSTANT over the window). A window (2x*qs rows) is io1,
// B window ((2x+1)*qs rows) is io2, output (row i0..) is io3; all binds
// are window-aligned so gid.x is the window-local row offset.
@compute @workgroup_size(256u)
fn whir_pair_strided(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + uni.d;
  if (i >= uni.a) { return; }
  let o = gid.x * 8u;
  var cells: array<u32, 16u>;
  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    cells[j] = io1[o + j];
    cells[j + 8u] = io2[o + j];
  }
  perm_mix(&cells);
  for (var j2: u32 = 0u; j2 < 8u; j2 = j2 + 1u) { io3[o + j2] = cells[j2]; }
}

// phase-2 adjacent pairing (windowed): uni.a = nsz, uni.d = window base
// row i0. io1 binds rows [2*i0, 2*i0 + 2*w) (64 B/output row), io2 binds
// out rows [i0, i0 + w) (32 B/row); gid.x is window-local.
@compute @workgroup_size(256u)
fn whir_pair_adj(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + uni.d;
  if (i >= uni.a) { return; }
  let o = gid.x * 8u;
  var cells: array<u32, 16u>;
  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    cells[j] = io1[o * 2u + j];
    cells[j + 8u] = io1[o * 2u + 8u + j];
  }
  perm_mix(&cells);
  for (var j2: u32 = 0u; j2 < 8u; j2 = j2 + 1u) { io2[o + j2] = cells[j2]; }
}
"#;

pub struct WhirFold {
    pub module: wgpu::ShaderModule,
    pub bgl: wgpu::BindGroupLayout,
    pub stage: wgpu::ComputePipeline,
    pub init4: wgpu::ComputePipeline,
    pub pair_strided: wgpu::ComputePipeline,
    pub pair_adj: wgpu::ComputePipeline,
    pub kc_buf: wgpu::Buffer,
}

impl WhirFold {
    pub fn new(dev: &WgpuDevice) -> WhirFold {
        let module = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("whir_fold"),
            source: wgpu::ShaderSource::Wgsl(
                crate::wgsl::with_prelude(dev.has_shader_int64, WGSL).into(),
            ),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(20),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
        let layout = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipe = |name: &str| {
            dev.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let kc = constants_buffer_mont();
        let kc_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("whir kc"),
            size: (kc.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&kc_buf, 0, &u32s_to_bytes(&kc));
        WhirFold {
            bgl,
            stage: pipe("whir_stage"),
            init4: pipe("whir_init4"),
            pair_strided: pipe("whir_pair_strided"),
            pair_adj: pipe("whir_pair_adj"),
            kc_buf,
            module,
        }
    }

    pub fn buf(&self, dev: &WgpuDevice, words: &[u32], label: &str) -> wgpu::Buffer {
        assert!(!words.is_empty());
        let n = (words.len() * 4).max(16) as u64;
        let b = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: n,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&b, 0, &u32s_to_bytes(words));
        b
    }
    pub fn zeros(&self, dev: &WgpuDevice, nwords: usize, label: &str) -> wgpu::Buffer {
        assert!(nwords > 0);
        dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (nwords * 4).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }
    pub fn empty16(&self, dev: &WgpuDevice) -> wgpu::Buffer {
        self.zeros(dev, 4, "dummy16")
    }
    pub fn dummy_ro(&self, dev: &WgpuDevice) -> wgpu::Buffer {
        dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy_ro"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
    pub fn uni16(&self, dev: &WgpuDevice) -> wgpu::Buffer {
        dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uni16"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// whole-buffer bind (small buffers; io3 defaults to a 16 B dummy).
    pub fn bind(
        &self,
        dev: &WgpuDevice,
        io1: &wgpu::Buffer,
        io2: &wgpu::Buffer,
        uni: &wgpu::Buffer,
        tw: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let d3 = self.empty16(dev);
        dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.kc_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: io1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: io2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: d3.as_entire_binding() },
            ],
        })
    }

    /// sliced/windowed bind with caller-provided BindingResources.
    pub fn bind_res<'a>(
        &self,
        dev: &WgpuDevice,
        io1: wgpu::BindingResource<'a>,
        io2: wgpu::BindingResource<'a>,
        io3: wgpu::BindingResource<'a>,
        uni: &wgpu::Buffer,
        tw: wgpu::BindingResource<'a>,
    ) -> wgpu::BindGroup {
        dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.kc_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: io1 },
                wgpu::BindGroupEntry { binding: 2, resource: io2 },
                wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tw },
                wgpu::BindGroupEntry { binding: 5, resource: io3 },
            ],
        })
    }

    pub fn set_uni(dev: &WgpuDevice, uni: &wgpu::Buffer, a: u32, b: u32, c: u32, d: u32) {
        Self::set_uni5(dev, uni, a, b, c, d, 0);
    }

    pub fn set_uni5(dev: &WgpuDevice, uni: &wgpu::Buffer, a: u32, b: u32, c: u32, d: u32, e: u32) {
        let mut x = [0u8; 32];
        x[0..4].copy_from_slice(&a.to_le_bytes());
        x[4..8].copy_from_slice(&b.to_le_bytes());
        x[8..12].copy_from_slice(&c.to_le_bytes());
        x[12..16].copy_from_slice(&d.to_le_bytes());
        x[16..20].copy_from_slice(&e.to_le_bytes());
        dev.queue.write_buffer(uni, 0, &x);
    }

    pub fn dispatch(dev: &WgpuDevice, p: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, threads: usize) {
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(p);
            pass.set_bind_group(0, bg, &[]);
            let mut left = threads.div_ceil(256);
            let mut off = 0u32;
            while left > 0 {
                let g = left.min(65535);
                pass.dispatch_workgroups(g as u32, 1, 1);
                // subsequent dispatches continue via base_instance-free re-bind:
                // kernels use global_invocation_id.x only per dispatch chunk, so we
                // split with a stride uniform patch instead — handled by caller when
                // threads > 65535*256 via dispatch_chunked below. For now assert.
                let _ = off;
                left -= g;
            }
        }
        dev.queue.submit(Some(enc.finish()));
    }

    /// chunked dispatch for kernels that need > 65535 groups: per chunk we
    /// rewrite uni.d = base thread offset and submit separately.
    pub fn dispatch_offset(
        dev: &WgpuDevice,
        p: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        uni: &wgpu::Buffer,
        a: u32,
        b: u32,
        c: u32,
        n_threads: usize,
    ) {
        const CHUNK: usize = 65535 * 256;
        let mut done = 0usize;
        while done < n_threads {
            let take = (n_threads - done).min(CHUNK);
            Self::set_uni(dev, uni, a, b, c, done as u32);
            let mut enc = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(p);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(take.div_ceil(256) as u32, 1, 1);
            }
            dev.queue.submit(Some(enc.finish()));
            done += take;
        }
    }

    pub fn read_back(dev: &WgpuDevice, b: &wgpu::Buffer, nwords: usize) -> Vec<u32> {
        let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb"),
            size: (nwords * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(b, 0, &staging, 0, (nwords * 4) as u64);
        dev.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let mut out = vec![0u32; nwords];
        let d = slice.get_mapped_range().unwrap();
        for (i, w) in out.iter_mut().enumerate() {
            *w = u32::from_le_bytes(d[i * 4..i * 4 + 4].try_into().unwrap());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Phase-2 windowed drivers: all buffers are ShardBufs, every bind is a
// pow2-aligned window <= device binding cap, so the WHIR stage/merkle path
// runs unchanged under WebGPU spec-floor limits. Butterfly arithmetic is
// untouched (window-local addressing), hence bit-identical results.
// ---------------------------------------------------------------------------

pub const SHARD_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

/// n EF rows x 16 B stage-data buffer.
pub fn data_shard(dev: &WgpuDevice, n_rows: usize, label: &str) -> ShardBuf {
    ShardBuf::new(dev, label, SHARD_USAGES, (n_rows * 16) as u64)
}

/// Run all mobius (mode 1) or DIT-FFT (mode 0, tws[stage] per stage)
/// butterfly stages in place on `data` (n rows, pow2).
pub fn whir_stages(
    dev: &WgpuDevice,
    wf: &WhirFold,
    data: &ShardBuf,
    n: usize,
    mode: u32,
    tws: &[wgpu::Buffer],
) {
    let uni = wf.uni16(dev);
    let dummy16 = wf.empty16(dev);
    let dummy16b = wf.empty16(dev); // WEBGPU: io2/io3 writable-alias rejected
    let ddummy = wf.dummy_ro(dev);
    let win_rows = (dev.win() / 16) as usize; // 16 B/row
    let mut half = 1usize;
    let mut s = 0usize;
    while half < n {
        let twb = if mode == 0 { &tws[s] } else { &ddummy };
        if 2 * half <= win_rows {
            // LOCAL: butterflies stay inside one window; blocks are
            // 2*half-aligned and windows pow2-aligned >= 2*half.
            let wrows = win_rows.min(n);
            let mut e0 = 0usize;
            while e0 < n {
                let w = wrows.min(n - e0);
                let blocks = w / (2 * half);
                let b0 = (e0 / (2 * half)) * half;
                WhirFold::set_uni5(
                    dev,
                    &uni,
                    (n / 2) as u32,
                    half as u32,
                    mode,
                    b0 as u32,
                    e0 as u32,
                );
                let bg = wf.bind_res(
                    dev,
                    data.bind(dev, (e0 * 16) as u64, (w * 16) as u64),
                    dummy16.as_entire_binding(),
                    dummy16b.as_entire_binding(),
                    &uni,
                    twb.as_entire_binding(),
                );
                WhirFold::dispatch(dev, &wf.stage, &bg, blocks * half);
                e0 += w;
            }
        } else {
            // FAR: partner block is outside any sub-128MiB window ->
            // bind the two block halves of each pair separately.
            let blocks = n / (2 * half);
            for bl in 0..blocks {
                let e0 = bl * 2 * half;
                WhirFold::set_uni5(
                    dev,
                    &uni,
                    (n / 2) as u32,
                    half as u32,
                    mode + 2,
                    0,
                    e0 as u32,
                );
                let bg = wf.bind_res(
                    dev,
                    data.bind(dev, (e0 * 16) as u64, (half * 16) as u64),
                    data.bind(dev, ((e0 + half) * 16) as u64, (half * 16) as u64),
                    dummy16.as_entire_binding(),
                    &uni,
                    twb.as_entire_binding(),
                );
                WhirFold::dispatch(dev, &wf.stage, &bg, half);
            }
        }
        half *= 2;
        s += 1;
    }
}

/// merkle leaf init: rows (n x 8 words) -> cur (n x 8 words), windowed.
pub fn merkle_init(dev: &WgpuDevice, wf: &WhirFold, rows: &ShardBuf, cur: &ShardBuf, n: usize) {
    let uni = wf.uni16(dev);
    let dummy = wf.dummy_ro(dev);
    let dummy16 = wf.empty16(dev);
    let wrows = (dev.win() / 32).min(n as u64) as usize;
    let mut r0 = 0usize;
    while r0 < n {
        let w = wrows.min(n - r0);
        WhirFold::set_uni(dev, &uni, (n * 8) as u32, 0, 0, r0 as u32);
        let bg = wf.bind_res(
            dev,
            rows.bind(dev, (r0 * 32) as u64, (w * 32) as u64),
            cur.bind(dev, (r0 * 32) as u64, (w * 32) as u64),
            dummy16.as_entire_binding(),
            &uni,
            dummy.as_entire_binding(),
        );
        WhirFold::dispatch(dev, &wf.init4, &bg, w);
        r0 += w;
    }
}

/// full merkle tree over initialized cur (n leaves); returns decoded root.
pub fn merkle_reduce(
    dev: &WgpuDevice,
    wf: &WhirFold,
    cur: &ShardBuf,
    n: usize,
    k_whir: usize,
) -> [u32; 8] {
    let uni = wf.uni16(dev);
    let dummy = wf.dummy_ro(dev);
    let dummy16 = wf.empty16(dev);
    let qs = n >> k_whir;
    let mut size = n;
    let mut nxt = ShardBuf::new(dev, "m nxt", SHARD_USAGES, ((n / 2) * 32) as u64);
    let mut src = cur;
    let mut ping = ShardBuf::new(dev, "m ping", SHARD_USAGES, ((n / 2) * 32) as u64);
    let mut use_nxt = true;
    // phase-1 strided layers (window <= qs output rows: x = i0/qs const)
    let wrows_s = (dev.win() / 32).min(qs as u64).max(1) as usize;
    for _ in 0..k_whir {
        let nsz = size / 2;
        let dst: &ShardBuf = if use_nxt { &nxt } else { &ping };
        let mut i0 = 0usize;
        while i0 < nsz {
            let w = wrows_s.min(nsz - i0);
            let x0 = i0 / qs;
            WhirFold::set_uni(dev, &uni, nsz as u32, qs as u32, 0, i0 as u32);
            let bg = wf.bind_res(
                dev,
                src.bind(dev, (2 * x0 * qs * 32) as u64, (w * 32) as u64),
                src.bind(dev, ((2 * x0 + 1) * qs * 32) as u64, (w * 32) as u64),
                dst.bind(dev, (i0 * 32) as u64, (w * 32) as u64),
                &uni,
                dummy.as_entire_binding(),
            );
            WhirFold::dispatch(dev, &wf.pair_strided, &bg, w);
            i0 += w;
        }
        src = if use_nxt { &nxt } else { &ping };
        use_nxt = !use_nxt;
        size = nsz;
    }
    // phase-2 adjacent layers: out window w (64 B in-row) <= win
    let wrows_a = (dev.win() / 64).max(1) as usize;
    while size > 1 {
        let nsz = size / 2;
        let dst: &ShardBuf = if use_nxt { &nxt } else { &ping };
        let mut i0 = 0usize;
        while i0 < nsz {
            let w = wrows_a.min(nsz - i0);
            WhirFold::set_uni(dev, &uni, nsz as u32, 0, 0, i0 as u32);
            let bg = wf.bind_res(
                dev,
                src.bind(dev, (i0 * 64) as u64, (w * 64) as u64),
                dst.bind(dev, (i0 * 32) as u64, (w * 32) as u64),
                dummy16.as_entire_binding(),
                &uni,
                dummy.as_entire_binding(),
            );
            WhirFold::dispatch(dev, &wf.pair_adj, &bg, w);
            i0 += w;
        }
        src = if use_nxt { &nxt } else { &ping };
        use_nxt = !use_nxt;
        size = nsz;
    }
    let root_m = src.read_words(dev, 0, 8);
    std::array::from_fn(|i| decode(root_m[i]))
}

/// convenience: rows (n x 8 words) -> full tree -> decoded root.
pub fn merkle_from_rows(
    dev: &WgpuDevice,
    wf: &WhirFold,
    rows: &ShardBuf,
    n: usize,
    k_whir: usize,
) -> [u32; 8] {
    let cur = ShardBuf::new(dev, "m cur", SHARD_USAGES, (n * 32) as u64);
    merkle_init(dev, wf, rows, &cur, n);
    merkle_reduce(dev, wf, &cur, n, k_whir)
}

// ================= W2C/W2D-BROWSER async port (Phase 3d) =================
//
// Awaitable twins of the fold-path readbacks (same cfg split as
// gkr::download_off_async: native = the exact sync poll(Wait) tail, wasm32 =
// oneshot await of the mapAsync callback; device.poll is a NO-OP on the web
// backend). Dispatch-only helpers (whir_stages / merkle_init / dispatch)
// have no host readback and work in the browser unchanged.

impl WhirFold {
    /// Awaitable twin of `read_back`.
    pub async fn read_back_async(dev: &WgpuDevice, b: &wgpu::Buffer, nwords: usize) -> Vec<u32> {
        let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb"),
            size: (nwords * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(b, 0, &staging, 0, (nwords * 4) as u64);
        dev.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            dev.device
                .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                .unwrap();
            rx.recv().unwrap().unwrap();
        }
        #[cfg(target_arch = "wasm32")]
        {
            let (tx, rx) =
                futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rx.await.unwrap().unwrap();
        }
        let mut out = vec![0u32; nwords];
        let d = slice.get_mapped_range().unwrap();
        for (i, w) in out.iter_mut().enumerate() {
            *w = u32::from_le_bytes(d[i * 4..i * 4 + 4].try_into().unwrap());
        }
        drop(d);
        staging.unmap();
        out
    }
}

/// Awaitable twin of `merkle_reduce` (only the final root read differs).
pub async fn merkle_reduce_async(
    dev: &WgpuDevice,
    wf: &WhirFold,
    cur: &ShardBuf,
    n: usize,
    k_whir: usize,
) -> [u32; 8] {
    let uni = wf.uni16(dev);
    let dummy = wf.dummy_ro(dev);
    let dummy16 = wf.empty16(dev);
    let qs = n >> k_whir;
    let mut size = n;
    let mut nxt = ShardBuf::new(dev, "m nxt", SHARD_USAGES, ((n / 2) * 32) as u64);
    let mut src = cur;
    let mut ping = ShardBuf::new(dev, "m ping", SHARD_USAGES, ((n / 2) * 32) as u64);
    let mut use_nxt = true;
    let wrows_s = (dev.win() / 32).min(qs as u64).max(1) as usize;
    for _ in 0..k_whir {
        let nsz = size / 2;
        let dst: &ShardBuf = if use_nxt { &nxt } else { &ping };
        let mut i0 = 0usize;
        while i0 < nsz {
            let w = wrows_s.min(nsz - i0);
            let x0 = i0 / qs;
            WhirFold::set_uni(dev, &uni, nsz as u32, qs as u32, 0, i0 as u32);
            let bg = wf.bind_res(
                dev,
                src.bind(dev, (2 * x0 * qs * 32) as u64, (w * 32) as u64),
                src.bind(dev, ((2 * x0 + 1) * qs * 32) as u64, (w * 32) as u64),
                dst.bind(dev, (i0 * 32) as u64, (w * 32) as u64),
                &uni,
                dummy.as_entire_binding(),
            );
            WhirFold::dispatch(dev, &wf.pair_strided, &bg, w);
            i0 += w;
        }
        src = if use_nxt { &nxt } else { &ping };
        use_nxt = !use_nxt;
        size = nsz;
    }
    let wrows_a = (dev.win() / 64).max(1) as usize;
    while size > 1 {
        let nsz = size / 2;
        let dst: &ShardBuf = if use_nxt { &nxt } else { &ping };
        let mut i0 = 0usize;
        while i0 < nsz {
            let w = wrows_a.min(nsz - i0);
            WhirFold::set_uni(dev, &uni, nsz as u32, 0, 0, i0 as u32);
            let bg = wf.bind_res(
                dev,
                src.bind(dev, (i0 * 64) as u64, (w * 64) as u64),
                dst.bind(dev, (i0 * 32) as u64, (w * 32) as u64),
                dummy16.as_entire_binding(),
                &uni,
                dummy.as_entire_binding(),
            );
            WhirFold::dispatch(dev, &wf.pair_adj, &bg, w);
            i0 += w;
        }
        src = if use_nxt { &nxt } else { &ping };
        use_nxt = !use_nxt;
        size = nsz;
    }
    let root_m = src.read_words_async(dev, 0, 8).await;
    std::array::from_fn(|i| decode(root_m[i]))
}

/// Awaitable twin of `merkle_from_rows`.
pub async fn merkle_from_rows_async(
    dev: &WgpuDevice,
    wf: &WhirFold,
    rows: &ShardBuf,
    n: usize,
    k_whir: usize,
) -> [u32; 8] {
    let cur = ShardBuf::new(dev, "m cur", SHARD_USAGES, (n * 32) as u64);
    merkle_init(dev, wf, rows, &cur, n);
    merkle_reduce_async(dev, wf, &cur, n, k_whir).await
}
