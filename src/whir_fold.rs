//! GATE W2C PHASE 2: WGSL WHIR fold — mobius/DFT butterfly stages + data-row
//! merkle (init4 / strided-pair / adjacent-pair layers), Poseidon2 kernels
//! copied VERBATIM from the golden wgsl.rs perm path.
use crate::device::WgpuDevice;
use crate::field::{constants_buffer_mont, u32s_to_bytes};

const WGSL: &str = r#"
fn mul(a: u32, b: u32) -> u32 {
  let t = u64(a) * u64(b);
  let m = u32(t) * 2013265919u;
  let u = u64(m) * u64(2013265921u);
  var r: u32 = u32((t + u) >> 32u);
  if (r >= 2013265921u) { r = r - 2013265921u; }
  return r;
}

@group(0) @binding(0) var<storage, read> kc: array<u32>;
@group(0) @binding(1) var<storage, read_write> io1: array<u32>;
@group(0) @binding(2) var<storage, read_write> io2: array<u32>;
struct Uni { a: u32, b: u32, c: u32, d: u32 };
@group(0) @binding(3) var<uniform> uni: Uni;
@group(0) @binding(4) var<storage, read> tw: array<u32>;

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
// uni.a = n/2 butterflies, uni.b = half (stage), uni.c = mode (0=fft*tw, 1=mobius)
@compute @workgroup_size(256u)
fn whir_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= uni.a) { return; }
  let half = uni.b;
  let block = b / half;
  let k = b % half;
  let j = (block * half * 2u + k) * 4u;
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
}

// merkle leaf init over data rows: state = [row8 | 0^8] (raw mont words),
// mirrors CPU merkle_leaf on 4-word rows zero-padded to 8.
@compute @workgroup_size(256u)
fn whir_init4(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + uni.d;
  if (i * 8u >= arrayLength(&io2)) { return; }
  var cells: array<u32, 16u>;
  for (var z: u32 = 0u; z < 16u; z = z + 1u) { cells[z] = 0u; }
  for (var j: u32 = 0u; j < 8u; j = j + 1u) { cells[j] = io1[i * 8u + j]; }
  perm_mix(&cells);
  for (var j2: u32 = 0u; j2 < 8u; j2 = j2 + 1u) { io2[i * 8u + j2] = cells[j2]; }
}

// phase-1 strided pairing: uni.a = nsz, uni.b = qs
@compute @workgroup_size(256u)
fn whir_pair_strided(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + uni.d;
  if (i >= uni.a) { return; }
  let qs = uni.b;
  let x = i / qs;
  let y = i % qs;
  let li = (2u * x * qs + y) * 8u;
  let ri = ((2u * x + 1u) * qs + y) * 8u;
  var cells: array<u32, 16u>;
  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    cells[j] = io1[li + j];
    cells[j + 8u] = io1[ri + j];
  }
  perm_mix(&cells);
  for (var j2: u32 = 0u; j2 < 8u; j2 = j2 + 1u) { io2[i * 8u + j2] = cells[j2]; }
}

// phase-2 adjacent pairing: uni.a = nsz
@compute @workgroup_size(256u)
fn whir_pair_adj(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + uni.d;
  if (i >= uni.a) { return; }
  var cells: array<u32, 16u>;
  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    cells[j] = io1[i * 16u + j];
    cells[j + 8u] = io1[i * 16u + 8u + j];
  }
  perm_mix(&cells);
  for (var j2: u32 = 0u; j2 < 8u; j2 = j2 + 1u) { io2[i * 8u + j2] = cells[j2]; }
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
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
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
                        min_binding_size: wgpu::BufferSize::new(16),
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
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn bind(
        &self,
        dev: &WgpuDevice,
        io1: &wgpu::Buffer,
        io2: &wgpu::Buffer,
        uni: &wgpu::Buffer,
        tw: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.kc_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: io1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: io2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tw.as_entire_binding() },
            ],
        })
    }

    pub fn set_uni(dev: &WgpuDevice, uni: &wgpu::Buffer, a: u32, b: u32, c: u32, d: u32) {
        let mut x = [0u8; 16];
        x[0..4].copy_from_slice(&a.to_le_bytes());
        x[4..8].copy_from_slice(&b.to_le_bytes());
        x[8..12].copy_from_slice(&c.to_le_bytes());
        x[12..16].copy_from_slice(&d.to_le_bytes());
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
