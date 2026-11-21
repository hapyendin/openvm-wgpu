//! W2B/W2C-BROWSER (§7c quirk): a COPY-only command encoder submitted right
//! after `queue.write_buffer` observes PRE-write contents on the web backend.
//! Dispatches submitted after a write DO observe it (proven by the W2B leaf:
//! blob write_slice -> DAG dispatch reads correct bytes; and leaf-sha copy
//! readbacks after DISPATCH-written buffers are exact). So to read back a
//! buffer whose last CPU-side touch was `queue.write_buffer`, first move it
//! with a DISPATCH (this copy kernel, src->dst across two buffers), then
//! map-copy the dispatch-written dst (dispatch-written copy readbacks are
//! the proven-good pattern).

use crate::device::WgpuDevice;
use crate::limitsbuf::ShardBuf;

const WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src_arr: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst_arr: array<u32>;
@compute @workgroup_size(64)
fn rcopy(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < arrayLength(&dst_arr)) {
        dst_arr[gid.x] = src_arr[gid.x];
    }
}
"#;

pub struct ReadFix {
    pipe: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl ReadFix {
    pub fn new(dev: &WgpuDevice) -> Self {
        let m = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("readfix-copy"),
                source: wgpu::ShaderSource::Wgsl(WGSL.into()),
            });
        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("readfix-bgl"),
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
        let pl = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("readfix-pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipe = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("readfix-pipe"),
                layout: Some(&pl),
                module: &m,
                entry_point: Some("rcopy"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self { pipe, bgl }
    }

    /// Awaitable read of a ShardBuf range that is SAFE right after
    /// queue.write_buffer: per-shard-chunk dispatch-copy -> map-copy read.
    pub async fn read_words_after_write(
        &self,
        dev: &WgpuDevice,
        sb: &ShardBuf,
        word_off: u64,
        nwords: u64,
    ) -> Vec<u32> {
        let mut out = vec![0u32; nwords as usize];
        let mut done = 0u64;
        while done < nwords {
            let o = (word_off + done) * 4;
            let (i, so) = sb.loc(o);
            let take = (nwords - done)
                .min((sb.shards[i].size() - so) / 4)
                .min(65535 * 64); // 4 191 936 words: keeps workgroup count <= 65535
            assert!(take > 0, "readfix stalled");
            let dst = dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readfix-dst"),
                size: take * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("readfix-bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &sb.shards[i],
                            offset: so,
                            size: Some(std::num::NonZeroU64::new(take * 4).unwrap()),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                ],
            });
            {
                let mut enc = dev.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("readfix-dispatch") },
                );
                {
                    let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("readfix"),
                        timestamp_writes: None,
                    });
                    p.set_pipeline(&self.pipe);
                    p.set_bind_group(0, &bg, &[]);
                    p.dispatch_workgroups((take.div_ceil(64)) as u32, 1, 1);
                }
                dev.queue.submit(Some(enc.finish()));
            }
            // dst was written by a DISPATCH -> map readback pattern is the
            // proven-good one (same as the leaf-sha readbacks).
            let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readfix-stage"),
                size: take * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&dst, 0, &staging, 0, take * 4);
            dev.queue.submit(Some(enc.finish()));
            let slice = staging.slice(..);
            #[cfg(not(target_arch = "wasm32"))]
            {
                slice.map_async(wgpu::MapMode::Read, |_| ());
                dev.device
                    .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                    .expect("readfix poll");
            }
            #[cfg(target_arch = "wasm32")]
            {
                let (tx, rx) =
                    futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                rx.await
                    .map_err(|e| e.to_string())
                    .and_then(|r| r.map_err(|e| e.to_string()))
                    .expect("readfix map");
            }
            {
                let data = slice.get_mapped_range().expect("readfix mapped");
                for k in 0..take as usize {
                    out[(done + k as u64) as usize] =
                        u32::from_le_bytes(data[4 * k..4 * k + 4].try_into().unwrap());
                }
            }
            staging.unmap();
            done += take;
        }
        out
    }
}
