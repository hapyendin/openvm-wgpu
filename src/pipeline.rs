// Pipeline set: one WGSL module, four entry points, shared bind layout:
//  b0 kc (u32 constants, read), b1 io1 (rw), b2 io2 (rw), b3 uniform(seed).
// Layout mirrors the CUDA launcher-cuh pattern: one "upload constants once,
// dispatch kernels" scheme per proving context.
//
// C3 adds FRI commit pipelines: fri_commit_8col, fri_commit_1col

use crate::device::WgpuDevice;

pub struct PipelineSet {
    pub bgl: wgpu::BindGroupLayout,
    pub perm_batch: wgpu::ComputePipeline,
    pub derive_row: wgpu::ComputePipeline,
    pub merkle_init: wgpu::ComputePipeline,
    pub merkle_compress: wgpu::ComputePipeline,
    pub fri_commit_8col: wgpu::ComputePipeline,
    pub fri_commit_1col: wgpu::ComputePipeline,
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

impl PipelineSet {
    pub fn new(dev: &WgpuDevice, u64_mul: bool) -> PipelineSet {
        let module = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openvm-wgpu"),
            source: wgpu::ShaderSource::Wgsl(crate::wgsl::source(u64_mul).into()),
        });
        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("4-slot"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
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
                ],
            });
        let layout = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipe = |name: &'static str, label: &'static str| {
            dev.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        PipelineSet {
            bgl,
            perm_batch: pipe("perm_batch", "perm_batch"),
            derive_row: pipe("derive_row", "derive_row"),
            merkle_init: pipe("merkle_init", "merkle_init"),
            merkle_compress: pipe("merkle_compress", "merkle_compress"),
            fri_commit_8col: pipe("fri_commit_8col", "fri_commit_8col"),
            fri_commit_1col: pipe("fri_commit_1col", "fri_commit_1col"),
        }
    }
}

pub struct KernelCtx {
    pub kc_buf: wgpu::Buffer,
    pub uni_buf: wgpu::Buffer,
}

impl KernelCtx {
    pub fn new(dev: &WgpuDevice, kc_mont: &[u32]) -> KernelCtx {
        let kc_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kc"),
            size: (kc_mont.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue
            .write_buffer(&kc_buf, 0, &crate::field::u32s_to_bytes(kc_mont));
        let uni_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uni"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue
            .write_buffer(&uni_buf, 0, &[0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        KernelCtx { kc_buf, uni_buf }
    }

    pub fn set_seed(&self, dev: &WgpuDevice, seed_canon: u32) {
        let mut b = [0u8; 16];
        b[0..4].copy_from_slice(&seed_canon.to_le_bytes());
        dev.queue.write_buffer(&self.uni_buf, 0, &b);
    }

    pub fn bind(&self, dev: &WgpuDevice, ps: &PipelineSet, io1: &wgpu::Buffer, io2: &wgpu::Buffer) -> wgpu::BindGroup {
        dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ps.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.kc_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: io1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: io2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uni_buf.as_entire_binding(),
                },
            ],
        })
    }
}

pub fn dispatch(dev: &WgpuDevice, pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, threads: usize) {
    let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups((threads as u32).div_ceil(32), 1, 1);
    }
    dev.queue.submit(Some(enc.finish()));
}

// ---- NTT 4096-pt (single workgroup, 512 threads) ----
// DIT-NTT with bit-reversed input, natural output. Twiddles are Montgomery words.
pub struct NttPipelineSet {
    pub bgl: wgpu::BindGroupLayout,
    pub fwd: wgpu::ComputePipeline,
    pub inv: wgpu::ComputePipeline,
}

impl NttPipelineSet {
    pub fn new(dev: &WgpuDevice, u64_mul: bool) -> NttPipelineSet {
        let module = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ntt"),
            source: wgpu::ShaderSource::Wgsl(crate::wgsl::ntt_source(u64_mul).into()),
        });
        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ntt-3buf"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                ],
            });
        let layout = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipe = |name: &'static str| {
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
        NttPipelineSet {
            bgl,
            fwd: pipe("ntt_4096_fwd"),
            inv: pipe("ntt_4096_inv"),
        }
    }
}

pub fn dispatch_ntt(dev: &WgpuDevice, pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup) {
    let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    dev.queue.submit(Some(enc.finish()));
}
