// Device discovery + init (API shapes mirror wgpu 30.0.2 as used by the
// golden-proven wgpu-probe). Vulkan-only backend. GATE W1: adapter selection
// is capability-driven (see adapters.rs) instead of "first adapter":
// SHADER_INT64 preferred, then max_buffer_size, then discrete. require_intel_arc
// stays as strict-mode (OPENVM_WGPU_ARC=1 / --arc); WGPU_DEVICE=<substring>
// pins by name. No adapter passes => FAIL CLOSED with the table printed.
#[derive(Debug, Clone, Copy)]
pub struct LimitsReport {
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_bindings_per_bind_group: u32,
}

#[derive(Debug)]
pub struct AdapterReport {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver: String,
    pub driver_info: String,
    pub device_type: String,
    pub features_dbg: String,
    pub available: Vec<String>,
    pub enabled: Vec<String>,
    pub limits: LimitsReport,
}

pub struct WgpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub report: AdapterReport,
    /// true iff SHADER_INT64 was granted (u64-REDC mul path)
    pub has_shader_int64: bool,
    /// adapter advertises the wgpu SHADER_INT64_ATOMIC feature
    pub has_int64_atomic_feature: bool,
}

pub struct InitOpts {
    /// require Intel (0x8086) discrete GPU — b70 strict-mode law; false = capability pick
    pub require_intel_arc: bool,
    /// tolerate missing SHADER_INT64 (pure-u32 limb path)
    pub allow_u32_only: bool,
    /// WGPU_DEVICE=<substring>: case-insensitive substring match on adapter
    /// name pins the adapter; no match fails closed. None = capability pick.
    pub device_substring: Option<String>,
}

fn feature_probe_list() -> Vec<(&'static str, wgpu::Features)> {
    use wgpu::Features as F;
    vec![
        ("SHADER_INT64", F::SHADER_INT64),
        ("SHADER_INT64_ATOMIC_MIN_MAX", F::SHADER_INT64_ATOMIC_MIN_MAX),
        ("SHADER_INT64_ATOMIC_ALL_OPS", F::SHADER_INT64_ATOMIC_ALL_OPS),
        ("TEXTURE_INT64_ATOMIC", F::TEXTURE_INT64_ATOMIC),
        ("TIMESTAMP_QUERY", F::TIMESTAMP_QUERY),
        ("TIMESTAMP_QUERY_INSIDE_ENCODERS", F::TIMESTAMP_QUERY_INSIDE_ENCODERS),
    ]
}

fn limits_of(l: &wgpu::Limits) -> LimitsReport {
    LimitsReport {
        max_storage_buffer_binding_size: l.max_storage_buffer_binding_size,
        max_buffer_size: l.max_buffer_size,
        max_compute_workgroup_storage_size: l.max_compute_workgroup_storage_size,
        max_compute_invocations_per_workgroup: l.max_compute_invocations_per_workgroup,
        max_compute_workgroups_per_dimension: l.max_compute_workgroups_per_dimension,
        max_bindings_per_bind_group: l.max_bindings_per_bind_group,
    }
}

impl WgpuDevice {
    pub fn init(opts: InitOpts) -> WgpuDevice {
        let t0 = std::time::Instant::now();
        // ---- GATE W1 selection path: enumerate + capability-driven pick ----
        let enumerated = crate::adapters::enumerate_full();
        let infos: Vec<crate::adapters::AdapterInfo> =
            enumerated.iter().map(|e| e.info.clone()).collect();
        let idx = match crate::adapters::select(
            &infos,
            opts.device_substring.as_deref(),
            opts.require_intel_arc,
            opts.allow_u32_only,
        ) {
            Ok(i) => i,
            Err(reason) => {
                crate::adapters::print_table(&infos);
                panic!("FAIL CLOSED: {reason}");
            }
        };
        let adapter = &enumerated[idx].adapter;
        println!(
            "SELECTED idx={idx}/{} strategy={} score={:?}",
            infos.len(),
            if opts.device_substring.is_some() {
                "WGPU_DEVICE substring pin"
            } else {
                "capability (int64, buffer_max, discrete)"
            },
            crate::adapters::capability_score(&infos[idx])
        );
        let info = adapter.get_info();
        let is_intel = info.vendor == 0x8086;
        assert!(
            !opts.require_intel_arc || is_intel,
            "ADAPTER ASSERT FAIL: expected Intel Arc (vendor 0x8086), got vendor 0x{:x} ({})",
            info.vendor,
            info.name
        );
        assert!(
            info.backend == wgpu::Backend::Vulkan,
            "expected Vulkan backend, got {:?}",
            info.backend
        );
        println!(
            "ADAPTER name={:?} vendor=0x{:x} device=0x{:x} driver={:?} driver_info={:?} type={:?}",
            info.name, info.vendor, info.device, info.driver, info.driver_info, info.device_type
        );
        println!(
            "ADAPTER_IS_INTEL_ARC: {}",
            info.name.to_lowercase().contains("arc") || is_intel
        );
        let feats_dbg = format!("{:?}", adapter.features());
        let avail: Vec<String> = feature_probe_list()
            .iter()
            .filter(|(_, f)| adapter.features().contains(*f))
            .map(|(n, _)| n.to_string())
            .collect();
        println!("FEATURES_AVAILABLE: {}", avail.join(","));
        // OPENVM_WGPU_FORCE_U32=1: spec-portable path — SHADER_INT64* features
        // are NOT requested even when the adapter advertises them, so any
        // int64 in a compiled shader is a hard device-error (self-auditing).
        let force_u32 = std::env::var("OPENVM_WGPU_FORCE_U32")
            .map(|v| v == "1")
            .unwrap_or(false);
        let has_i64 = !force_u32 && adapter.features().contains(wgpu::Features::SHADER_INT64);
        let has_atomic = !force_u32
            && adapter
            .features()
            .contains(wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS);
        if force_u32 {
            println!("OPENVM_WGPU_FORCE_U32=1: SHADER_INT64 features excluded (u32-limb path)");
        }
        let has_atomic_minmax = adapter
            .features()
            .contains(wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX);
        println!("SHADER_INT64_ATOMIC_ALL_OPS_available: {has_atomic}");
        println!("SHADER_INT64_ATOMIC_MIN_MAX_available: {has_atomic_minmax}");
        assert!(
            has_i64 || opts.allow_u32_only,
            "SHADER_INT64 unavailable and u32-fallback not allowed"
        );
        let mut want = wgpu::Features::empty();
        if has_i64 {
            want |= wgpu::Features::SHADER_INT64;
        }
        if has_atomic {
            // enable so the u64-atomic probe below can compile; cheap on ANV
            want |= wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS;
        }
        // NB: SHADER_INT64_ATOMIC is *not* required here — the gate's
        // correctness path is atomic-free. The atomic probe below reports
        // ground-truth availability separately.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("openvm-wgpu"),
            required_features: want,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("request_device failed");
        let enabled: Vec<String> = want.iter().map(|f| f.to_string()).collect();
        let report = AdapterReport {
            name: info.name.clone(),
            vendor_id: info.vendor,
            device_id: info.device,
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            device_type: format!("{:?}", info.device_type),
            features_dbg: feats_dbg,
            available: avail,
            enabled,
            limits: limits_of(&adapter.limits()),
        };
        println!("FEATURES_ENABLED: {}", report.enabled.join(","));
        println!(
            "LIMITS storage_binding_max={} buffer_max={} wg_smem={} wg_inv={} wg_per_dim={} bindings_per_bg={}",
            report.limits.max_storage_buffer_binding_size,
            report.limits.max_buffer_size,
            report.limits.max_compute_workgroup_storage_size,
            report.limits.max_compute_invocations_per_workgroup,
            report.limits.max_compute_workgroups_per_dimension,
            report.limits.max_bindings_per_bind_group
        );
        println!("FEATURES_FULL {:?}", report.features_dbg);
        println!("DEVICE_INIT_MS {:.1}", t0.elapsed().as_secs_f64() * 1e3);
        WgpuDevice {
            device,
            queue,
            report,
            has_shader_int64: has_i64,
            has_int64_atomic_feature: has_atomic,
        }
    }

    /// Runtime ground-truth probe: u64 atomicStore/CAS/ADD in one WGSL
    /// kernel. Ok(Some(...)) if the driver accepted AND ran u64 atomics;
    /// Ok(None) if the wgpu feature flag is absent; Err if the probe module
    /// failed to compile/execute (the wgpu 30 default error sink panics on
    /// validation errors — e.g. naga@30 rejects u64 literals > i64::MAX in
    /// the probe WGSL — so the whole probe runs under catch_unwind and the
    /// failure surfaces as Err; the golden kernels are unaffected because
    /// they never reference ATOMIC_PROBE).
    pub fn probe_u64_atomics(&self) -> Result<Option<(u32, u32, u32)>, String> {
        if !self.has_shader_int64 || !self.has_int64_atomic_feature {
            return Ok(None);
        }
        // silence the panic hook during the probe; restore it afterwards.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.probe_u64_atomics_inner()
        }));
        std::panic::set_hook(prev_hook);
        match result {
            Ok(inner) => inner,
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                Err(format!("probe panicked (naga/driver rejected probe WGSL?): {msg}"))
            }
        }
    }

    fn probe_u64_atomics_inner(&self) -> Result<Option<(u32, u32, u32)>, String> {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("atomic-probe"),
                source: wgpu::ShaderSource::Wgsl(crate::wgsl::ATOMIC_PROBE.into()),
            });
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("atomic-probe"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let mk = |size: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let at = mk(16);
        let out = mk(16);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: at.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            enc.copy_buffer_to_buffer(&out, 0, &staging, 0, 16);
            self.queue.submit(Some(enc.finish()));
        }
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| format!("probe poll failed: {e}"))?;
        let data = slice
            .get_mapped_range()
            .map_err(|e| format!("probe map failed: {e}"))?;
        let lo = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let hi = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let winners = u32::from_le_bytes(data[8..12].try_into().unwrap());
        drop(data);
        staging.unmap();
        Ok(Some((winners, lo, hi)))
    }
}
