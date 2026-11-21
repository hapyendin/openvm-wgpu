// Device discovery + init (API shapes mirror wgpu 30.0.2 as used by the
// golden-proven wgpu-probe). Vulkan-only backend. On b70 asserts Intel
// vendor 0x8086 discrete (Arc); the farm-c1 run passes require_intel_arc=false.
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
    /// true iff device limits were pinned to the WebGPU spec floors
    pub webgpu_limits: bool,
}

pub struct InitOpts {
    /// require Intel (0x8086) discrete GPU — b70 law; false = any discrete
    pub require_intel_arc: bool,
    /// tolerate missing SHADER_INT64 (pure-u32 limb path)
    pub allow_u32_only: bool,
}

/// WebGPU spec-floor limits (pinned in code, NOT downlevel defaults —
/// those differ and would hide violations). Requested as
/// floor.min(adapter.limits()); the effective values are printed.
pub const WEBGPU_FLOOR_BINDING: u64 = 134_217_728; // 128 MiB
pub const WEBGPU_FLOOR_BUFFER: u64 = 268_435_456; // 256 MiB
pub const WEBGPU_FLOOR_WG_SMEM: u32 = 16_384;
pub const WEBGPU_FLOOR_WG_INV: u32 = 256;

pub fn webgpu_limits_env() -> bool {
    std::env::var("OPENVM_WGPU_WEBGPU_LIMITS").as_deref() == Ok("1")
}

pub fn pow2_floor(x: u64) -> u64 {
    if x < 2 {
        1
    } else {
        1u64 << (63 - x.leading_zeros())
    }
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
    /// Native sync entry (pollster-driven). On wasm `init` is unavailable
    /// (no thread-blocking event-loop pumping in the browser); wasm tiers
    /// call `init_async` / `init_browser`.
    pub fn init(opts: InitOpts) -> WgpuDevice {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(WgpuDevice::init_async(opts))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = opts;
            panic!("WgpuDevice::init is native-only on wasm; use init_async/init_browser")
        }
    }

    /// Async core of device init. Native behavior identical to the
    /// historical sync body (same env knobs, asserts, prints); on wasm the
    /// env knobs read as unset (browser = spec floors, u32 path).
    pub async fn init_async(opts: InitOpts) -> WgpuDevice {
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .expect("no vulkan adapter");
        let info = adapter.get_info();
        let is_intel = info.vendor == 0x8086;
        // OPENVM_WGPU_FARM=1 = documented farm path (v2 portability rounds):
        // accepts the first Vulkan adapter of any vendor; Arc goldens unaffected.
        let farm = std::env::var("OPENVM_WGPU_FARM").is_ok();
        assert!(
            !opts.require_intel_arc || is_intel || farm,
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
            has_i64 || opts.allow_u32_only || force_u32,
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
        // OPENVM_WGPU_WEBGPU_LIMITS=1: pin the requested device limits to
        // the WebGPU spec floors so wgpu's own validation audits every
        // bind group and buffer we create.
        let wl = webgpu_limits_env();
        let al = adapter.limits();
        let req_limits = if wl {
            wgpu::Limits {
                max_bind_groups: 4.min(al.max_bind_groups),
                max_bindings_per_bind_group: 640.min(al.max_bindings_per_bind_group),
                max_storage_buffer_binding_size: WEBGPU_FLOOR_BINDING
                    .min(al.max_storage_buffer_binding_size),
                max_buffer_size: WEBGPU_FLOOR_BUFFER.min(al.max_buffer_size),
                max_compute_workgroup_storage_size: WEBGPU_FLOOR_WG_SMEM
                    .min(al.max_compute_workgroup_storage_size),
                max_compute_invocations_per_workgroup: WEBGPU_FLOOR_WG_INV
                    .min(al.max_compute_invocations_per_workgroup),
                max_compute_workgroup_size_x: 256.min(al.max_compute_workgroup_size_x),
                max_compute_workgroup_size_y: 256.min(al.max_compute_workgroup_size_y),
                max_compute_workgroup_size_z: 64.min(al.max_compute_workgroup_size_z),
                max_compute_workgroups_per_dimension: 65535
                    .min(al.max_compute_workgroups_per_dimension),
                ..al
            }
        } else {
            al
        };
        println!(
            "WEBGPU_LIMITS_MODE: {} (floors binding={} buffer={} wg_smem={} wg_inv={})",
            wl, WEBGPU_FLOOR_BINDING, WEBGPU_FLOOR_BUFFER, WEBGPU_FLOOR_WG_SMEM, WEBGPU_FLOOR_WG_INV
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("openvm-wgpu"),
                required_features: want,
                required_limits: req_limits,
                ..Default::default()
            })
            .await
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
            // report the EFFECTIVE device limits (what wgpu validates against)
            limits: limits_of(&device.limits()),
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
        #[cfg(not(target_arch = "wasm32"))]
        println!("DEVICE_INIT_MS {:.1}", t0.elapsed().as_secs_f64() * 1e3);
        WgpuDevice {
            device,
            queue,
            report,
            has_shader_int64: has_i64,
            has_int64_atomic_feature: has_atomic,
            webgpu_limits: wl,
        }
    }

    /// Browser entry: BROWSER_WEBGPU backend, request adapter with
    /// compatible_surface None, device with NO extra features and
    /// `Limits::default()` (on wasm that IS the WebGPU spec-floor set —
    /// the browser is the floor). Adapter vendor is reported, not asserted.
    pub async fn init_browser() -> WgpuDevice {
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .expect("no webgpu adapter (WebGPU unavailable in this browser?)");
        let info = adapter.get_info();
        println!(
            "ADAPTER name={:?} vendor=0x{:x} device=0x{:x} driver={:?} driver_info={:?} type={:?} backend={:?}",
            info.name, info.vendor, info.device, info.driver, info.driver_info, info.device_type, info.backend
        );
        println!(
            "ADAPTER_IS_INTEL_ARC: {}",
            info.vendor == 0x8086 || info.name.to_lowercase().contains("arc")
        );
        let feats_dbg = format!("{:?}", adapter.features());
        let avail: Vec<String> = feature_probe_list()
            .iter()
            .filter(|(_, f)| adapter.features().contains(*f))
            .map(|(n, _)| n.to_string())
            .collect();
        let avail_s = if avail.is_empty() { "-".to_string() } else { avail.join(",") };
        println!("FEATURES_AVAILABLE: {avail_s}");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("openvm-wasm"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("request_device failed");
        let report = AdapterReport {
            name: info.name.clone(),
            vendor_id: info.vendor,
            device_id: info.device,
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            device_type: format!("{:?}", info.device_type),
            features_dbg: feats_dbg,
            available: avail,
            enabled: Vec::new(),
            limits: limits_of(&device.limits()),
        };
        println!(
            "LIMITS storage_binding_max={} buffer_max={} wg_smem={} wg_inv={} wg_per_dim={} bindings_per_bg={}",
            report.limits.max_storage_buffer_binding_size,
            report.limits.max_buffer_size,
            report.limits.max_compute_workgroup_storage_size,
            report.limits.max_compute_invocations_per_workgroup,
            report.limits.max_compute_workgroups_per_dimension,
            report.limits.max_bindings_per_bind_group
        );
        #[cfg(not(target_arch = "wasm32"))]
        println!("DEVICE_INIT_MS {:.1}", t0.elapsed().as_secs_f64() * 1e3);
        WgpuDevice {
            device,
            queue,
            report,
            has_shader_int64: false,
            has_int64_atomic_feature: false,
            webgpu_limits: true,
        }
    }

    /// Effective device cap on one storage binding (bytes).
    pub fn bind_cap(&self) -> u64 {
        self.report.limits.max_storage_buffer_binding_size
    }
    /// Effective device cap on one buffer (bytes).
    pub fn buf_cap(&self) -> u64 {
        self.report.limits.max_buffer_size
    }
    /// pow2-aligned window for sliced binds of large buffers.
    /// 256 MiB ceiling keeps windows pow2-divisors of the 1 GiB shard
    /// size used on unlimited devices (containment for free).
    pub fn win(&self) -> u64 {
        self.bind_cap().min(256 << 20)
    }
    /// GKR pool/eq window row count: 160 B per row, 640 MiB geometry cap
    /// (preserves the historical wsz = 2^22 on ANV, 2^19 under floors).
    pub fn gkr_wsz(&self) -> u32 {
        pow2_floor(self.bind_cap().min(640 << 20) / 160) as u32
    }
    /// GKR leaf-chunk bytes for a 2^t-row leaf (32 B per row). Row count
    /// is pow2 and bounded by the unpack bind window (128 B/row in pool
    /// layout) and the pool window, so every unpack bind is legal.
    pub fn leaf_chunk_bytes(&self, t: u32) -> u64 {
        (self.win() / 128).min(self.gkr_wsz() as u64).min(1u64 << (t - 2)) * 32
    }

    /// Runtime ground-truth probe: u64 atomicStore/CAS/ADD in one WGSL
    /// kernel. Ok(Some(...)) if the driver accepted AND ran u64 atomics;
    /// Ok(None) if the wgpu feature flag is absent; Err if the feature was
    /// advertised but compilation/execution failed (naga error surfaces via
    /// device error handler — we also force a poll + error-scope check).
    pub fn probe_u64_atomics(&self) -> Option<(u32, u32, u32)> {
        if !self.has_shader_int64 || !self.has_int64_atomic_feature {
            return None;
        }
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
            .ok()?;
        let data = slice.get_mapped_range().ok()?;
        let lo = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let hi = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let winners = u32::from_le_bytes(data[8..12].try_into().unwrap());
        drop(data);
        staging.unmap();
        Some((winners, lo, hi))
    }
}
