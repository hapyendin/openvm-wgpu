// GATE W1: adapter enumeration + capability-driven selection.
//
// Host-layer only — no WGSL/kernel changes. Honest reporting: everything
// printed here is exactly what wgpu/the Vulkan driver advertises, nothing
// inferred. Used by `WgpuDevice::init` for selection and by the `--list-adapters`
// flag on every binary.
//
// Selection law (community-test module):
//   1. WGPU_DEVICE=<substring> (case-insensitive match on adapter name) pins
//      the adapter; if nothing matches, FAIL CLOSED with the table printed.
//   2. Otherwise pick by capability, in priority order:
//        a. has SHADER_INT64 (u64-REDC path) preferred over u32 limb-split,
//        b. higher max_buffer_size wins (the ~2 GB ANV-class cap matters:
//           pipeline.rs dispatches around it, but bigger is strictly better),
//        c. discrete GPU beats integrated/CPU.
//   3. require_intel_arc (strict mode, OPENVM_WGPU_ARC=1 / --arc) filters to
//      Intel vendor 0x8086 before scoring, same as the old assert.
//   4. allow_u32_only=false (e.g. ntt_test) filters to SHADER_INT64 adapters.
//   5. If no adapter survives the filters: FAIL CLOSED, table printed first.

use wgpu::Adapter;

/// Everything wgpu reports about one adapter that we care about.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver: String,
    pub driver_info: String,
    pub device_type: String,
    pub is_discrete: bool,
    // capability bits relevant to the proving kernels
    pub has_shader_int64: bool,
    pub has_shader_int64_atomic_all_ops: bool,
    pub has_shader_int64_atomic_min_max: bool,
    // limits relevant to the proving kernels
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    /// max_compute_invocations_per_workgroup (the workgroup-size limit)
    pub max_compute_workgroup_size: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_bindings_per_bind_group: u32,
    /// memory-model hint if the wgpu version exposes one (wgpu 30: n/a)
    pub memory_hints: String,
}

/// An enumerated adapter paired with its report (the live handle is kept
/// so `init` can `request_device` on the selected one without re-enumerating).
pub struct Enumerated {
    pub adapter: Adapter,
    pub info: AdapterInfo,
}

fn info_of(adapter: &Adapter) -> AdapterInfo {
    let i = adapter.get_info();
    let l = adapter.limits();
    let f = adapter.features();
    use wgpu::Features as F;
    AdapterInfo {
        name: i.name.clone(),
        backend: format!("{:?}", i.backend),
        vendor_id: i.vendor,
        device_id: i.device,
        driver: i.driver.clone(),
        driver_info: i.driver_info.clone(),
        device_type: format!("{:?}", i.device_type),
        is_discrete: matches!(i.device_type, wgpu::DeviceType::DiscreteGpu),
        has_shader_int64: f.contains(F::SHADER_INT64),
        has_shader_int64_atomic_all_ops: f.contains(F::SHADER_INT64_ATOMIC_ALL_OPS),
        has_shader_int64_atomic_min_max: f.contains(F::SHADER_INT64_ATOMIC_MIN_MAX),
        max_buffer_size: l.max_buffer_size,
        max_storage_buffer_binding_size: l.max_storage_buffer_binding_size,
        max_compute_workgroup_size: l.max_compute_invocations_per_workgroup,
        max_compute_workgroup_storage_size: l.max_compute_workgroup_storage_size,
        max_compute_workgroups_per_dimension: l.max_compute_workgroups_per_dimension,
        max_bindings_per_bind_group: l.max_bindings_per_bind_group,
        // wgpu 30.0.2 exposes no Adapter::memory_hints(); report honestly.
        memory_hints: "n/a (wgpu 30 Adapter exposes no memory_hints)".to_string(),
    }
}

/// Vulkan instance used for enumeration (same descriptor as device.rs).
pub fn vulkan_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

/// Enumerate all Vulkan adapters and keep the live handles.
pub fn enumerate_full() -> Vec<Enumerated> {
    let instance = vulkan_instance();
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    adapters
        .into_iter()
        .map(|a| {
            let info = info_of(&a);
            Enumerated { adapter: a, info }
        })
        .collect()
}

/// Enumerate all Vulkan adapters, report-only (the `--list-adapters` path).
pub fn enumerate() -> Vec<AdapterInfo> {
    enumerate_full().into_iter().map(|e| e.info).collect()
}

/// Capability score, compared lexicographically: higher is better.
/// (int64 path, buffer headroom, discrete). See module docs for the law.
pub fn capability_score(i: &AdapterInfo) -> (u8, u64, u8) {
    (
        i.has_shader_int64 as u8,
        i.max_buffer_size,
        i.is_discrete as u8,
    )
}

/// Print the enumeration table exactly as wgpu reports it.
pub fn print_table(infos: &[AdapterInfo]) {
    println!("=== ADAPTER TABLE (Vulkan, via wgpu 30.0.2 pinned in Cargo.toml) ===");
    println!(
        "{:<3} {:<34} {:<10} {:<10} {:<14} {:<7} {:<9} {:>12} {:>12} {:>6}",
        "IDX", "NAME", "VENDOR", "DEVICE", "TYPE", "I64", "I64ATOM", "BUF_MAX", "STORAGE_B", "WG_SZ"
    );
    for (idx, i) in infos.iter().enumerate() {
        println!(
            "{:<3} {:<34} {:<10} {:<10} {:<14} {:<7} {:<9} {:>12} {:>12} {:>6}",
            idx,
            truncate(&i.name, 34),
            format!("0x{:04x}", i.vendor_id),
            format!("0x{:04x}", i.device_id),
            i.device_type,
            if i.has_shader_int64 { "yes" } else { "no" },
            if i.has_shader_int64_atomic_all_ops {
                "all"
            } else if i.has_shader_int64_atomic_min_max {
                "minmax"
            } else {
                "no"
            },
            fmt_size(i.max_buffer_size),
            fmt_size(i.max_storage_buffer_binding_size),
            i.max_compute_workgroup_size,
        );
    }
    for (idx, i) in infos.iter().enumerate() {
        println!(
            "[{idx}] driver={:?} driver_info={:?} backend={} mem_hints={} wg_smem={} wg_per_dim={} bindings_per_bg={}",
            i.driver, i.driver_info, i.backend, i.memory_hints,
            i.max_compute_workgroup_storage_size, i.max_compute_workgroups_per_dimension,
            i.max_bindings_per_bind_group
        );
    }
    // Honest note: ANV-class drivers commonly cap max_buffer_size near 2 GiB;
    // we print the raw value above, this is only an annotation.
    println!("NOTE: BUF_MAX is the driver-advertised max_buffer_size; ANV-class drivers cap near 2 GiB (pipeline.rs dispatches stay under per-binding caps regardless).");
    println!("=== {} adapter(s) ===", infos.len());
}

/// Enumerate + print + return count (shared by the bins' --list-adapters).
pub fn list_adapters() -> usize {
    let infos = enumerate();
    print_table(&infos);
    infos.len()
}

/// Capability-driven selection over the enumerated infos.
/// Returns the index of the chosen adapter or Err(reason).
/// `require_intel_arc` / `allow_u32_only` have the same meaning as in `InitOpts`.
pub fn select(
    infos: &[AdapterInfo],
    device_substring: Option<&str>,
    require_intel_arc: bool,
    allow_u32_only: bool,
) -> Result<usize, String> {
    if infos.is_empty() {
        return Err("NO VULKAN ADAPTERS ENUMERATED (driver/ICD missing?)".to_string());
    }
    let mut elig: Vec<usize> = (0..infos.len()).collect();
    if require_intel_arc {
        elig.retain(|&k| infos[k].vendor_id == 0x8086);
        if elig.is_empty() {
            return Err("STRICT-ARC MODE: no adapter with vendor 0x8086 (Intel) found".to_string());
        }
    }
    if !allow_u32_only {
        elig.retain(|&k| infos[k].has_shader_int64);
        if elig.is_empty() {
            return Err("STRICT INT64 MODE: no adapter advertises SHADER_INT64 and u32 fallback is not allowed".to_string());
        }
    }
    if let Some(sub) = device_substring {
        let needle = sub.to_lowercase();
        elig.retain(|&k| infos[k].name.to_lowercase().contains(&needle));
        if elig.is_empty() {
            return Err(format!(
                "WGPU_DEVICE={sub:?} matched no adapter (case-insensitive substring on adapter name)"
            ));
        }
    }
    // score descending; stable sort keeps enumeration order on ties
    elig.sort_by(|&a, &b| capability_score(&infos[b]).cmp(&capability_score(&infos[a])));
    Ok(elig[0])
}

/// WGPU_DEVICE env → device_substring opt (empty string treated as unset).
pub fn device_substring_from_env() -> Option<String> {
    std::env::var("WGPU_DEVICE").ok().filter(|s| !s.is_empty())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n - 1).collect();
        t.push('…');
        t
    }
}

fn fmt_size(b: u64) -> String {
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    if b >= GIB && b % GIB == 0 {
        format!("{}Gi", b / GIB)
    } else if b >= MIB && b % MIB == 0 {
        format!("{}Mi", b / MIB)
    } else if b >= GIB {
        format!("{:.2}Gi", b as f64 / GIB as f64)
    } else {
        format!("{}B", b)
    }
}
