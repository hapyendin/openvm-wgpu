// Phase-2 self-audit probe: under OPENVM_WGPU_WEBGPU_LIMITS=1, wgpu MUST
// reject (a) a buffer > max_buffer_size and (b) a bind > max_storage_-
// Buffer_binding_size. If either silently succeeds, our limits mode is a
// no-op and the ladder green would be meaningless. Runs with catch_unwind
// so all three checks report in one process.
use std::panic;
use wgpu_backend::device::{InitOpts, WgpuDevice};

fn mk(dev: &WgpuDevice, size: u64) -> wgpu::Buffer {
    dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe"),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

/// Returns true if wgpu rejected the op (panic at create OR device error at
/// poll). Either signal is a valid validation rejection in wgpu 30.
fn try_run(label: &str, f: impl FnOnce() -> bool) -> bool {
    let rejected = match panic::catch_unwind(panic::AssertUnwindSafe(f)) {
        Ok(poll_err) => poll_err, // closure returned poll().is_err()
        Err(_) => true,           // panicked at create
    };
    println!("PROBE {label}: rejected={rejected}");
    rejected
}

fn main() {
    let dev = WgpuDevice::init(InitOpts {
        require_intel_arc: true,
        allow_u32_only: true,
    });
    let caps = dev.device.limits();
    println!(
        "device binding_cap={} buf_cap={}",
        caps.max_storage_buffer_binding_size, caps.max_buffer_size
    );
    let over_buf = caps.max_buffer_size + (64 << 20);
    let over_bind = caps.max_storage_buffer_binding_size + (64 << 20);
    let ok_bind = 100u64 << 20;
    // buffer cap check (validation error surfaces at create on the
    // device's error handler; wgpu panics the thread via device error)
    let p1 = try_run("big_buffer_over_cap", {
        let dev2 = &dev;
        move || {
            let _b = mk(dev2, over_buf);
            dev2.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).is_err()
        }
    });
    // oversize bind group
    let p2 = try_run("bind_over_cap", {
        let dev2 = &dev;
        move || {
            let b = mk(dev2, over_bind.next_power_of_two());
            let bgl = dev2.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            let _bg = dev2.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &b,
                        offset: 0,
                        size: std::num::NonZeroU64::new(over_bind),
                    }),
                }],
            });
            dev2.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).is_err()
        }
    });
    // control: legal bind must NOT panic
    let p3 = try_run("legal_bind", {
        let dev2 = &dev;
        move || {
            let b = mk(dev2, ok_bind);
            let bgl = dev2.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            let _bg = dev2.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &b,
                        offset: 0,
                        size: std::num::NonZeroU64::new(ok_bind),
                    }),
                }],
            });
            dev2.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).is_err()
        }
    });
    let wl = std::env::var("OPENVM_WGPU_WEBGPU_LIMITS").as_deref() == Ok("1");
    let verdict = if wl {
        p1 && p2 && !p3
    } else {
        // default device (2 GiB binds): big_buffer_over_cap may still panic
        // (adapter cap 4 GiB+ => p1 false), bind_over_cap must NOT panic
        !p2 && !p3
    };
    println!(
        "PROBE_VERDICT {} (limits_mode={wl})",
        if verdict { "PASS" } else { "FAIL" }
    );
    if !verdict {
        std::process::exit(1);
    }
}
