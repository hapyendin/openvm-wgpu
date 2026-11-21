//! GATE W2C step 3: WGSL grind vs 90-site CPU goldens.
//! Per site: GPU scans w = 0..limit (one perm per invocation, w == gid.x),
//! winner = lowest w passing the low-bits mask (atomicMin; lowest global
//! linear invocation id tie-break is total since w == gid.x). Then a 1-thread
//! kernel recomputes the winner's check word. CPU recomputes the same check
//! via w2c::Duplex::grind_check on the dumped base state. Runs R times,
//! requires bit-identical (site, wmin, check) vectors every run.
use std::path::PathBuf;
use wgpu_backend::device::{InitOpts, WgpuDevice};
use wgpu_backend::field::{constants_buffer_mont, sha256_hex, u32s_to_bytes, Poseidon2};
use wgpu_backend::w2c::{mask_ok, Duplex};
use wgpu_backend::whir_fold::WhirFold;
use wgpu_backend::wgsl;

const GRIND_WGSL: &str = r#"
@group(0) @binding(4) var<storage, read_write> amin: array<atomic<u32>>;

@compute @workgroup_size(256u)
fn grind_scan(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  if (w >= uni.a2) { return; }
  var cells: array<u32, 16u>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) { cells[i] = io1[i]; }
  cells[uni.seed] = encw(w);
  perm_mix(&cells);
  let c = decode(cells[7u]);
  if ((c & uni.a1) == 0u) { atomicMin(&amin[0], w); }
}

@compute @workgroup_size(1u)
fn grind_check(@builtin(global_invocation_id) gid: vec3<u32>) {
  var cells: array<u32, 16u>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) { cells[i] = io1[i]; }
  cells[uni.seed] = encw(uni.a2);
  perm_mix(&cells);
  io2[0] = decode(cells[7u]);
}
"#;

struct Site {
    inst: u32,
    site: u32,
    bits: u32,
    idx: u32,
    w_rec: u32,
    check_rec: u32,
    base: [u32; 16],
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut bases = PathBuf::from("w2c-grind-bases.bin");
    let mut sites_f = PathBuf::from("w2c/pow_sites_r1.jsonl");
    let mut gold = PathBuf::from("w2c-results-r1-GREEN.json");
    let mut out = PathBuf::from("w2c-grind-report.json");
    let mut runs = 5usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bases" => { i += 1; bases = args[i].clone().into(); }
            "--sites" => { i += 1; sites_f = args[i].clone().into(); }
            "--gold" => { i += 1; gold = args[i].clone().into(); }
            "--out" => { i += 1; out = args[i].clone().into(); }
            "--runs" => { i += 1; runs = args[i].parse().unwrap(); }
            _ => {}
        }
        i += 1;
    }
    let bw: Vec<u32> = std::fs::read(&bases)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(bw.len() % 22, 0, "bases file must be 22 words/site");
    let mut sites: Vec<Site> = Vec::new();
    for r in bw.chunks_exact(22) {
        sites.push(Site {
            inst: r[0], site: r[1], bits: r[2], idx: r[3],
            w_rec: r[4], check_rec: r[5],
            base: r[6..22].try_into().unwrap(),
        });
    }
    // cross-check against recorded table
    let tbl: Vec<serde_json::Value> = std::fs::read_to_string(&sites_f)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(tbl.len(), sites.len(), "site count");
    let mut cpu_min_rec_true = 0usize;
    let goldv: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&gold).unwrap()).unwrap();
    for (_k, instv) in goldv["instances"].as_object().unwrap() {
        if let Some(rows) = instv["grind"].as_array() {
            cpu_min_rec_true += rows.iter().filter(|r| r["min_equals_recorded"] == true).count();
        }
    }
    for (j, s) in sites.iter().enumerate() {
        let t = &tbl[j];
        assert_eq!(t["site"].as_u64().unwrap() as u32, s.site, "site id");
        assert_eq!(t["witness"].as_u64().unwrap() as u32, s.w_rec, "w_rec");
        assert_eq!(t["bits"].as_u64().unwrap() as u32, s.bits, "bits");
    }

    let dev = WgpuDevice::init(InitOpts { require_intel_arc: true, allow_u32_only: false });
    println!("ADAPTER {}", dev.report.name);
    let wf = WhirFold::new(&dev);
    let src = format!("{}{}\n{}", wgsl::mul_prelude(dev.has_shader_int64), wgsl::COMMON, GRIND_WGSL);
    let module = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grind"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::COMPUTE, count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::COMPUTE, count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::COMPUTE, count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3, visibility: wgpu::ShaderStages::COMPUTE, count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4, visibility: wgpu::ShaderStages::COMPUTE, count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
            },
        ],
    });
    let layout = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
    });
    let mk = |name: &str| dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(name),
        compilation_options: Default::default(),
        cache: None,
    });
    let p_scan = mk("grind_scan");
    let p_check = mk("grind_check");

    let kc = wf.buf(&dev, &constants_buffer_mont(), "kc");
    let io1 = wf.buf(&dev, &[0u32; 16], "bases");
    let io2 = wf.buf(&dev, &[0u32; 8], "check");
    let amin = wf.buf(&dev, &[0xFFFFFFFFu32; 4], "amin");
    let uni = wf.uni16(&dev);
    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: kc.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: io1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: io2.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: amin.as_entire_binding() },
        ],
    });

    let p2 = Poseidon2::new();
    let mut run_hashes: Vec<String> = Vec::new();
    let mut table: Vec<serde_json::Value> = Vec::new();
    let mut all_ok = true;
    let mut det = true;
    for run in 0..runs {
        let mut rows: Vec<serde_json::Value> = Vec::new();
        for s in &sites {
            let limit = ((s.w_rec + 1).next_power_of_two() << 1).max(256);
            let mask = (1u64 << s.bits) - 1;
            dev.queue.write_buffer(&io1, 0, &u32s_to_bytes(&s.base));
            dev.queue.write_buffer(&amin, 0, &u32s_to_bytes(&[0xFFFFFFFFu32; 4]));
            WhirFold::set_uni(&dev, &uni, s.idx, mask as u32, limit, 0);
            WhirFold::dispatch(&dev, &p_scan, &bg, limit as usize);
            let wmin = WhirFold::read_back(&dev, &amin, 1)[0];
            WhirFold::set_uni(&dev, &uni, s.idx, mask as u32, wmin, 0);
            WhirFold::dispatch(&dev, &p_check, &bg, 1);
            let check_gpu = WhirFold::read_back(&dev, &io2, 1)[0];
            let check_cpu = Duplex::grind_check(&p2, &s.base, s.idx as usize, wmin);
            let ok = wmin == s.w_rec
                && check_gpu == s.check_rec
                && check_cpu == s.check_rec
                && mask_ok(check_gpu, s.bits as usize);
            all_ok &= ok;
            rows.push(serde_json::json!({
                "site": s.site, "inst": s.inst, "bits": s.bits,
                "w_rec": s.w_rec, "w_min_gpu": wmin,
                "check_rec": s.check_rec, "check_gpu": check_gpu, "check_cpu": check_cpu,
                "mask_ok": mask_ok(check_gpu, s.bits as usize), "ok": ok,
            }));
        }
        let canon = serde_json::to_string(&rows).unwrap();
        let h = sha256_hex(canon.as_bytes());
        if run > 0 && h != run_hashes[0] {
            det = false;
        }
        println!("run {run} ok={} hash={}", all_ok, &h[..16]);
        run_hashes.push(h);
        if run == 0 {
            table = rows;
        }
    }
    let rep = serde_json::json!({
        "adapter": dev.report.name,
        "runs": runs,
        "sites": sites.len(),
        "grind_gpu_ok": all_ok,
        "determinism": det,
        "run_hashes": run_hashes,
        "cpu_min_equals_recorded_count": cpu_min_rec_true,
        "table": table,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&rep).unwrap()).unwrap();
    println!("W2C_GRIND: ok={} det={} sites={} -> {}", all_ok, det, sites.len(), out.display());
    if !(all_ok && det) {
        std::process::exit(2);
    }
}
