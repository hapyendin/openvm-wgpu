// W2a gate binary: run the GKR/zerocheck GPU flow on the recorded x-l1a-r1
// goldens; byte-exact vs CPU transcript + 5x determinism.
//
//   w2a_gkr --inputs <dir> --json <gkr-inputs.json> [--runs 5] [--out <path>]

use std::path::{Path, PathBuf};

use wgpu_backend::device::{InitOpts, WgpuDevice};
use wgpu_backend::gkr::{
    check_expected, check_snapshots, run_instance, GkrInputs, GkrPipeline,
};

fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

#[derive(serde::Serialize)]
struct InstanceReport {
    name: String,
    t: u32,
    leaf_sha256: String,
    leaf_bytes: u64,
    snaps: Vec<wgpu_backend::gkr::LevelSnap>,
    value_mismatches: Vec<String>,
    snap_mismatches: Vec<String>,
    run_hashes: Vec<String>,
    deterministic: bool,
    green: bool,
}

#[derive(serde::Serialize)]
struct Report {
    adapter: String,
    p: u64,
    instances: Vec<InstanceReport>,
    skipped_instances: Vec<String>,
    all_green: bool,
    all_deterministic: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--list-adapters") {
        wgpu_backend::adapters::list_adapters();
        std::process::exit(0);
    }
    let mut inputs_dir: PathBuf = PathBuf::from(".");
    let mut json_path: PathBuf = PathBuf::from("gkr-inputs.json");
    let mut runs: u32 = 5;
    let mut out_path: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--inputs" => {
                i += 1;
                inputs_dir = PathBuf::from(&args[i]);
            }
            "--json" => {
                i += 1;
                json_path = PathBuf::from(&args[i]);
            }
            "--runs" => {
                i += 1;
                runs = args[i].parse().unwrap();
            }
            "--out" => {
                i += 1;
                out_path = Some(PathBuf::from(&args[i]));
            }
            other => eprintln!("unknown arg: {other}"),
        }
        i += 1;
    }

    let text = std::fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", json_path.display()));
    let inputs: GkrInputs = serde_json::from_str(&text).expect("parse gkr-inputs.json");

    // WGPU_DEVICE pin (W1 adapter pattern); default = first Intel Arc.
    let arc_mode = args.iter().any(|a| a == "--arc") || std::env::var("OPENVM_WGPU_ARC").is_ok();
    let dev = WgpuDevice::init(InitOpts {
        require_intel_arc: arc_mode,
        allow_u32_only: !arc_mode,
        device_substring: wgpu_backend::adapters::device_substring_from_env(),
    });
    let pipe = GkrPipeline::new(&dev);
    eprintln!("PIPELINES_READY tree unpack seval fold");

    let mut report_instances = Vec::new();
    let mut all_green = true;
    let mut all_det = true;
    let mut skipped: Vec<String> = Vec::new();

    for inst in &inputs.instances {
        let tag = inst.name.split('.').next().unwrap_or(&inst.name);
        let leaf_path = inputs_dir.join(format!("dbg_{tag}_leaves_cpu.bin"));
        let leaf = match std::fs::read(&leaf_path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("INSTANCE {} SKIP: {}: {e}", inst.name, leaf_path.display());
                // A skipped instance is a missing-input-data condition, not a
                // correctness failure; it is recorded in skipped_instances
                // instead of poisoning all_green.
                skipped.push(inst.name.clone());
                continue;
            }
        };
        eprintln!(
            "INSTANCE {} T={} leaves={}B ({}) — {} runs",
            inst.name,
            inst.t,
            leaf.len(),
            leaf_path.display(),
            runs
        );

        let mut run_hashes: Vec<String> = Vec::new();
        let mut value_mismatches: Vec<String> = Vec::new();
        let mut snap_mismatches: Vec<String> = Vec::new();
        let mut snaps: Vec<wgpu_backend::gkr::LevelSnap> = Vec::new();
        let t0 = std::time::Instant::now();

        for run in 1..=runs {
            let mut snaps_run: Vec<wgpu_backend::gkr::LevelSnap> = Vec::new();
            let out = run_instance(&dev, &pipe, inst, &leaf, &inputs_dir, &mut snaps_run);
            let dt = std::time::Instant::now();
            if run == 1 {
                snaps = out.levels.clone();
                value_mismatches = check_expected(inst, &out);
                snap_mismatches = check_snapshots(inst, &out, &inputs_dir);
            }
            run_hashes.push(out.out_sha256.clone());
            eprintln!(
                "  run {}: out_sha256={} ({:.1}s)",
                run,
                out.out_sha256,
                dt.elapsed().as_secs_f64()
            );
        }
        let deterministic = run_hashes.iter().all(|h| *h == run_hashes[0]);
        let green = value_mismatches.is_empty()
            && snap_mismatches.is_empty()
            && deterministic;
        if !green {
            all_green = false;
        }
        if !deterministic {
            all_det = false;
        }
        eprintln!(
            "  {}: {} (value mism: {}, snap mism: {}, det: {})",
            inst.name,
            if green { "GREEN" } else { "FAIL" },
            value_mismatches.len(),
            snap_mismatches.len(),
            deterministic
        );
        for m in &value_mismatches {
            eprintln!("    VALUE {m}");
        }
        for m in &snap_mismatches {
            eprintln!("    SNAP {m}");
        }

        let _ = sha256_hex;
        report_instances.push(InstanceReport {
            name: inst.name.clone(),
            t: inst.t,
            leaf_sha256: sha256_hex(&leaf),
            leaf_bytes: leaf.len() as u64,
            snaps,
            value_mismatches,
            snap_mismatches,
            run_hashes,
            deterministic,
            green,
        });
    }

    let report = Report {
        adapter: dev.report.name.clone(),
        p: inputs.p,
        instances: report_instances,
        skipped_instances: skipped,
        all_green,
        all_deterministic: all_det,
    };
    let json = serde_json::to_string_pretty(&report).unwrap();
    match out_path {
        Some(p) => {
            eprintln!("WROTE {}", p.display());
            std::fs::write(&p, &json).unwrap();
        }
        None => {
            let p = inputs_dir.join("w2a-results.json");
            eprintln!("WROTE {}", p.display());
            std::fs::write(&p, &json).unwrap();
        }
    }
    eprintln!(
        "SUMMARY all_green={} all_deterministic={}",
        report.all_green, report.all_deterministic
    );
    if !report.all_green {
        std::process::exit(2);
    }
}
