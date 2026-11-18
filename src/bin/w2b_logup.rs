// W2b phase-2 gate binary: WGSL logup leaf build + logup zerocheck GKR rounds
// on the cpu9 world; byte-exact vs the phase-1 CPU goldens (streams) + 5x
// determinism. Reuses the proven W2a round machinery (gkr.rs / gkr_wgsl.rs).
//
//   w2b_logup --inputs <w2b-inputs dir> --traces <dump-cpu9 trace dir>
//             --snapdir <dir with dbg_*_cpu.bin> [--runs 5] [--out PATH]
//             [--leafonly]

use std::path::{Path, PathBuf};

use sha2::Digest;

use wgpu_backend::device::{InitOpts, WgpuDevice};
use wgpu_backend::gkr::{check_expected, check_snapshots, run_instance_gpu, GkrInputs, GkrPipeline};
use wgpu_backend::logup_leaf::{build_leaf, LeafPipeline, LeafWorld};

fn sha256_hex(data: &[u8]) -> String {
    let mut h = sha2::Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

#[derive(serde::Deserialize)]
struct LayerShas {
    layers: Vec<LayerSha>,
    concat_sha256: String,
    leaf_sha256: String,
    n_goldens: u32,
}
#[derive(serde::Deserialize)]
struct LayerSha {
    layer: u32,
    n: u32,
    bytes: u32,
    sha: String,
}

#[derive(serde::Serialize)]
struct LayerRow {
    layer: u32,
    n: u32,
    bytes: u32,
    sha_gpu: String,
    sha_expected: String,
    match_: bool,
}

#[derive(serde::Serialize)]
struct RunReport {
    leaf_sha256: String,
    out_sha256: String,
    secs: f64,
}

#[derive(serde::Serialize)]
struct Report {
    adapter: String,
    world: String,
    t: u32,
    n_goldens: u32,
    leaf_expected_sha256: String,
    leaf_match: bool,
    layers: Vec<LayerRow>,
    layers_all_match: bool,
    concat_expected: String,
    concat_gpu: String,
    concat_match: bool,
    value_mismatches: Vec<String>,
    snap_mismatches: Vec<String>,
    runs: Vec<RunReport>,
    deterministic: bool,
    green: bool,
    root_p: Vec<u32>,
    root_q: Vec<u32>,
}

fn quads_bytes(groups: &[&[u32]]) -> Vec<u8> {
    let mut v = Vec::new();
    for g in groups {
        assert_eq!(g.len() % 4, 0);
        for w in *g {
            v.extend_from_slice(&w.to_le_bytes());
        }
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--list-adapters") {
        wgpu_backend::adapters::list_adapters();
        std::process::exit(0);
    }
    let mut inputs_dir = PathBuf::from(".");
    let mut traces_dir = PathBuf::from(".");
    let mut snap_dir: Option<PathBuf> = None;
    let mut json_path: Option<PathBuf> = None;
    let mut runs: u32 = 5;
    let mut out_path: Option<PathBuf> = None;
    let mut leafonly = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--inputs" => { i += 1; inputs_dir = PathBuf::from(&args[i]); }
            "--traces" => { i += 1; traces_dir = PathBuf::from(&args[i]); }
            "--snapdir" => { i += 1; snap_dir = Some(PathBuf::from(&args[i])); }
            "--json" => { i += 1; json_path = Some(PathBuf::from(&args[i])); }
            "--runs" => { i += 1; runs = args[i].parse().unwrap(); }
            "--out" => { i += 1; out_path = Some(PathBuf::from(&args[i])); }
            "--leafonly" => leafonly = true,
            other => eprintln!("unknown arg: {other}"),
        }
        i += 1;
    }
    let json_path = json_path.unwrap_or_else(|| inputs_dir.join("gkr-inputs-w2b.json"));
    let shas: LayerShas = {
        let t = std::fs::read_to_string(inputs_dir.join("layer-shas.json")).unwrap();
        serde_json::from_str(&t).unwrap()
    };
    let world: LeafWorld = {
        let t = std::fs::read_to_string(inputs_dir.join("leafworld.json")).unwrap();
        serde_json::from_str(&t).unwrap()
    };
    let inputs: GkrInputs = {
        let t = std::fs::read_to_string(&json_path).unwrap();
        serde_json::from_str(&t).unwrap()
    };
    assert_eq!(inputs.instances.len(), 1);
    let inst = &inputs.instances[0];
    let snap_dir = snap_dir.unwrap_or(inputs_dir.clone());

    let arc_mode = args.iter().any(|a| a == "--arc") || std::env::var("OPENVM_WGPU_ARC").is_ok();
    let dev = WgpuDevice::init(InitOpts { require_intel_arc: arc_mode, allow_u32_only: !arc_mode, device_substring: wgpu_backend::adapters::device_substring_from_env() });
    let pipe = GkrPipeline::new(&dev);
    let lp = LeafPipeline::new(&dev);
    eprintln!("PIPELINES_READY gkr(4) logup(3)");

    let mut cache = wgpu_backend::logup_leaf::TraceCache::new();
    let mut run_reports: Vec<RunReport> = Vec::new();
    let mut value_mismatches: Vec<String> = Vec::new();
    let mut snap_mismatches: Vec<String> = Vec::new();
    let mut layers: Vec<LayerRow> = Vec::new();
    let mut concat_gpu = String::new();
    let mut leaf_match = false;
    let mut root_p = Vec::new();
    let mut root_q = Vec::new();

    for run in 1..=runs {
        let t0 = std::time::Instant::now();
        let (leaf_bufs, leaf_sha) =
            build_leaf(&dev, &lp, &world, &inputs_dir, &traces_dir, &mut cache, 1 << 29);
        eprintln!("run {run}: leaf built sha={} ({:.1}s)", leaf_sha, t0.elapsed().as_secs_f64());
        if leaf_sha != shas.leaf_sha256 {
            eprintln!("LEAF SHA MISMATCH gpu={} expected={}", leaf_sha, shas.leaf_sha256);
        }
        if leafonly {
            run_reports.push(RunReport {
                leaf_sha256: leaf_sha,
                out_sha256: String::new(),
                secs: t0.elapsed().as_secs_f64(),
            });
            continue;
        }
        let mut snaps = Vec::new();
        let out = run_instance_gpu(
            &dev, &pipe, inst, leaf_bufs, leaf_sha.clone(), &snap_dir, &mut snaps,
        );
        let dt = t0.elapsed().as_secs_f64();
        eprintln!("run {run}: out_sha256={} ({:.1}s)", out.out_sha256, dt);
        if run == 1 {
            leaf_match = leaf_sha == shas.leaf_sha256;
            value_mismatches = check_expected(inst, &out);
            snap_mismatches = check_snapshots(inst, &out, &snap_dir);
            root_p = out.root_p.clone();
            root_q = out.root_q.clone();
            // per-layer golden-stream sha table (phase-1 layout)
            let mut concat = sha2::Sha256::new();
            for l in 0..=(inst.t - 1) {
                let groups: Vec<&[u32]> = if l == 0 {
                    vec![&out.root_q, &out.claims0]
                } else {
                    let r = &out.rounds[l as usize - 1];
                    let mut g: Vec<&[u32]> = Vec::with_capacity(r.subs.len() + 1);
                    for s in &r.subs {
                        g.push(s.as_slice());
                    }
                    g.push(r.claims.as_slice());
                    g
                };
                let bytes = quads_bytes(&groups);
                let n = (bytes.len() / 16) as u32;
                let sha = sha256_hex(&bytes);
                concat.update(&bytes);
                let exp = &shas.layers[l as usize];
                assert_eq!(exp.layer as u32, l);
                assert_eq!(exp.n, n, "layer {l} golden count");
                assert_eq!(exp.bytes as usize, bytes.len());
                layers.push(LayerRow {
                    layer: l,
                    n,
                    bytes: bytes.len() as u32,
                    sha_gpu: sha[..16].to_string(),
                    sha_expected: exp.sha.clone(),
                    match_: sha[..16] == exp.sha,
                });
            }
            concat_gpu = format!("{:x}", concat.finalize());
        }
        run_reports.push(RunReport {
            leaf_sha256: leaf_sha,
            out_sha256: out.out_sha256.clone(),
            secs: dt,
        });
    }

    let deterministic = run_reports.iter().all(|r| {
        r.out_sha256 == run_reports[0].out_sha256 && r.leaf_sha256 == run_reports[0].leaf_sha256
    });
    let layers_all_match = layers.iter().all(|r| r.match_);
    let concat_match = concat_gpu == shas.concat_sha256;
    let green = leaf_match
        && layers_all_match
        && concat_match
        && value_mismatches.is_empty()
        && snap_mismatches.is_empty()
        && deterministic;
    let report = Report {
        adapter: dev.report.name.clone(),
        world: "cpu9 (dump-cpu9 traces + cpu9-values.bin)".to_string(),
        t: inst.t,
        n_goldens: shas.n_goldens,
        leaf_expected_sha256: shas.leaf_sha256.clone(),
        leaf_match,
        layers,
        layers_all_match,
        concat_expected: shas.concat_sha256.clone(),
        concat_gpu,
        concat_match,
        value_mismatches,
        snap_mismatches,
        runs: run_reports,
        deterministic,
        green,
        root_p,
        root_q,
    };
    let js = serde_json::to_string_pretty(&report).unwrap();
    let out_path = out_path.unwrap_or_else(|| inputs_dir.join("w2b-results.json"));
    std::fs::write(&out_path, &js).unwrap();
    eprintln!("WROTE {}", out_path.display());
    eprintln!(
        "SUMMARY green={} leaf={} layers={} concat={} det={} valmism={} snapmism={}",
        green, leaf_match, layers_all_match, concat_match, deterministic,
        report.value_mismatches.len(), report.snap_mismatches.len(),
    );
    for m in &report.value_mismatches {
        eprintln!("  VALUE {m}");
    }
    for m in &report.snap_mismatches {
        eprintln!("  SNAP {m}");
    }
    if !green {
        std::process::exit(2);
    }
    let _ = Path::new(".").join(""); // keep `Path` import used across cfg
}
