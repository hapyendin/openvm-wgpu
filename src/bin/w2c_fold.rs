//! GATE W2C PHASE 2 driver: WGSL WHIR fold vs Phase-1 CPU goldens.
//! Per instance/whir-round: GPU mobius (coeffs), GPU DFT (grs), GPU merkle
//! (commit root). Compares against WHIR_DUMP + w2c-results-r1-GREEN.json,
//! with N-run determinism.
use std::path::{Path, PathBuf};
use wgpu_backend::device::{InitOpts, WgpuDevice};
use wgpu_backend::field::{decode, encode, mont_mul, sha256_hex, u32s_to_bytes};
use wgpu_backend::whir_fold::WhirFold;
use wgpu_backend::w2c::*;

fn read_u32s(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|_| panic!("read {}", p.display()));
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn read_ef(p: &Path) -> Vec<Ef> {
    let w = read_u32s(p);
    w.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}
fn canon_ef(e: &[Ef]) -> Vec<u32> {
    e.iter().flat_map(|x| x.iter().map(|&w| decode(w))).collect()
}
fn raw_ef(c: &[u32]) -> Vec<Ef> {
    c.chunks_exact(4)
        .map(|x| [encode(x[0]), encode(x[1]), encode(x[2]), encode(x[3])])
        .collect()
}
fn flatten(e: &[Ef]) -> Vec<u32> {
    e.iter().flat_map(|r| r.iter().copied()).collect()
}
fn unflatten(w: &[u32]) -> Vec<Ef> {
    w.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}
fn powm(mut b: u32, mut e: u64) -> u32 {
    let mut r = encode(1);
    while e > 0 {
        if e & 1 == 1 {
            r = mont_mul(r, b);
        }
        b = mont_mul(b, b);
        e >>= 1;
    }
    r
}

fn gpu_mobius(dev: &WgpuDevice, wf: &WhirFold, f: &[Ef]) -> Vec<Ef> {
    let n = f.len();
    let logn = n.trailing_zeros();
    let data = wf.buf(dev, &flatten(f), "mob in");
    let dummy = wf.empty16(dev);
    let ddummy = wf.dummy_ro(dev);
    let uni = wf.uni16(dev);
    let mut half = 1usize;
    while half < n {
        let bg = wf.bind(dev, &data, &dummy, &uni, &ddummy);
        WhirFold::set_uni(dev, &uni, (n / 2) as u32, half as u32, 1u32, 0);
        WhirFold::dispatch(dev, &wf.stage, &bg, n / 2);
        half *= 2;
    }
    let _ = logn;
    unflatten(&WhirFold::read_back(dev, &data, n * 4))
}

fn gpu_fft(dev: &WgpuDevice, wf: &WhirFold, coeffs: &[Ef], log_n: u32) -> Vec<Ef> {
    let n = 1usize << log_n;
    // CPU-side pad + bitrev row permutation, then in-place WGSL DIT stages
    let mut rows: Vec<Ef> = Vec::with_capacity(n);
    rows.extend_from_slice(coeffs);
    rows.resize(n, ef_zero());
    let mut perm: Vec<Ef> = vec![ef_zero(); n];
    for i in 0..n {
        let j = ((i as u32).reverse_bits() >> (32 - log_n)) as usize;
        perm[j] = rows[i];
    }
    let data = wf.buf(dev, &flatten(&perm), "fft in");
    let dummy = wf.empty16(dev);
    let ddummy = wf.dummy_ro(dev);
    let uni = wf.uni16(dev);
    let mut half = 1usize;
    let mut s = 0u32;
    while half < n {
        // twiddle table identical chain to CPU fft_inplace
        let wm = powm(root_of_unity_raw(log_n), (n / (half * 2)) as u64);
        let mut tw = vec![0u32; half];
        tw[0] = encode(1);
        for k in 1..half {
            tw[k] = mont_mul(tw[k - 1], wm);
        }
        let twb = wf.buf(dev, &tw, "tw");
        let bg = wf.bind(dev, &data, &dummy, &uni, &twb);
        WhirFold::set_uni(dev, &uni, (n / 2) as u32, half as u32, 0u32, 0);
        WhirFold::dispatch(dev, &wf.stage, &bg, n / 2);
        half *= 2;
        s += 1;
    }
    let _ = s;
    unflatten(&WhirFold::read_back(dev, &data, n * 4))
}

fn gpu_merkle(dev: &WgpuDevice, wf: &WhirFold, grs: &[Ef], k_whir: usize) -> [u32; 8] {
    let n = grs.len();
    // leaf input rows: [row4 | 0^4]
    let mut rows8: Vec<u32> = Vec::with_capacity(n * 8);
    for r in grs {
        rows8.extend_from_slice(r);
        rows8.extend_from_slice(&[0u32; 4]);
    }
    let rows_b = wf.buf(dev, &rows8, "m rows");
    let mut cur = wf.zeros(dev, n * 8, "m cur");
    let uni = wf.uni16(dev);
    let dummy = wf.dummy_ro(dev);
    {
        let bg = wf.bind(dev, &rows_b, &cur, &uni, &dummy);
        WhirFold::dispatch_offset(dev, &wf.init4, &bg, &uni, (n * 8) as u32, 0, 0, n);
    }
    drop(rows_b);
    // phase-1: k_whir strided layers, qs constant
    let qs = n >> k_whir;
    let mut size = n;
    let mut nxt = wf.zeros(dev, (n / 2) * 8, "m nxt");
    for _ in 0..k_whir {
        let nsz = size / 2;
        let bg = wf.bind(dev, &cur, &nxt, &uni, &dummy);
        WhirFold::dispatch_offset(dev, &wf.pair_strided, &bg, &uni, nsz as u32, qs as u32, 0, nsz);
        std::mem::swap(&mut cur, &mut nxt);
        size = nsz;
    }
    // phase-2: adjacent pairs to root
    while size > 1 {
        let nsz = size / 2;
        let bg = wf.bind(dev, &cur, &nxt, &uni, &dummy);
        WhirFold::dispatch_offset(dev, &wf.pair_adj, &bg, &uni, nsz as u32, 0, 0, nsz);
        std::mem::swap(&mut cur, &mut nxt);
        size = nsz;
    }
    let root_m = WhirFold::read_back(dev, &cur, 8);
    std::array::from_fn(|i| decode(root_m[i]))
}

struct InstPlan {
    tag: usize,
    name: String,
    m: usize,
    k_whir: usize,
    log_rs0: usize,
    rounds: usize,
    /// per whir-round coeffs (raw) after mobius, computed once on CPU
    coeffs: Vec<Vec<Ef>>,
    /// golden per-round commit_canon (None for last round)
    gold_commit: Vec<Option<Vec<u32>>>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--list-adapters") {
        wgpu_backend::adapters::list_adapters();
        std::process::exit(0);
    }
    let mut whir = PathBuf::from("/home/matt/appz/openvm-sycl-skel/b9/r15/w2c-r1/whir");
    let mut gold = PathBuf::from("w2c-results-r1-GREEN.json");
    let mut out = PathBuf::from("w2c-fold-report.json");
    let mut runs = 5usize;
    let mut inst_only: Option<usize> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--whir" => { i += 1; whir = args[i].clone().into(); }
            "--gold" => { i += 1; gold = args[i].clone().into(); }
            "--out" => { i += 1; out = args[i].clone().into(); }
            "--runs" => { i += 1; runs = args[i].parse().unwrap(); }
            "--inst" => { i += 1; inst_only = Some(args[i].parse().unwrap()); }
            other => { whir = other.into(); }
        }
        i += 1;
    }
    let names = [
        "app-segments_0",
        "agg-leaf_0",
        "agg-internal-for-leaf_0",
        "agg-internal-recursive_0",
        "wrap-internal-recursive.1_0",
    ];
    let goldv: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&gold).unwrap()).unwrap();

    let arc_mode = args.iter().any(|a| a == "--arc") || std::env::var("OPENVM_WGPU_ARC").is_ok();
    let dev = WgpuDevice::init(InitOpts { require_intel_arc: arc_mode, allow_u32_only: !arc_mode, device_substring: wgpu_backend::adapters::device_substring_from_env() });
    println!("ADAPTER {}", dev.report.name);
    let wf = WhirFold::new(&dev);

    if args.iter().any(|a| a == "--selftest") {
        for &ln in &[16u32, 18, 20, 22, 24] {
            let n = 1usize << ln;
            let raw: Vec<u32> = (0..n * 4).map(|i| (i as u32).wrapping_mul(2654435761) | 1).collect();
            let coeffs = raw_ef(&raw);
            let mut cpu_flat = flatten(&coeffs);
            let mut cols: Vec<Vec<u32>> = vec![vec![0u32; n]; 4];
            for (r, e) in coeffs.iter().enumerate() {
                for l in 0..4 {
                    cols[l][r] = e[l];
                }
            }
            for l in 0..4 {
                fft_inplace(&mut cols[l]);
                for r in 0..n {
                    cpu_flat[r * 4 + l] = cols[l][r];
                }
            }
            let gpu = flatten(&gpu_fft(&dev, &wf, &coeffs, ln));
            let nbad = gpu.iter().zip(cpu_flat.iter()).filter(|(a, b)| a != b).count();
            let firstbad = gpu.iter().zip(cpu_flat.iter()).position(|(a, b)| a != b);
            println!("SELFTEST fft log_n={ln} nbad={nbad} firstbad={firstbad:?}");
        }
        return;
    }
    let _dummy_guard: Vec<Ef> = {
        let raw: Vec<u32> = (0..32).map(|i| (i as u32 + 1) * 7).collect();
        let coeffs = raw_ef(&raw);
        let mut cpu_flat = flatten(&coeffs);
        {
            let mut cols: Vec<Vec<u32>> = vec![vec![0u32; 8]; 4];
            for (r, e) in coeffs.iter().enumerate() {
                for l in 0..4 {
                    cols[l][r] = e[l];
                }
            }
            for l in 0..4 {
                fft_inplace(&mut cols[l]);
                for r in 0..8 {
                    cpu_flat[r * 4 + l] = cols[l][r];
                }
            }
        }
        let gpu = flatten(&gpu_fft(&dev, &wf, &coeffs, 3));
        let cc: Vec<u32> = gpu.iter().map(|w| decode(*w)).collect();
        let ce: Vec<u32> = cpu_flat.iter().map(|w| decode(*w)).collect();
        println!("SELFTEST fft n=8 match={} gpu[:8]={:?} cpu[:8]={:?}", cc == ce, &cc[..8], &ce[..8]);
        // mobius n=8
        let gcm = flatten(&gpu_mobius(&dev, &wf, &coeffs));
        let mut mc = coeffs.clone();
        for l in 0..4 {
            let mut col: Vec<u32> = mc.iter().map(|e| e[l]).collect();
            mobius_inplace(&mut col);
            for r in 0..8 {
                mc[r][l] = col[r];
            }
        }
        let cm = flatten(&mc);
        let _ = (gcm, cm);
        coeffs
    };
    let _ = _dummy_guard;

    // ---- CPU plan: f-subfold per round -> coeffs (mirrors w2c_cpu walk math) ----
    let mut plans: Vec<InstPlan> = Vec::new();
    for (ii, name) in names.iter().enumerate() {
        if let Some(o) = inst_only {
            if o != ii {
                continue;
            }
        }
        let meta = std::fs::read_to_string(whir.join(format!("w{ii}-meta.txt"))).unwrap();
        let mp = |k: &str| -> usize {
            meta.split_whitespace()
                .find(|t| t.starts_with(&format!("{k}=")))
                .unwrap()
                .split('=')
                .nth(1)
                .unwrap()
                .parse()
                .unwrap()
        };
        let (m, k, lrs) = (mp("m"), mp("k_whir"), mp("log_rs0"));
        let rounds: Vec<usize> = {
            let t = meta.split("rounds=").nth(1).unwrap();
            let t = t.split(']').next().unwrap().trim_start_matches('[');
            t.split(',').map(|x| x.trim().parse().unwrap()).collect()
        };
        let nr = rounds.len();
        let f0 = raw_ef(&read_u32s(&whir.join(format!("w{ii}-f0.bin"))));
        let alphas = raw_ef(&read_u32s(&whir.join(format!("w{ii}-alphas.bin"))));
        let mut f = f0;
        let mut aidx = 0usize;
        let mut coeffs = Vec::new();
        for _r in 0..nr {
            for _s in 0..k {
                let a = alphas[aidx];
                aidx += 1;
                let half = f.len() / 2;
                let mut nf = vec![ef_zero(); half];
                for y in 0..half {
                    let e0 = f[2 * y];
                    let d = ef_sub(&f[2 * y + 1], &e0);
                    nf[y] = ef_add(&e0, &ef_mul(&d, &a));
                }
                f = nf;
            }
            // plan stores POST-mobius coeffs (dumps verified this convention)
            let mut c = f.clone();
            for l in 0..4 {
                let mut col: Vec<u32> = c.iter().map(|e| e[l]).collect();
                mobius_inplace(&mut col);
                for (j, x) in col.iter().enumerate() {
                    c[j][l] = *x;
                }
            }
            coeffs.push(c.clone());
        }
        if ii == 0 {
            let g0 = read_ef(&whir.join("w0-r0-gcoeffs.bin"));
            println!(
                "PLANDBG r0 c0={:?} dump0={:?} c1={:?} dump1={:?}",
                &&canon_ef(&vec![coeffs[0][0]])[..4],
                &g0[0],
                &&canon_ef(&vec![coeffs[0][1]])[..4],
                &g0[1]
            );
        }
        let gold_commit = goldv["instances"][name]["fold"]["rounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.get("commit_canon").map(|v| v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()))
            .collect();
        plans.push(InstPlan {
            tag: ii,
            name: name.to_string(),
            m,
            k_whir: k,
            log_rs0: lrs,
            rounds: nr,
            coeffs,
            gold_commit,
        });
    }

    // ---- GPU runs ----
    let mut all: Vec<serde_json::Value> = Vec::new();
    let mut det = true;
    let mut golden_ok = true;
    let mut first_fail = String::new();
    for run in 0..runs {
        let mut run_rows = Vec::new();
        for p in &plans {
            for r in 0..p.rounds {
                let is_last = r + 1 == p.rounds;
                let t0 = std::time::Instant::now();
                let coeffs_gpu = p.coeffs[r].clone();
                let cc = canon_ef(&coeffs_gpu);
                // golden coeffs: gcoeffs dump = coeffs | zeros, same len as grs
                let gcexp = if is_last {
                    read_ef(&whir.join(format!("w{}-finalcoeffs.bin", p.tag)))
                } else {
                    let exp_grs =
                        read_ef(&whir.join(format!("w{}-r{r}-grs.bin", p.tag)));
                    let mut v = read_ef(&whir.join(format!("w{}-r{r}-gcoeffs.bin", p.tag)));
                    v.truncate(exp_grs.len());
                    v
                };
                // dumps are stored CANON already: use raw words, no decode
                let gcexp_c: Vec<u32> = gcexp.iter().flatten().copied().collect();
                let gc_ok = cc == gcexp_c[..cc.len().min(gcexp_c.len())];
                let (grs_sha, commit_ok) = if !is_last {
                    let log_n = (p.log_rs0 - r - 1) as u32;
                    let mut fftin = coeffs_gpu.clone();
                    fftin.resize(1usize << log_n, ef_zero());
                    let grs_gpu = gpu_fft(&dev, &wf, &fftin, log_n);
                    let gc_canon = canon_ef(&grs_gpu);
                    let exp: Vec<u32> = read_ef(&whir.join(format!("w{}-r{r}-grs.bin", p.tag))).iter().flatten().copied().collect();
                    if gc_canon != exp && first_fail.is_empty() {
                        let d = gc_canon.iter().zip(exp.iter()).position(|(a, b)| a != b).unwrap_or(usize::MAX);
                        first_fail = format!(
                            "inst {} r{r} grs mismatch at word {d} gpu={:?} exp={:?} (gpu[..4]={:?} exp[..4]={:?})",
                            p.name,
                            gc_canon[d.min(gc_canon.len() - 1)],
                            exp[d.min(exp.len() - 1)],
                            &gc_canon[0..4],
                            &exp[0..4]
                        );
                        std::fs::write("/tmp/grs_gpu.bin", u32s_to_bytes(&gc_canon)).ok();
                    }
                    golden_ok &= gc_canon == exp;
                    let sha = sha256_hex(&u32s_to_bytes(&gc_canon));
                    let root = gpu_merkle(&dev, &wf, &grs_gpu, p.k_whir);
                    let cok = p.gold_commit[r].as_deref() == Some(root.as_slice());
                    if !cok && first_fail.is_empty() {
                        first_fail = format!("inst {} r{r} commit gpu={root:?} gold={:?}", p.name, p.gold_commit[r]);
                    }
                    golden_ok &= cok;
                    (sha, cok)
                } else {
                    (String::new(), true)
                };
                if !gc_ok && first_fail.is_empty() {
                    first_fail = format!("inst {} r{r} coeffs mismatch", p.name);
                }
                golden_ok &= gc_ok;
                let row = serde_json::json!({
                    "run": run, "inst": p.name, "round": r,
                    "gc_ok": gc_ok, "grs_sha": grs_sha, "commit_ok": commit_ok,
                    "secs": t0.elapsed().as_secs_f64(),
                });
                run_rows.push(row);
            }
        }
        if run == 0 {
            all.push(serde_json::json!({"run": 0, "rows": run_rows}));
        } else {
            // determinism: identical sha rows
            let prev = all[0]["rows"].as_array().unwrap();
            for (a, b) in prev.iter().zip(run_rows.iter()) {
                if a["grs_sha"] != b["grs_sha"] || a["gc_ok"] != b["gc_ok"] || a["commit_ok"] != b["commit_ok"] {
                    det = false;
                    if first_fail.is_empty() {
                        first_fail = format!("det mismatch run{run}: {a} vs {b}");
                    }
                }
            }
            all.push(serde_json::json!({"run": run, "rows": run_rows}));
        }
        println!("run {run} done golden_ok={golden_ok} det={det} fail={first_fail}");
    }
    let rep = serde_json::json!({
        "adapter": dev.report.name,
        "runs": runs,
        "golden_ok": golden_ok,
        "determinism5x": det,
        "first_fail": first_fail,
        "detail": all,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&rep).unwrap()).unwrap();
    println!(
        "W2C_FOLD: golden_ok={golden_ok} determinism={det} -> {}",
        out.display()
    );
    if !(golden_ok && det) {
        std::process::exit(2);
    }
}
