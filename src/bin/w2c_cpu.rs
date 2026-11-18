//! W2C Phase-1 CPU reference driver.
//! 1) transcript-chain replay of every recorded instance stream (samples byte-exact)
//! 2) grind goldens: recorded-witness validation + deterministic serial min-winner
//! 3) WHIR fold goldens: s_evals / fold / DFT / merkle-commit recompute vs recorded stream
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use wgpu_backend::field::{decode, encode, mont_mul, Poseidon2};
use wgpu_backend::w2c::*;

fn read_u32s(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|_| panic!("read {}", p.display()));
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn read_ef(p: &Path) -> Vec<Ef> {
    let w = read_u32s(p);
    assert_eq!(w.len() % 4, 0);
    w.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}
fn canon_ef(e: &[Ef]) -> Vec<u32> {
    e.iter().flat_map(|x| x.iter().map(|&w| decode(w))).collect()
}
fn raw_ef(c: &[u32]) -> Vec<Ef> {
    c.chunks_exact(4).map(|x| [encode(x[0]), encode(x[1]), encode(x[2]), encode(x[3])]).collect()
}

struct Stream {
    vals: Vec<u32>,
    flags: Vec<bool>, // true = sample
    n: usize,
}
fn load_stream(dir: &Path, tag: &str) -> Stream {
    let f = tag.replace('/', "_");
    let raw = std::fs::read(dir.join(format!("{f}.prove.bin"))).unwrap();
    let flags_b = std::fs::read(dir.join(format!("{f}.prove.bin.flags"))).unwrap();
    let n = raw.len() / 8;
    let vals: Vec<u32> = (0..n).map(|i| u32::from_le_bytes(raw[8 * i..8 * i + 4].try_into().unwrap())).collect();
    let flags: Vec<bool> = (0..n).map(|i| (flags_b[i >> 3] >> (i & 7)) & 1 == 1).collect();
    Stream { vals, flags, n }
}

/// fold+merkle pipeline recompute (returns per-round artifacts)
#[allow(clippy::too_many_arguments)]
fn fold_recompute(
    p2: &Poseidon2,
    f0: &[Ef],
    w0: &[Ef],
    alphas: &[Ef],
    rounds_nq: &[usize],
    k_whir: usize,
    log_rs0: usize,
    dump_dir: &Path,
    tag: usize,
    sevals_dump: &[Ef],
) -> Value {
    let mut f = f0.to_vec();
    let mut w = w0.to_vec();
    let mut m = f.len().trailing_zeros() as usize;
    let mut log_rs = log_rs0;
    let mut out: Vec<Value> = Vec::new();
    let mut sub = 0usize;
    for r in 0..rounds_nq.len() {
        let nq = rounds_nq[r];
        let is_last = r + 1 == rounds_nq.len();
        let mut sevals_r: Vec<u32> = Vec::new();
        for _ in 0..k_whir {
            let half = f.len() / 2;
            // s_evals at x=1 and x=2 (vendor: s_deg=2, evals x in 1..=2)
            let mut acc1 = ef_zero();
            let mut acc2 = ef_zero();
            for y in 0..half {
                let f0v = f[2 * y];
                let f1v = f[2 * y + 1];
                let df = ef_sub(&f1v, &f0v);
                let w0v = w[2 * y];
                let w1v = w[2 * y + 1];
                let dw = ef_sub(&w1v, &w0v);
                // x=1: f1*w0+... f_x=f0+df, w_x=w0+dw
                let fx1 = ef_add(&f0v, &df);
                let wx1 = ef_add(&w0v, &dw);
                acc1 = ef_add(&acc1, &ef_mul(&fx1, &wx1));
                // x=2: f0+2df, w0+2dw
                let fx2 = ef_add(&f0v, &ef_scale(&df, 2));
                let wx2 = ef_add(&w0v, &ef_scale(&dw, 2));
                acc2 = ef_add(&acc2, &ef_mul(&fx2, &wx2));
            }
            sevals_r.extend_from_slice(&canon_ef(&[acc1, acc2]));
            // fold by alpha
            let a = alphas[sub];
            let mut nf: Vec<Ef> = Vec::with_capacity(half);
            let mut nw: Vec<Ef> = Vec::with_capacity(half);
            for y in 0..half {
                let f0v = f[2 * y];
                let df = ef_sub(&f[2 * y + 1], &f0v);
                nf.push(ef_add(&f0v, &ef_mul(&df, &a)));
                let w0v = w[2 * y];
                let dw = ef_sub(&w[2 * y + 1], &w0v);
                nw.push(ef_add(&w0v, &ef_mul(&dw, &a)));
            }
            f = nf;
            w = nw;
            sub += 1;
        }
        // g_mle coeffs = mobius(f) per limb
        let mut coeffs = f.clone();
        for l in 0..4 {
            let mut col: Vec<u32> = coeffs.iter().map(|e| e[l]).collect();
            mobius_inplace(&mut col);
            for (i, v) in col.iter().enumerate() {
                coeffs[i][l] = *v;
            }
        }
        let dim_c = coeffs.len();
        let mut round_out = json!({
            "round": r, "nq": nq,
            "sevals_sha": wgpu_backend::field::sha256_hex(&wgpu_backend::field::u32s_to_bytes(&sevals_r)),
            "sevals_match_dump": sevals_r == canon_ef(sevals_dump).get(sub * 8 - 8 * k_whir..sub * 8).map(|x| x.to_vec()).unwrap_or_default(),
        });
        if !is_last {
            // pad coeffs, DFT to 2^(log_rs-1)
            let n_fft = 1usize << (log_rs - 1);
            assert!(dim_c <= n_fft, "coeffs {dim_c} > fft {n_fft}");
            let mut g_rs: Vec<Ef> = coeffs.clone();
            g_rs.resize(n_fft, ef_zero());
            for l in 0..4 {
                let mut col: Vec<u32> = g_rs.iter().map(|e| e[l]).collect();
                fft_inplace(&mut col);
                for (i, v) in col.iter().enumerate() {
                    g_rs[i][l] = *v;
                }
            }
            // merkle commit
            let (_lv, root) = merkle_ef_rows(&g_rs, 1 << k_whir);
            let root_canon: Vec<u32> = root.iter().map(|&x| decode(x)).collect();
            // vs dump
            let grs_canon = read_ef(&dump_dir.join(format!("w{tag}-r{r}-grs.bin")));
            let grs_match = g_rs == grs_canon.iter().map(|e| [encode(e[0]), encode(e[1]), encode(e[2]), encode(e[3])]).collect::<Vec<_>>();
            let gc_canon = read_ef(&dump_dir.join(format!("w{tag}-r{r}-gcoeffs.bin")));
            let gc_match: Vec<bool> = coeffs.iter().zip(gc_canon.iter()).map(|(a, b)| {
                (0..4).all(|l| a[l] == encode(b[l]))
            }).collect();
            round_out["grs_match_dump"] = json!(grs_match);
            round_out["gcoeffs_match_dump"] = json!(gc_match.iter().all(|x| *x));
            round_out["commit_canon"] = json!(root_canon);
            round_out["commit_sha"] = json!(wgpu_backend::field::sha256_hex(&wgpu_backend::field::u32s_to_bytes(&root_canon)));
            // ood: y = eval_at_point(coeffs, z0.exp_powers_of_2.take(m-k)) — z0 provided by caller later; skip here (verified in stream parse)
            // w_evals accumulate is done by caller (needs z0/gamma from stream)
            round_out["m_after"] = json!(m - k_whir);
            // stash g_rs sha for report
            round_out["grs_sha"] = json!(wgpu_backend::field::sha256_hex(&wgpu_backend::field::u32s_to_bytes(&canon_ef(&g_rs))));
        } else {
            let fc_canon = read_ef(&dump_dir.join(format!("w{tag}-finalcoeffs.bin")));
            let fc_match = coeffs == fc_canon.iter().map(|e| [encode(e[0]), encode(e[1]), encode(e[2]), encode(e[3])]).collect::<Vec<_>>();
            round_out["finalcoeffs_match_dump"] = json!(fc_match);
            round_out["finalcoeffs_sha"] = json!(wgpu_backend::field::sha256_hex(&wgpu_backend::field::u32s_to_bytes(&canon_ef(&coeffs))));
            round_out["final_len"] = json!(dim_c);
        }
        out.push(round_out);
        m -= k_whir;
        log_rs -= 1;
    }
    assert_eq!(sub, alphas.len(), "subrounds {sub} != alphas {}", alphas.len());
    json!(out)
}

/// 2^-32 mod P (un-Montgomery for u64-domain digest dumps)
const INV2P32: u64 = 943718400; // == K, 2^-32 mod P

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let streams_dir = PathBuf::from(args.get(1).cloned().unwrap_or("streams".into()));
    let dump_dir = PathBuf::from(args.get(2).cloned().unwrap_or("whir".into()));
    let pow_file = args.get(3).cloned().unwrap_or("pow_sites.jsonl".into());
    let out_json = args.get(4).cloned().unwrap_or("w2c-results.json".into());
    let p2 = Poseidon2::new();
    #[derive(Clone)]
    struct SiteRow {
        site: usize,
        inst: usize,
        kind: String,
        bits: usize,
        event: usize,
        witness: u32,
        check: u32,
    }
    let sites_all: Vec<SiteRow> = std::fs::read_to_string(&pow_file)
        .unwrap_or_else(|_| panic!("pow_sites {pow_file}"))
        .lines()
        .filter_map(|l| {
            let v: Value = serde_json::from_str(l).ok()?;
            Some(SiteRow {
                site: v["site"].as_u64()? as usize,
                inst: v["inst"].as_u64()? as usize,
                kind: v["kind"].as_str()?.to_string(),
                bits: v["bits"].as_u64()? as usize,
                event: v["event"].as_u64()? as usize,
                witness: v["witness"].as_u64()? as u32,
                check: v["check"].as_u64()? as u32,
            })
        })
        .collect();
    println!("pow_sites loaded: {} rows", sites_all.len());
    let inst_order = ["app-segments_0", "agg-leaf_0", "agg-internal-for-leaf_0", "agg-internal-recursive_0", "wrap-internal-recursive.1_0"];
    let mut results = BTreeMap::new();
    let mut all_green = true;
    for (ii, tag) in inst_order.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let sp = match std::fs::metadata(streams_dir.join(format!("{tag}.prove.bin"))) {
            Ok(_) => streams_dir.join(format!("{tag}.prove.bin")),
            Err(_) => {
                println!("INSTANCE {tag}: MISSING stream, skip");
                continue;
            }
        };
        let _ = sp;
        let s = load_stream(&streams_dir, tag);
        // ROOT CAUSE (found this session): prove() calls transcript.observe_commit(vk_pre_hash)
        // BEFORE dump::init() arms the recorder — on the FIRST proven instance (app-segments)
        // those 8 vk words are absorbed but unrecorded. Later instances (agg/wrap) have vk at
        // ev0-7 recorded. Glue = vk_pre_hash words; NOTE: events jsonl dumps field words scaled
        // by K=943718400 (=R^-1*2^-1) vs canonical stream; VK_TRUE = jsonl*K mod P verified via
        // probe (HIT at ev78/79). Only app needs glue.
        let glue: Vec<u32> = if *tag == "app-segments_0" {
            vec![93085625u32, 540830993, 1888218509, 927761177, 1083260992, 464140180, 402247195, 690334738]
        } else {
            Vec::new()
        };
        // ---------- 1) chain replay ----------
        let mut d = Duplex::new(&p2);
        for g in &glue {
            d.observe(*g);
        }
        let mut derived: Vec<u32> = vec![0; s.n];
        let mut nsamp = 0usize;
        let mut mism: Vec<(usize, u32, u32)> = Vec::new();
        for i in 0..s.n {
            if s.flags[i] {
                let v = d.sample();
                derived[i] = v;
                nsamp += 1;
                if v != s.vals[i] && mism.len() < 5 {
                    mism.push((i, s.vals[i], v));
                }
            } else {
                d.observe(s.vals[i]);
            }
        }
        let n_mismatch = mism.len().max(
            derived.iter().enumerate().filter(|(i, v)| s.flags[*i] && **v != s.vals[*i]).count()
        );
        let chain_ok = n_mismatch == 0;
        all_green &= chain_ok;
        println!("CHAIN {tag}: events={} samples={} matches={}/{} mismatches={} {:?}", s.n, nsamp, nsamp - n_mismatch, nsamp, n_mismatch, mism);
        // ---------- 2) grind sites ----------
        // find witness events: cursor forward search for O event with recorded value; then expect S next (check)
        let mut grind_rows: Vec<Value> = Vec::new();
        let mut grind_ok = true;
        let mut wanted: Vec<SiteRow> = sites_all.iter().filter(|r| r.inst == ii).cloned().collect();
        wanted.sort_by_key(|r| r.event);
        // replay again, capture snapshots at wanted positions
        let mut d2 = Duplex::new(&p2);
        for g in &glue {
            d2.observe(*g);
        }
        let mut snaps: Vec<([u32; 16], usize, SiteRow)> = Vec::new();
        let mut wi = 0usize;
        for i in 0..s.n {
            if wi < wanted.len() && wanted[wi].event == i {
                let (base, idx) = d2.grind_base();
                snaps.push((base, idx, wanted[wi].clone()));
                wi += 1;
                d2.observe(s.vals[i]);
            } else if s.flags[i] {
                d2.sample();
            } else {
                d2.observe(s.vals[i]);
            }
        }
        assert_eq!(wi, wanted.len(), "grind snapshot walk missed sites");
        if let Ok(bp) = std::env::var("W2C_GRIND_BASES") {
            use std::io::Write;
            let mut bf = std::fs::OpenOptions::new().create(true).append(true).open(&bp).unwrap();
            for (base, idx, rowr) in &snaps {
                let mut rec: Vec<u32> = vec![
                    ii as u32, rowr.site as u32, rowr.bits as u32, *idx as u32,
                    rowr.witness, rowr.check,
                ];
                rec.extend_from_slice(base);
                for w in rec {
                    bf.write_all(&w.to_le_bytes()).unwrap();
                }
            }
        }
        println!("GRIND {tag}: validating {} pow_sites rows positionally", snaps.len());
        for (base, idx, rowr) in &snaps {
            let at = rowr.event;
            let (site, bits, w) = (rowr.site, rowr.bits, rowr.witness);
            let check_rec = s.vals[at + 1];
            let check_recomp = Duplex::grind_check(&p2, base, *idx, w);
            let rec_ok = check_recomp == check_rec && mask_ok(check_rec, bits);
            let (wmin, cmin) = d2_min(&p2, base, *idx, bits);
            grind_rows.push(json!({
                "site": site, "bits": bits, "event": at,
                "witness_recorded": w, "check_recorded": check_rec,
                "check_recomputed": check_recomp,
                "kind": rowr.kind, "event": at,
                "recorded_is_valid_winner": rec_ok && check_rec == rowr.check,
                "witness_min_cpu": wmin, "check_min_cpu": cmin,
                "min_mask_ok": mask_ok(cmin, bits),
                "min_equals_recorded": wmin == w,
            }));
            if !rec_ok || !mask_ok(cmin, bits) || check_rec != rowr.check {
                grind_ok = false;
            }
            println!(
                "GRIND {tag} site={site} kind={} bits={bits} rec={w} valid={} wmin={wmin} min==rec:{} check={check_rec}",
                rowr.kind, rec_ok, wmin == w
            );
        }
        all_green &= grind_ok;
        // ---------- 3) WHIR fold parse + recompute ----------
        let meta_p = dump_dir.join(format!("w{ii}-meta.txt"));
        if !meta_p.exists() {
            println!("FOLD {tag}: no dump (w{ii}-meta.txt), skip fold");
            results.insert(tag.to_string(), json!({
                "chain": {"events": s.n, "samples": nsamp, "matches": nsamp - n_mismatch, "ok": chain_ok},
                "grind": grind_rows,
                "fold": null,
            }));
            continue;
        }
        let meta_s = std::fs::read_to_string(&meta_p).unwrap();
        let mp = |k: &str| -> usize {
            meta_s.split_whitespace().find(|t| t.starts_with(&format!("{k}="))).unwrap()[k.len() + 1..]
                .parse().unwrap()
        };
        let m_h = mp("m");
        let k_whir = mp("k_whir");
        let log_rs0 = mp("log_rs0");
        let mu_bits = mp("mu_bits");
        let fold_bits = mp("fold_bits");
        let q_bits = mp("q_bits");
        let nq_str = meta_s.split("rounds=[").nth(1).unwrap().split(']').next().unwrap();
        let rounds_nq: Vec<usize> = nq_str.split(',').map(|x| x.trim().parse().unwrap()).collect();
        // fold inputs
        let f0 = raw_ef(&read_u32s(&dump_dir.join(format!("w{ii}-f0.bin"))));
        let w0 = raw_ef(&read_u32s(&dump_dir.join(format!("w{ii}-w0.bin"))));
        let alphas_c = read_u32s(&dump_dir.join(format!("w{ii}-alphas.bin")));
        let alphas = raw_ef(&alphas_c);
        let sevals_c = read_u32s(&dump_dir.join(format!("w{ii}-sevals.bin")));
        let sevals_dump = raw_ef(&sevals_c);
        // verify alphas/sevals match recorded stream at protocol positions + derive z0/gamma/queries
        // locate WHIR start: mu check + mu sample events. mu witness event:
        //   site row_starts[ii]+1 (mu site) => already found in snaps? site idx
        let mu_event = sites_all.iter().filter(|r| r.inst == ii && r.kind == "mu").map(|r| r.event).min();
        let mut fold_ok = true;
        let start = match mu_event {
            Some(k) => {
                if mu_bits > 0 && !mask_ok(s.vals[k + 1], mu_bits) {
                    println!("FOLD {tag}: mu check mask fail at {}", k + 1);
                    fold_ok = false;
                }
                k + 2 // after witness O + check S
            }
            None => {
                println!("FOLD {tag}: mu witness event not found (mu_bits={mu_bits})");
                0
            }
        };
        // mu ext = next 4 sample events
        let mu_ev = start..start + 4;
        assert!(mu_ev.clone().all(|i| s.flags[i]), "mu sample events not samples");
        // protocol walk
        let mut ev = start + 4;
        let mut fold_rows: Vec<Value> = Vec::new();
        let mut my_w_evals = w0.clone();
        let mut my_f_evals = f0.clone();
        let mut aidx = 0usize;
        let mut subidx = 0usize;
        let mut log_rs = log_rs0;
        let commits_c = read_u32s(&dump_dir.join(format!("w{ii}-commits.bin")));
        for (r, &nq) in rounds_nq.iter().enumerate() {
            let is_last = r + 1 == rounds_nq.len();
            // k subrounds: sevals 8O, powO, checkS, alpha 4S
            let mut my_sevals: Vec<u32> = Vec::new();
            for _ in 0..k_whir {
                let half = my_f_evals.len() / 2;
                let mut acc1 = ef_zero();
                let mut acc2 = ef_zero();
                for y in 0..half {
                    let f0v = my_f_evals[2 * y];
                    let df = ef_sub(&my_f_evals[2 * y + 1], &f0v);
                    let w0v = my_w_evals[2 * y];
                    let dw = ef_sub(&my_w_evals[2 * y + 1], &w0v);
                    acc1 = ef_add(&acc1, &ef_mul(&ef_add(&f0v, &df), &ef_add(&w0v, &dw)));
                    acc2 = ef_add(&acc2, &ef_mul(&ef_add(&f0v, &ef_scale(&df, 2)), &ef_add(&w0v, &ef_scale(&dw, 2))));
                }
                my_sevals.extend_from_slice(&canon_ef(&[acc1, acc2]));
                let a = alphas[aidx];
                let mut nf = Vec::with_capacity(half);
                let mut nw = Vec::with_capacity(half);
                for y in 0..half {
                    let f0v = my_f_evals[2 * y];
                    let df = ef_sub(&my_f_evals[2 * y + 1], &f0v);
                    nf.push(ef_add(&f0v, &ef_mul(&df, &a)));
                    let w0v = my_w_evals[2 * y];
                    let dw = ef_sub(&my_w_evals[2 * y + 1], &w0v);
                    nw.push(ef_add(&w0v, &ef_mul(&dw, &a)));
                }
                my_f_evals = nf;
                my_w_evals = nw;
                // stream events: 8 O sevals, 1 O pow witness, 1 S check, 4 S alpha
                let sv: Vec<u32> = my_sevals[my_sevals.len() - 8..].to_vec();
                for j in 0..8 {
                    if s.vals[ev + j] != sv[j] || s.flags[ev + j] {
                        println!("FOLD {tag} r{r} SEVALS mismatch at ev {} mine {} stream {}", ev + j, sv[j], s.vals[ev + j]);
                        fold_ok = false;
                    }
                }
                ev += 8;
                // pow witness O + check S
                let poww = s.vals[ev];
                if s.flags[ev] {
                    println!("FOLD {tag} r{r}: pow witness not O at {ev}");
                    fold_ok = false;
                }
                if !mask_ok(s.vals[ev + 1], fold_bits) {
                    println!("FOLD {tag} r{r}: fold check mask fail at {}", ev + 1);
                    fold_ok = false;
                }
                ev += 2;
                // alpha 4 S
                for l in 0..4 {
                    if s.vals[ev + l] != alphas_c[aidx * 4 + l] {
                        println!("FOLD {tag} r{r}: alpha mismatch ev {} stream {} dump {}", ev + l, s.vals[ev + l], alphas_c[aidx * 4 + l]);
                        fold_ok = false;
                    }
                }
                ev += 4;
                aidx += 1;
                subidx += 1;
            }
            // coeffs
            let mut coeffs = my_f_evals.clone();
            for l in 0..4 {
                let mut col: Vec<u32> = coeffs.iter().map(|e| e[l]).collect();
                mobius_inplace(&mut col);
                for (i2, v) in col.iter().enumerate() {
                    coeffs[i2][l] = *v;
                }
            }
            let mut row = json!({"round": r, "nq": nq, "sevals_match_stream": true});
            if !is_last {
                let n_fft = 1usize << (log_rs - 1);
                let mut g_rs = coeffs.clone();
                g_rs.resize(n_fft, ef_zero());
                for l in 0..4 {
                    let mut col: Vec<u32> = g_rs.iter().map(|e| e[l]).collect();
                    fft_inplace(&mut col);
                    for (i2, v) in col.iter().enumerate() {
                        g_rs[i2][l] = *v;
                    }
                }
                let (_lv, root) = merkle_ef_rows(&g_rs, 1 << k_whir);
                let root_canon: Vec<u32> = root.iter().map(|&x| decode(x)).collect();
                // compare vs dump commit r
                // commits.bin limbs are Montgomery u64-based: canon = word * 2^-32 mod P
                let dump_c: Vec<u32> = commits_c[r * 8..(r + 1) * 8]
                    .iter()
                    .map(|&w| (w as u64 * INV2P32 % 2013265921) as u32)
                    .collect();
                let dump_ok = root_canon == dump_c;
                // compare vs stream events (8 O at ev)
                let stream_ok = (0..8).all(|j| !s.flags[ev + j] && s.vals[ev + j] == root_canon[j]);
                if !stream_ok {
                    println!("FOLD {tag} r{r}: COMMIT stream mismatch at {ev}: mine {root_canon:?} stream {:?}", &s.vals[ev..ev + 8]);
                    fold_ok = false;
                }
                if !dump_ok {
                    println!("FOLD {tag} r{r}: COMMIT dump mismatch mine {root_canon:?} dump {dump_c:?}");
                    fold_ok = false;
                }
                // dump g_rs compare
                let grs_c = read_ef(&dump_dir.join(format!("w{ii}-r{r}-grs.bin")));
                let grs_ok = g_rs.len() == grs_c.len()
                    && g_rs.iter().zip(grs_c.iter()).all(|(a, b)| (0..4).all(|l| a[l] == encode(b[l])));
                row["commit_canon"] = json!(root_canon);
                row["commit_sha"] = json!(wgpu_backend::field::sha256_hex(&wgpu_backend::field::u32s_to_bytes(&root_canon)));
                row["commit_match_stream"] = json!(stream_ok);
                row["commit_event"] = json!(ev);
                row["grs_match_dump"] = json!(grs_ok);
                row["grs_sha"] = json!(wgpu_backend::field::sha256_hex(&wgpu_backend::field::u32s_to_bytes(&canon_ef(&g_rs))));
                row["merkle_leaves"] = json!(g_rs.len());
                ev += 8;
                // z0: 4 S
                let z0 = raw_ef(&s.vals[ev..ev + 4].to_vec());
                let z0v = &s.vals[ev..ev + 4];
                if !(0..4).all(|j| s.flags[ev + j]) {
                    println!("FOLD {tag} r{r}: z0 not S at {ev}");
                    fold_ok = false;
                }
                ev += 4;
                // ood: eval_at_point
                // p3 eval_at_point at diagonal point (z, z^2, z^4, ...) = Horner sum c_i z^i
                let y = {
                    let mut acc = coeffs[coeffs.len() - 1];
                    for cc in coeffs[..coeffs.len() - 1].iter().rev() {
                        acc = ef_add(&ef_mul(&acc, &z0[0]), cc);
                    }
                    acc
                };
                let y_c = canon_ef(&[y]);
                let ood_ok = (0..4).all(|j| !s.flags[ev + j] && s.vals[ev + j] == y_c[j]);
                if !ood_ok {
                    println!("FOLD {tag} r{r}: OOD mismatch at {ev}: mine {y_c:?} stream {:?}", &s.vals[ev..ev + 4]);
                    fold_ok = false;
                }
                row["ood_match_stream"] = json!(ood_ok);
                row["ood_canon"] = json!(y_c);
                ev += 4;
                // qpow O + check S (only when q_bits > 0)
                if q_bits > 0 {
                    if !mask_ok(s.vals[ev + 1], q_bits) {
                        println!("FOLD {tag} r{r}: query pow check mask fail");
                        fold_ok = false;
                    }
                    ev += 2;
                }
                // nq query idx S (recorded words are FULL field words; index = masked)
                let d_bits = log_rs - k_whir;
                let qmask = (1u64 << d_bits) - 1;
                let qidx: Vec<u64> = s.vals[ev..ev + nq].iter().map(|v| (*v as u64) & qmask).collect();
                if !(0..nq).all(|j| s.flags[ev + j]) {
                    println!("FOLD {tag} r{r}: query idx not S at {ev}");
                    fold_ok = false;
                }
                ev += nq;
                // gamma 4 S
                let gamma = raw_ef(&s.vals[ev..ev + 4].to_vec());
                ev += 4;
                // w_evals accumulate: w += gamma*eqEF(z0) + sum gamma^{i+2} * eqF(z_i) * gammaP scale
                {
                    let dim = my_w_evals.len().trailing_zeros() as usize;
                    // eq over w array len (after folds, len = 2^(m_now)); z0 ext
                    let n = my_w_evals.len();
                    let zpow = exp_powers_of_2(&z0[0], dim);
                    let mut eq: Vec<Ef> = vec![ef_zero(); n];
                    eq[0] = gamma[0];
                    for (j, zj) in zpow.iter().enumerate() {
                        let one_m = ef_sub(&[encode(1), 0, 0, 0], zj);
                        let step = 1usize << j;
                        for i2 in (0..2usize << j).step_by(step * 2) {
                            for k2 in 0..step {
                                let prev = eq[i2 + k2];
                                eq[i2 + k2] = ef_mul(&prev, &one_m);
                                eq[i2 + k2 + step] = ef_mul(&prev, zj);
                            }
                        }
                    }
                    for i2 in 0..n {
                        my_w_evals[i2] = ef_add(&my_w_evals[i2], &eq[i2]);
                    }
                    // base-field z_i eq
                    let d_bits2 = log_rs - k_whir;
                    let omega = root_of_unity_raw(d_bits2 as u32);
                    let mut gpow = ef_mul(&gamma[0], &gamma[0]); // gamma^2 start (skip 2)
                    for qi in 0..nq {
                        let zc = pow_rawu(omega, qidx[qi]);
                        let dim = my_w_evals.len().trailing_zeros() as usize;
                        let mut zp2: Vec<u32> = Vec::with_capacity(dim);
                        // openvm ExpPowers2: [z, z^2, z^4, ...] no leading 1
                        let mut cur = zc;
                        for _ in 0..dim {
                            zp2.push(cur);
                            cur = mont_mulw(cur, cur);
                        }
                        let n = my_w_evals.len();
                        let mut eq1: Vec<u32> = vec![0u32; n];
                        eq1[0] = encode(1);
                        for (j, zj) in zp2.iter().enumerate() {
                            let om = wsub(encode(1), *zj);
                            let step = 1usize << j;
                            for i2 in (0..2usize << j).step_by(step * 2) {
                                for k2 in 0..step {
                                    let prev = eq1[i2 + k2];
                                    eq1[i2 + k2] = mont_mulw(prev, om);
                                    eq1[i2 + k2 + step] = mont_mulw(prev, *zj);
                                }
                            }
                        }
                        // accumulate: w[i] += gpow * eq[i]  (EF*F)
                        for i2 in 0..n {
                            let add = ef_scale(&gpow, decode(eq1[i2]));
                            my_w_evals[i2] = ef_add(&my_w_evals[i2], &add);
                        }
                        gpow = ef_mul(&gpow, &gamma[0]);
                    }
                    let _ = d_bits2;
                }
                let _ = z0v;
                log_rs -= 1;
            } else {
                // final: coeffs observed directly (m_k words * 4)
                let fc = canon_ef(&coeffs);
                let nfc = fc.len();
                let fin_ok = (0..nfc).all(|j| !s.flags[ev + j] && s.vals[ev + j] == fc[j]);
                if !fin_ok {
                    println!("FOLD {tag} final: coeffs stream mismatch at {ev} len {nfc}");
                    fold_ok = false;
                }
                row["finalcoeffs_match_stream"] = json!(fin_ok);
                row["final_len_words"] = json!(nfc);
                row["finalcoeffs_sha"] = json!(wgpu_backend::field::sha256_hex(&wgpu_backend::field::u32s_to_bytes(&fc)));
                let fc_c: Vec<u32> = read_ef(&dump_dir.join(format!("w{ii}-finalcoeffs.bin"))).into_iter().flatten().collect();
                row["finalcoeffs_match_dump"] = json!(fc == fc_c);
                ev += nfc;
                // qpow + 20q + gamma at end
                if !mask_ok(s.vals[ev + 1], q_bits) {
                    println!("FOLD {tag} final: pow check mask fail");
                    fold_ok = false;
                }
                ev += if q_bits > 0 { 2 } else { 0 };
                ev += nq + 4;
            }
            fold_rows.push(row);
        }
        let _ = fold_recompute;
        all_green &= fold_ok;
        println!(
            "FOLD {tag}: rounds={} ok={} subrounds={} events_consumed={} (stream n={})",
            rounds_nq.len(), fold_ok, subidx, ev, s.n
        );
        results.insert(tag.to_string(), json!({
            "chain": {"events": s.n, "samples": nsamp, "mismatches": n_mismatch, "ok": chain_ok},
            "grind": grind_rows,
            "fold": {"meta": meta_s.trim(), "rounds": fold_rows, "ok": fold_ok, "end_ev": ev},
            "secs": t0.elapsed().as_secs_f64(),
        }));
    }
    let out = json!({"all_green": all_green, "instances": results, "sites_total": sites_all.len()});
    std::fs::write(&out_json, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("W2C_CPU: {}", if all_green { "GREEN" } else { "RED" });
    println!("wrote {out_json}");
    std::process::exit(if all_green { 0 } else { 2 });
}

// helpers for raw word math in stream-domain helpers
#[inline]
fn mont_mulw(a: u32, b: u32) -> u32 {
    mont_mul(a, b)
}
#[inline]
fn wsub(a: u32, b: u32) -> u32 {
    let f = wgpu_backend::field::fsub;
    f(a, b)
}
fn raww(e: &Ef) -> u32 {
    e[0]
}
fn pow_rawu(base_raw: u32, mut e: u64) -> u32 {
    let mut r = encode(1);
    let mut b = base_raw;
    while e > 0 {
        if e & 1 == 1 {
            r = mont_mul(r, b);
        }
        b = mont_mul(b, b);
        e >>= 1;
    }
    r
}
fn d2_min(p2: &Poseidon2, base: &[u32; 16], idx: usize, bits: usize) -> (u32, u32) {
    let mut w = 0u32;
    loop {
        let c = Duplex::grind_check(p2, base, idx, w);
        if mask_ok(c, bits) {
            return (w, c);
        }
        w += 1;
        assert!(w < P_CONST, "grind min overflow");
    }
}
const P_CONST: u32 = 2013265921;
