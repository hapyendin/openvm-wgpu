// FRI Commit Test v6 — C3 APPLY: b3 two-adic generator table + host-pipeline
// fixes per C3-PROTO-CLOSED.md §4, + C4 GPU parity across ALL 6 commit layers.
// Pipeline: trace 64x8 (Lcg MMIX, reduce mod p) -> LDE(blowup=1)->128x8 ->
// hash_rows -> col_fold(gamma) -> row_fold(beta)x4 -> concat sha256 == 99fc8df2.

use wgpu_backend::field::{Poseidon2, encode, mont_mul, to_hex, P};
use wgpu_backend::{InitOpts, WgpuBackend, WgpuMat};

const SEED: u64 = 0x0000f11e5eedbeef;
const GAMMA: u32 = 0x16900435;
const BETAS: [u32; 4] = [0x577941c9, 0x3d51ee15, 0x06516440, 0x5b92f614];
const TROWS: usize = 64;
const RROWS: usize = 128;
const COLS: usize = 8;
const INV2: u64 = 1006632961; // canonical 1/2 = (P+1)/2

const GH: [&str; 6] = [
    "554755441ef4a56a6ff3173214e348df4a203f7a2d5e6c875fcb4c7d60e35a98",
    "1e121f7d55cc1d7c61703af855fb51401a7f0d412dcf824d00d54fba44cf2c9c",
    "38ac4bb64ad645f53e63d7b8608699d83442d33839b7b8446abcf506393c153a",
    "65f7f7263362dff90a81779c3b537822639b87f43bdcce362b0f1d1b22f9598e",
    "67051ee55ebcb4bc6166823e42e1cf2d2b7cd7c06906fe6861519b4b4cdc87bf",
    "104a66623a620a114488f8fe497562dd6c750b83116baf2635ae608254f9a4df",
];
const GSHA: &str = "99fc8df2bfc5e70f48e45b79e657e3ba207925f7871de7423b1c8f5e616c1fda";

/* b3 fp.hpp TWO_ADIC_GENERATORS table (canonical values, k=0..27).
   NEVER derive twiddles by exponentiating a single primitive root (GG style):
   odd-power rotations like ω^39/ω^103 pass order tests but permute LDE rows
   (r→103r mod 128) and desync the Merkle roots. Used: ω_64=OMEGA[6],
   ω_128=OMEGA[7]; fold at old_h=2^k uses OMEGA[k]. */
const OMEGA: [u32; 28] = [
    1, 0x78000000, 0x67055c21, 0x5ee99486, 0x0bb4c4e4, 0x2d4cc4da, 0x669d6090, 0x17b56c64,
    0x67456167, 0x688442f9, 0x145e952d, 0x4fe61226, 0x4c734715, 0x11c33e2a, 0x62c3d2b1, 0x77cad399,
    0x54c131f4, 0x4cabd6a6, 0x5cf5713f, 0x3e9430e8, 0x0ba067a3, 0x18adc27d, 0x21fd55bc, 0x4b859b3d,
    0x3bd57996, 0x4483d85a, 0x3a26eef8, 0x1a427a41,
];

fn gen_lcg(n: usize, seed: u64) -> Vec<u32> {
    // b3 Lcg.next_f(): MMIX LCG, HIGH32 REDUCED MOD P (fri_test.cpp:52)
    let (mut s, mut v) = (seed, Vec::with_capacity(n));
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        v.push(((s >> 32) as u32) % P);
    }
    v
}

fn inv_p(a: u64) -> u64 {
    let (mut r, mut b, mut e) = (1u64, a % P as u64, P as u64 - 2);
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % P as u64;
        }
        b = b * b % P as u64;
        e >>= 1;
    }
    r
}
fn mpow(base: u64, exp: u64, modulus: u64) -> u64 {
    let (mut r, mut b, mut e) = (1u64, base % modulus, exp);
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % modulus;
        }
        b = b * b % modulus;
        e >>= 1;
    }
    r
}
/* Row hash on CPU: perm_canon = encw→perm_mix→decode, first 8 cells. */
fn h_row_cannon(row: &[u32; 8], p2: &Poseidon2) -> [u32; 8] {
    let mut st = [0u32; 16];
    for i in 0..8 {
        st[i] = row[i];
    }
    let res = p2.perm_canon(&st);
    let mut o = [0u32; 8];
    for i in 0..8 {
        o[i] = res[i];
    }
    o
}

/* Merkle tree (bottom-up compression using perm_canon). */
fn mroot(lvls: &[[u32; 8]], p2: &Poseidon2) -> [u32; 8] {
    let mut cur = lvls.to_vec();
    while cur.len() > 1 {
        let nxt: Vec<_> = cur
            .chunks(2)
            .map(|p| {
                let mut buf = [0u32; 16];
                for i in 0..8 {
                    buf[i] = p[0][i];
                    buf[i + 8] = p[1][i];
                }
                let res = p2.perm_canon(&buf);
                let mut r = [0u32; 8];
                for i in 0..8 {
                    r[i] = res[i];
                }
                r
            })
            .collect();
        cur = nxt;
    }
    cur[0]
}

/* Col-fold: 128x8 -> 128 scalars (gamma-weighted sum per row, canonical). */
fn fold_col(data: &[u32], nr: usize, nc: usize, gamma: u32) -> Vec<u32> {
    (0..nr)
        .map(|r| {
            let mut acc: u64 = 0u64;
            for c in 0..nc {
                let cm = u64::from(encode(data[r * nc + c]));
                acc = (acc + mn(cm, mpow(gamma as u64, c as u64, P as u64)) as u64) % (P as u64);
            }
            acc as u32
        })
        .collect()
}

/* Row fold per b3 row_fold_body (C3-PROTO-CLOSED §4.3):
   pairs (j, j+half); even=(a+b)/2; odd=(a-b)/2 * OMEGA[log2(old_h)]^{-j}
   (positive sign, inv2 on odd too); out = even + beta*odd with CONSTANT beta.
   Lanes of padded [v,0*7] rows fold lane-wise; zero lanes stay zero. */
fn do_fold(d: &[[u32; 8]], beta: u32) -> Vec<[u32; 8]> {
    let old_h = d.len();
    let nh = old_h / 2;
    let log_h = old_h.trailing_zeros() as usize; // 128→7, 64→6, 32→5, 16→4
    let ginv = inv_p(OMEGA[log_h] as u64);
    let mut xinv = vec![1u64; nh]; // exact chain of g^{-1} powers (b3 xinv_table)
    for j in 1..nh {
        xinv[j] = xinv[j - 1] * ginv % P as u64;
    }
    (0..nh)
        .map(|j| {
            let (a, b) = (d[j], d[j + nh]);
            let mut res = [0u32; 8];
            for k in 0..8 {
                let sum = a[k] as u64 + b[k] as u64;
                let sum = if sum >= P as u64 { sum - P as u64 } else { sum };
                let even = sum * INV2 % P as u64;
                let diff = if a[k] >= b[k] {
                    (a[k] - b[k]) as u64
                } else {
                    a[k] as u64 + P as u64 - b[k] as u64
                };
                let odd = diff * INV2 % P as u64 * xinv[j] % P as u64;
                res[k] = ((even + beta as u64 * odd % P as u64) % P as u64) as u32;
            }
            res
        })
        .collect()
}

/* Montgomery multiply helper (canonical-in / canonical-out via encw).
   NOTE: the previous local REDC skipped the mod-2^32 on the REDC multiplier,
   letting m*P wrap the u64 accumulator (result truncated to u32) — it silently
   corrupted the LDE. Use the golden field.rs mont_mul. */
fn mn(u: u64, v: u64) -> u32 {
    mont_mul(u as u32, v as u32)
}

/* LDE: IDFT(64)+zero-pad+DFT(128) with b3 table generators (constants, no
   exponentiation from a primitive root). */
fn do_lde(trace_canon: &[u32]) -> Vec<u32> {
    let (w64, w128) = (OMEGA[6] as u64, OMEGA[7] as u64);
    let half = TROWS;
    let nfull = RROWS;
    let mut out = vec![0u32; nfull * COLS];
    for c in 0..COLS {
        let mut col = Vec::with_capacity(half);
        for r in 0..half {
            col.push(trace_canon[r * COLS + c]);
        }
        let mut coeffs = vec![0u32; half];
        for k in 0..half {
            let mut acc: u64 = 0;
            for j in 0..half {
                let wp = inv_p(mpow(w64, (j * k) as u64, P as u64));
                acc = (acc + mn(u64::from(encode(col[j])), wp) as u64) % (P as u64);
            }
            coeffs[k] = ((acc * inv_p(half as u64)) % P as u64) as u32;
        }
        for idx in 0..nfull {
            let mut acc: u64 = 0;
            for k in 0..half {
                let wp = mpow(w128, idx as u64 * k as u64, P as u64);
                acc = (acc + mn(u64::from(encode(coeffs[k])), wp) as u64) % (P as u64);
            }
            out[idx * COLS + c] = acc as u32;
        }
    }
    out
}

/* SHA256 hex digest. */
fn sha2hex(input: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut s = String::with_capacity(64);
    for b in Sha256::digest(input) {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/* GPU row-hash of commit rows via WGSL kernels (encw→perm_mix→decode).
   width 8 -> fri_commit_8col (LDE rows); width 1 -> fri_commit_1col (padded). */
fn gpu_hash(gb: &WgpuBackend, width: usize, data: &[u32], n_rows: usize) -> Vec<[u32; 8]> {
    let bmat = WgpuMat::from_host(&gb.dev, data, width);
    let obuf = WgpuMat::zeros(&gb.dev, n_rows, 8);
    if width == 8 {
        gb.fri_commit_8col(&bmat, &obuf);
    } else {
        gb.fri_commit_1col(&bmat, &obuf);
    }
    let w = obuf.copy_out(&gb.dev);
    (0..n_rows)
        .map(|r| {
            let mut d = [0u32; 8];
            d.copy_from_slice(&w[r * 8..r * 8 + 8]);
            d
        })
        .collect()
}

/* ---- MAIN ---- */
fn main() {
    println!("=== FRI COMMIT TEST v6 (C3 APPLY: b3 generators + pipeline fixes) ===\n");
    let p2 = Poseidon2::new();

    /* --- 1. Generate trace (b3 law: high32 reduced mod p) --- */
    let trace = gen_lcg(TROWS * COLS, SEED);
    println!("TRACE: {}x{} (seed={})", TROWS, COLS, SEED);
    println!("  canonical[0..8]: {}", to_hex(&trace[0..8]));

    /* --- 2. C4 baseline: CPU vs GPU hash of 64 raw trace rows --- */
    let cpu_hashes_64: Vec<[u32; 8]> = (0..TROWS)
        .map(|r| {
            let mut rw = [0u32; 8];
            for c in 0..8 {
                rw[c] = trace[r * COLS + c];
            }
            h_row_cannon(&rw, &p2)
        })
        .collect();

    // any discrete GPU by default; OPENVM_WGPU_ARC=1 asserts Intel Arc + int64 path
    let farm_mode = std::env::var("OPENVM_WGPU_ARC").is_err();
    let opts = InitOpts {
        require_intel_arc: !farm_mode,
        allow_u32_only: farm_mode,
    };
    let gb = WgpuBackend::new(opts);
    println!("Adapter: {} (INT64={})\n", gb.dev.report.name, gb.dev.has_shader_int64);

    let gpu_hashes_64 = gpu_hash(&gb, COLS, &trace, TROWS);
    let mut eq_count: usize = 0;
    let mut bad_words: Vec<(usize, String, String)> = Vec::new();
    for r in 0..std::cmp::min(TROWS, 4) {
        for i in 0..8 {
            if cpu_hashes_64[r][i] == gpu_hashes_64[r][i] {
                eq_count += 1;
            } else if bad_words.len() < 16 {
                bad_words.push((
                    r * 8 + i,
                    format!("{:08x}", cpu_hashes_64[r][i]),
                    format!("{:08x}", gpu_hashes_64[r][i]),
                ));
            }
        }
    }
    println!("C4 Baseline - CPU vs GPU Row-hash (first 4 rows):");
    println!("  Matches: {} / 32 words  (Mismatches: {})", eq_count, 32 - eq_count);
    if eq_count != 32 {
        println!("C4 Baseline FAILED: CPU↔GPU hashes disagree.");
        for (idx, cv, gv) in &bad_words {
            println!("  Word {}: CPU={} vs GPU={}", idx, cv, gv);
        }
        std::process::exit(1);
    }
    let cpu_root64 = mroot(&cpu_hashes_64, &p2);
    println!("Hash pipeline: CPU↔GPU MATCH ✓\nRoot (64 raw trace rows): {}\n", to_hex(&cpu_root64));

    /* --- 3. FULL 128-RUN pipeline (CPU reference) --- */
    println!("--- FULL 128-RUN (LDE via IDFT/DFT, b3 OMEGA table) ---");
    let lde = do_lde(&trace);
    println!("LDE: {}x{} (log_blowup=1)", RROWS, COLS);
    println!("  LDE[0]:  {}", to_hex(&lde[0..COLS]));
    println!("  LDE[last {}]: {}", RROWS - 1, to_hex(&lde[(RROWS - 1) * COLS..]));

    let cpu_d0: Vec<[u32; 8]> = (0..RROWS)
        .map(|r| {
            let mut rw = [0u32; 8];
            for c in 0..8 {
                rw[c] = lde[r * COLS + c];
            }
            h_row_cannon(&rw, &p2)
        })
        .collect();

    let cf = fold_col(&lde, RROWS, COLS, GAMMA);
    // commit rows pad as [v,0,0,0,0,0,0,0] (b3 out8 layout), NOT fill(v)
    let cfl: Vec<[u32; 8]> = (0..RROWS)
        .map(|r| {
            let mut l = [0u32; 8];
            l[0] = cf[r];
            l
        })
        .collect();
    let cpu_d1: Vec<[u32; 8]> = cfl.iter().map(|r| h_row_cannon(r, &p2)).collect();

    // per-layer commit inputs (width, flat words) + CPU digests + roots
    let mut cpu_roots: Vec<[u32; 8]> = Vec::with_capacity(6);
    let mut layer_cpu_digests: Vec<Vec<[u32; 8]>> = Vec::with_capacity(6);
    let mut layer_inputs: Vec<(usize, Vec<u32>)> = Vec::with_capacity(6);

    cpu_roots.push(mroot(&cpu_d0, &p2));
    layer_cpu_digests.push(cpu_d0);
    layer_inputs.push((COLS, lde.clone()));

    // Leaves of every layer tree are the hashed rows (b3: merkle_tree_init
    // output feeds compress); mroot() only compresses, never hashes leaves.
    cpu_roots.push(mroot(&cpu_d1, &p2));
    layer_cpu_digests.push(cpu_d1);
    layer_inputs.push((1, cf.clone()));

    let mut d = cfl.clone();
    for layer in 2..6 {
        let beta = BETAS[(layer - 2) % 4];
        d = do_fold(&d, beta);
        let digs: Vec<[u32; 8]> = d.iter().map(|r| h_row_cannon(r, &p2)).collect();
        let col: Vec<u32> = d.iter().map(|r| r[0]).collect();
        cpu_roots.push(mroot(&digs, &p2));
        layer_cpu_digests.push(digs);
        layer_inputs.push((1, col));
        println!("Layer {}: {} rows, beta={:08x}", layer, d.len(), beta);
    }

    let mut cpu_ok = true;
    for layer in 0..6 {
        let got = to_hex(&cpu_roots[layer]);
        let ok = got == GH[layer];
        cpu_ok &= ok;
        println!(
            "root{}: {} (expected {} ) [{}]",
            layer,
            got,
            GH[layer],
            if ok { "PASS" } else { "FAIL" }
        );
    }
    let cb: Vec<u8> = cpu_roots
        .iter()
        .flat_map(|rd| rd.iter().flat_map(|w| w.to_le_bytes()))
        .collect();
    let cpu_sha = sha2hex(&cb);
    cpu_ok &= cpu_sha == GSHA;
    println!("CPU SHA256: {} (expected: {})", cpu_sha, GSHA);
    println!("GOLDEN_99FC8DF2_CPU: {}", if cpu_ok { "PASS" } else { "FAIL" });

    /* --- 4. GPU path: WGSL hash of every commit row + parity + roots --- */
    println!("\n--- GPU path (encw→perm_mix→decode across all 6 layers) ---");
    let mut gpu_roots: Vec<[u32; 8]> = Vec::with_capacity(6);
    let mut parity_words: usize = 0;
    let mut parity_bad: usize = 0;
    let mut first_bad: Option<(usize, usize, u32, u32)> = None;
    for (li, (width, words)) in layer_inputs.iter().enumerate() {
        // layer0 = 128 8-col rows; layers 1..5 = 1-col padded rows
        let n_rows = if *width == 8 { RROWS } else { words.len() };
        let gd = gpu_hash(&gb, *width, words, n_rows);
        for (r, (g, c)) in gd.iter().zip(layer_cpu_digests[li].iter()).enumerate() {
            for i in 0..8 {
                parity_words += 1;
                if g[i] != c[i] {
                    parity_bad += 1;
                    if first_bad.is_none() {
                        first_bad = Some((li, r, c[i], g[i]));
                    }
                }
            }
        }
        gpu_roots.push(mroot(&gd, &p2));
    }
    let mut gpu_ok = true;
    for layer in 0..6 {
        let got = to_hex(&gpu_roots[layer]);
        let ok = got == GH[layer];
        gpu_ok &= ok;
        println!(
            "GPU root{}: {} [{}]",
            layer,
            got,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    let cb2: Vec<u8> = gpu_roots
        .iter()
        .flat_map(|rd| rd.iter().flat_map(|w| w.to_le_bytes()))
        .collect();
    let gpu_sha = sha2hex(&cb2);
    gpu_ok &= gpu_sha == GSHA;
    println!("GPU SHA256: {}", gpu_sha);
    let parity_ok = parity_bad == 0;
    match first_bad {
        Some((li, r, c, g)) => println!(
            "GPU==CPU parity: {}/{} words BAD, first layer{} row{} CPU={:08x} GPU={:08x}",
            parity_bad, parity_words, li, r, c, g
        ),
        None => println!("GPU==CPU parity: 0/{} words bad", parity_words),
    }
    println!("GPU_CPU_PARITY: {}", if parity_ok { "PASS" } else { "FAIL" });
    println!("GOLDEN_99FC8DF2_GPU: {}", if gpu_ok { "PASS" } else { "FAIL" });

    if cpu_ok && gpu_ok && parity_ok {
        println!("\nC3-APPLY: ALL 6 LAYERS + SHA PASS on CPU and GPU (99fc8df2)");
    } else {
        println!("\nC3-APPLY: FAIL");
        std::process::exit(1);
    }
}
