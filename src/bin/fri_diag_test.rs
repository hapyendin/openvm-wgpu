// Quick diagnostic: does our CPU Poseidon2 match GPU for a single row?
use wgpu_backend::{InitOpts, WgpuBackend, WgpuMat};
use wgpu_backend::field::{Poseidon2, encode, decode, to_hex, mont_mul, fadd, P};

const SEED: u64 = 0x0000f11e5eedbeef;
fn gen_lcg(n: usize, seed: u64) -> Vec<u32> {
    let (mut s, mut v) = (seed, Vec::with_capacity(n));
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        v.push((s >> 32) as u32);
    }
    v
}

/* Single row hash: [row|0^8] -> Poseidon2.perm_canon -> top 8 decoded. */
fn h_row(row: &[u32; 8], p2: &Poseidon2) -> [u32; 8] {
    let mut st = [0u32; 16];
    for i in 0..8 { st[i] = encode(row[i]); }
    p2.mix(&mut st);
    let mut o = [0u32; 8];
    for i in 0..8 { o[i] = decode(st[i]); }
    o
}

fn main() {
    println!("=== FRI_DIAG: Individual Row Hash Comparison ===\n");
    
    let p2 = Poseidon2::new();
    
    /* Generate trace (canonical). */
    let trace = gen_lcg(512, SEED);
    println!("Generated 64x{} trace (LCG)", 8);
    println!("  First 8 canonical: {}", to_hex(&trace[0..8]));
    
    /* Compute a few row hashes on CPU. */
    for r in 0..4 {
        let mut rw = [0u32; 8];
        for c in 0..8 { rw[c] = trace[r*8+c]; }
        let rh = h_row(&rw, &p2);
        println!("  CPU Row {} hash: {}", r, to_hex(&rh));
    }
    
    /* Now GPU: same 64 traces, hashed by fri_commit_8col kernel. */
    let opts = InitOpts { require_intel_arc: true, allow_u32_only: false };
    let gb = WgpuBackend::new(opts);
    
    /* Use the 64 trace rows directly (no LDE) -- just to verify hash function. */
    let lde_mat = WgpuMat::from_host(&gb.dev, &trace, 8);  /* 64 rows x 8 cols */
    let dbuf = WgpuMat::zeros(&gb.dev, 64, 8);
    gb.fri_commit_8col(&lde_mat, &dbuf);
    let gm = dbuf.copy_out(&gb.dev);
    
    /* Decode GPU digests (canonical). */
    let gpu_digests: Vec<[u32; 8]> = (0..64).map(|r| {
        let mut d = [0u32; 8];
        for c in 0..8 { d[c] = decode(gm[r*8+c]); }
        d
    }).collect();
    
    println!("\n  GPU Row 0-3 hash:");
    for r in 0..4 {
        println!("    Row {}: {}", r, to_hex(&gpu_digests[r]));
    }
    
    /* Compare word-by-word. */
    let mut eq_count = 0u32;
    let mut ne_count = 0u32;
    for r in 0..4 {
        let mut cw = [0u32; 8];
        for c in 0..8 { cw[c] = trace[r*8+c]; }
        let cpu_h = h_row(&cw, &p2);
        for i in 0..8 {
            if cpu_h[i] == gpu_digests[r][i] {
                eq_count += 1;
            } else {
                ne_count += 1;
                if ne_count <= 4 {
                    println!("  MISMATCH ({},{}) cpu={:08x} gpu={:08x}", 
                             r, i, cpu_h[i], gpu_digests[r][i]);
                }
            }
        }
    }
    println!("\n  Match: {} words equal, {} mismatched", eq_count, ne_count);
    if eq_count == 32 {
        println!("  POSEIDON2 HASH: EXACT MATCH (GPU kernel == CPU reference)");
        std::process::exit(0);
    } else {
        /* Encoding direction checks on mismatches. */
        for r in 0..4 {
            let mut cw = [0u32; 8];
            for c in 0..8 { cw[c] = trace[r*8+c]; }
            let cpu_h = h_row(&cw, &p2);
            for i in 0..8 {
                if cpu_h[i] != gpu_digests[r][i] {
                    let gc = cpu_h[i];
                    let gw = gpu_digests[r][i];
                    /* Try R2 multiply (decode reverse): encode(gw) = GW*R2 mod P */
                    if encode(gw) == gc {
                        println!("  -> GW is ENCODED form of CW (needs decode: GW*R2^-1)");
                    }
                    /* Try multiply by 1: should give canonical from Montgomery */
                    if mont_mul(gw, 1u32) == gc {
                        println!("  -> GW decoded (x1) matches CW correctly");
                    }
                    /* Byte-swap */
                    if u32::from_be_bytes(gw.to_ne_bytes()) == gc {
                        println!("  -> Byte-order reversed (LE vs BE input)");
                    }
                    break;  /* one sample per row */
                }
            }
        }
        std::process::exit(1);
    }
}
