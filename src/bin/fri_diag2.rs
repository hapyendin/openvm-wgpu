// Minimal diagnostic: send known canonical values through GPU, compare to CPU.
use wgpu_backend::{InitOpts, WgpuBackend, WgpuMat};
use wgpu_backend::field::{Poseidon2, encode, decode, to_hex, mont_mul, fadd, P};

fn gen_lcg(n: usize, seed: u64) -> Vec<u32> {
    let (mut s, mut v) = (seed, Vec::with_capacity(n));
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        v.push((s >> 32) as u32);
    }
    v
}

fn enc(u: u32) -> u32 {
    mont_mul(u, 1172168163u32)  // multiply by R^2 mod P
}

fn inv_p(a: u64) -> u64 {
    let (mut r, mut b, mut e) = (1u64, a % P as u64, P as u64 - 2);
    while e > 0 { if e & 1 == 1 { r = r*b % P as u64; } b = b*b % P as u64; e >>= 1; }
    r
}

/* h_row on CPU: encode->mix->decode. */
fn h_row(row: &[u32; 8], p2: &Poseidon2) -> [u32; 8] {
    let mut st = [0u32; 16];
    for i in 0..8 { st[i] = encode(row[i]); }
    p2.mix(&mut st);
    let mut o = [0u32; 8];
    for i in 0..8 { o[i] = decode(st[i]); }
    o
}

fn main() {
    let p2 = Poseidon2::new();
    let trace = gen_lcg(64, 0x0000f11e5eedbeef);

    /* --- CPU hashes on 64 raw rows (canonical input) --- */
    println!("CPU row-hash (encode→perm→decode):");
    for r in 0..std::cmp::min(4, 64) {
        let mut rw = [0u32; 8];
        for c in 0..8 { rw[c] = trace[r*8+c]; }
        let rh = h_row(&rw, &p2);
        print!("  Row {}:", r);
        for v in &rh { print!(" {:08x}", v); }
        println!();
    }

    /* --- GPU hashes via pipeline (NEW kernel: no encw, no decode) --- */
    let opts = InitOpts { require_intel_arc: true, allow_u32_only: false };
    let gb = WgpuBackend::new(opts);

    let mat = WgpuMat::from_host(&gb.dev, &trace, 8);
    let dbuf = WgpuMat::zeros(&gb.dev, 64, 8);
    gb.fri_commit_8col(&mat, &dbuf);
    let gm = dbuf.copy_out(&gb.dev);

    /* Read raw GPU output (no decoding attempt). */
    println!("\nGPU row-hash (raw read-back):");
    for r in 0..std::cmp::min(4, 64) {
        print!("  Row {}:", r);
        for c in 0..8 { print!(" {:08x}", gm[r*8+c]); }
        println!();
    }

    /* Check various transformations:
       1) raw GPU ≈ CPU (direct match)?
       2) raw GPU × 1 ≟ CPU (decoded)?  
       3) raw GPU × R² ≟ CPU (still encoded/Montgomery)?
       4) raw GPU ≈ CPU × R² (reversed)? */
    const R2: u32 = 1172168163u32;
    const ONE: u32 = 1u32;
    
    let mut eq_raw: usize = 0;
    let mut eq_decoded: usize = 0;
    let mut eq_encoded: usize = 0;
    let mut eq_inv_r2: usize = 0;  /* inverse of R2 = ? */
    
    /* Compute inverse of R2: inv(R2) mod P where we just do: find inv from inv_p. */
    let inv_r2 = inv_p((R2 as u64) * P as u64) % P as u64;
    let inv_r2_val = (inv_r2) as u32;
    
    for r in 0..std::cmp::min(4, 64) {
        let mut rw = [0u32; 8];
        for c in 0..8 { rw[c] = trace[r*8+c]; }
        let ch = h_row(&rw, &p2);
        
        for c in 0..8 {
            let cv = ch[c];
            let gv = gm[r*8+c];
            
            if cv == gv { eq_raw += 1; }
            if cv == mont_mul(gv, ONE) { eq_decoded += 1; }
            if cv == mont_mul(gv, R2) { eq_encoded += 1; }
            if cv == mont_mul(gv, inv_r2_val) { eq_inv_r2 += 1; }
        }
    }
    
    println!("\n=== TRANSFORMATION MATCH ===");
    println!("  Raw GPU ≈ CPU:           {} / 32", eq_raw);
    println!("  GPU × 1 ≟ CPU (decoded): {} / 32", eq_decoded);
    println!("  GPU × R² ≟ CPU (encoded): {} / 32", eq_encoded);
    println!("  GPU × InvR² ≟ CPU (inv-R²): {} / 32\n", eq_inv_r2);
    
    std::process::exit(0);
}
