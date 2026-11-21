// Minimal diag: find which variant matches GPU output
use wgpu_backend::field::{Poseidon2, encode, decode, mont_mul, P};
use wgpu_backend::{InitOpts, WgpuBackend, WgpuMat};

fn mpow(base: u64, exp: u64, modulus: u64) -> u64 {
    let (mut r, mut b, mut e) = (1u64, base % modulus, exp);
    while e > 0 { if e & 1 == 1 { r = r*b % modulus; } b = b*b % modulus; e >>= 1; }
    r
}

fn inv_p(a: u64) -> u64 { mpow(a, P as u64 - 2, P as u64) }
const R2: u32 = 1172168163u32;
const INV_R2: u32 = 171208848u32; /* inverse of R² mod P, computed offline */

fn enc(u: u32) -> u32 { mont_mul(u, R2) }
fn dec(u: u32) -> u32 { mont_mul(u, INV_R2) }

/* Old: encode → mix → decode. */
fn h_old(row: &[u32; 8], p2: &Poseidon2) -> [u32; 8] {
    let mut st = [0u32; 16];
    for i in 0..8 { st[i] = enc(row[i]); }
    p2.mix(&mut st);
    let mut o = [0u32; 8];
    for i in 0..8 { o[i] = decode(st[i]); }
    o
}

/* Perm_canon with raw canonical data (no encoding). */
fn h_plan(row: &[u32; 8], p2: &Poseidon2) -> [u32; 8] {
    let mut st = [0u32; 16];
    for i in 0..8 { st[i] = row[i]; }
    let res = p2.perm_canon(&st);
    let mut o = [0u32; 8];
    for i in 0..8 { o[i] = res[i]; }
    o
}

/* Perm_canon with Gold (encoded) data. */
fn h_gold(row: &[u32; 8], p2: &Poseidon2) -> [u32; 8] {
    let mut st = [0u32; 16];
    for i in 0..8 { st[i] = enc(row[i]); }
    let res = p2.perm_canon(&st);
    let mut o = [0u32; 8];
    for i in 0..8 { o[i] = res[i]; }
    o
}

fn gen_lcg(n: usize, seed: u64) -> Vec<u32> {
    let (mut s, mut v) = (seed, Vec::with_capacity(n));
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        v.push((s >> 32) as u32);
    }
    v
}

fn count_match(a: &[u32; 8], b: &[u32; 8]) -> usize {
    (0..8).filter(|&i| a[i] == b[i]).count()
}

fn main() {
    let p2 = Poseidon2::new();
    let trace = gen_lcg(64, 0x0000f11e5eedbeef);
    let row0: [u32; 8] = trace[0..8].try_into().unwrap();
    
    println!("Row 0 canon:");
    print!("  "); for v in &row0 { print!("{:08x}", v); } println!();
    
    /* Compute all variants on CPU side. */
    let oa = h_plan(&row0, &p2);  /* perm_canon(canon) */
    print!("  A (perm_canon(canon)): "); for v in &oa { print!("{:08x}", v); } println!();
    
    let ob = h_gold(&row0, &p2);  /* perm_canon(enc(row)) */
    print!("  B (perm_canon(enc)):   "); for v in &ob { print!("{:08x}", v); } println!();
    
    let oc = h_old(&row0, &p2);  /* enc+mix+dec (old style) */
    print!("  C (enc+mix+dec):       "); for v in &oc { print!("{:08x}", v); } println!();
    
    /* Also try applying various transforms to GPU output. */
    let opts = InitOpts { require_intel_arc: true, allow_u32_only: false };
    let gb = WgpuBackend::new(opts);
    
    let mat = WgpuMat::from_host(&gb.dev, &trace, 8);
    let dbuf = WgpuMat::zeros(&gb.dev, 8, 8);
    gb.fri_commit_8col(&mat, &dbuf);
    let gm = dbuf.copy_out(&gb.dev);
    
    let gpu: [u32; 8] = (0..8).map(|c| gm[c]).collect::<Vec<u32>>().try_into().unwrap();
    print!("  GPU (post-patch kernel): "); for v in &gpu { print!("{:08x}", v); } println!();
    
    /* Check matches. */
    println!("\n=== MATCH RESULTS (GPU is ground truth) ===");
    let ma = count_match(&oa, &gpu);
    let mb = count_match(&ob, &gpu);
    let mc = count_match(&oc, &gpu);
    println!("A (perm_canon on canon):  {} / 8", ma);
    println!("B (perm_canon on gold):   {} / 8", mb);
    println!("C (enc+mix+dec):          {} / 8", mc);
    
    /* Try fixing GPU by transforming it to match each variant. */
    /* D: decoded (= GPU * invR²): what does that look like vs OA, OB, OC? */
    let g_decoded: [u32; 8] = (0..8).map(|i| dec(gpu[i])).collect::<Vec<u32>>().try_into().unwrap();
    print!("  G-decoded:               "); for v in &g_decoded { print!("{:08x}", v); } println!();
    println!("  Decoded vs OA: {} / 8", count_match(&g_decoded, &oa));
    println!("  Decoded vs OB: {} / 8", count_match(&g_decoded, &ob));
    println!("  Decoded vs OC: {} / 8", count_match(&g_decoded, &oc));
    
    /* E: scaled by R² (still encoded/Montgomery form). */
    let g_enc: [u32; 8] = (0..8).map(|i| enc(gpu[i])).collect::<Vec<u32>>().try_into().unwrap();
    print!("  G-encoded:               "); for v in &g_enc { print!("{:08x}", v); } println!();
    println!("  Encoded vs OA: {} / 8", count_match(&g_enc, &oa));
    println!("  Encoded vs OB: {} / 8", count_match(&g_enc, &ob));
    println!("  Encoded vs OC: {} / 8", count_match(&g_enc, &oc));
    
    std::process::exit(0);
}
