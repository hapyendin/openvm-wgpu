//! Host mirror of wgsl::MUL_U32 (pure-u32 16-limb Montgomery REDC) vs the
//! native u64 REDC of wgsl::MUL_U64. Same expression tree as the WGSL,
//! same wrapping-u32 semantics (WGSL u32 ops wrap by spec; Rust here uses
//! explicit wrapping ops). P = 2013265921, INV = -P^{-1} mod 2^32 = 2013265919
//! (verified: 2013265919 * 2013265921 == -1 mod 2^32).
const P: u32 = 0x78000001; // 2013265921
const INV: u32 = 2013265919; // -P^{-1} mod 2^32

struct W64 {
    lo: u32,
    hi: u32,
}

// verbatim mirror of MUL_U32::mul64
fn mul64(a: u32, b: u32) -> W64 {
    let al = a & 65535u32;
    let ah = a >> 16u32;
    let bl = b & 65535u32;
    let bh = b >> 16u32;
    let p0 = al.wrapping_mul(bl);
    let p1 = ah.wrapping_mul(bl);
    let p2 = al.wrapping_mul(bh);
    let p3 = ah.wrapping_mul(bh);
    let mid = (p0 >> 16u32)
        .wrapping_add(p1 & 65535u32)
        .wrapping_add(p2 & 65535u32);
    W64 {
        lo: (p0 & 65535u32) | mid.wrapping_shl(16u32),
        hi: p3.wrapping_add(p1 >> 16u32)
            .wrapping_add(p2 >> 16u32)
            .wrapping_add(mid >> 16u32),
    }
}

// verbatim mirror of MUL_U32::mul
fn mul_u32(a: u32, b: u32) -> u32 {
    let t = mul64(a, b);
    let m = t.lo.wrapping_mul(INV);
    let u = mul64(m, P);
    let co: u32 = if t.lo != 0u32 { 1u32 } else { 0u32 };
    let mut r = t.hi.wrapping_add(u.hi).wrapping_add(co);
    if r >= P {
        r -= P;
    }
    r
}

// mirror of MUL_U64::mul (the golden path) on u64
fn mul_u64(a: u32, b: u32) -> u32 {
    let t = (a as u64) * (b as u64);
    let m = (t as u32).wrapping_mul(INV);
    let u = (m as u64) * (P as u64);
    let mut r = ((t + u) >> 32u32) as u32;
    if r >= P {
        r -= P;
    }
    r
}

// sanity: mul64 exact over full u32 domain (spot-check corners)
fn check_mul64() -> u32 {
    let mut n = 0u64;
    let edges = [0u32, 1, 2, 0xFFFF, 0x10000, 0xFFFFFFFF, 1 << 16, P - 1, P, P + 1];
    for &a in edges.iter() {
        for &b in edges.iter() {
            let w = mul64(a, b);
            assert_eq!(((w.hi as u64) << 32) | w.lo as u64, (a as u64) * (b as u64), "mul64 {} {}", a, b);
            n += 1;
        }
    }
    // + 1<<20 random pairs full-range
    let mut s: u64 = 0x243F6A8885A308D3;
    for _ in 0..(1u64 << 20) {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = (s >> 32) as u32;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let b = (s >> 32) as u32;
        let w = mul64(a, b);
        assert_eq!(((w.hi as u64) << 32) | w.lo as u64, (a as u64) * (b as u64), "mul64 rnd {} {}", a, b);
        n += 1;
    }
    n as u32
}

fn main() {
    // INV check: INV*P == 2^32*k - 1 i.e. (INV*P) mod 2^32 == 2^32-1
    assert_eq!(((INV as u64) * (P as u64)) as u32, u32::MAX, "INV constant");

    let n64 = check_mul64();
    println!("MUL64_EXACT: PASS ({} pairs, full u32 domain)", n64);

    // REDC compare over a,b < P: extremes + 2^24+ pseudorandom pairs
    let mut cases: Vec<(u32, u32)> = vec![
        (0, 0), (0, 1), (1, 0), (1, 1), (P - 1, P - 1), (P - 1, 1), (1, P - 1),
        (2, P - 2), (P - 2, 2), (1 << 30, 1 << 30), (P - 1, P >> 1), (0x40000000 - 1, 0x40000000 - 1),
        (P - 1, 0x7FFFFFFE & (P - 1)),
    ];
    let mut s: u64 = 0x9E3779B97F4A7C15;
    const N: usize = 1 << 24; // 16_777_216 random pairs
    let mut rngs: Vec<(u32, u32)> = Vec::with_capacity(N);
    for _ in 0..N {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let a = ((s >> 32) as u32) % P;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let b = ((s >> 32) as u32) % P;
        rngs.push((a, b));
    }
    cases.extend_from_slice(&rngs);

    let mut bad = 0usize;
    let mut first: Option<(u32, u32, u32, u32)> = None;
    for &(a, b) in cases.iter() {
        let e = mul_u64(a, b);
        let g = mul_u32(a, b);
        if e != g {
            bad += 1;
            if first.is_none() {
                first = Some((a, b, e, g));
            }
        }
    }
    let total = cases.len();
    match first {
        Some((a, b, e, g)) => println!(
            "U32_REDC_COMPARE: FAIL count={}/{} first a={} b={} expect_u64={} got_u32={}",
            bad, total, a, b, e, g
        ),
        None => println!("U32_REDC_COMPARE: PASS {}/{} equal", total, total),
    }
    if bad != 0 {
        std::process::exit(1);
    }
}
