// CPU reference field ops + Poseidon2 — verbatim port of b1/WGPU-PROBE reference code.
// Montgomery raw-word domain mirrors bb31_t/fp.hpp. DO NOT alter without
// re-gating against b1 goldens (probe law).
include!("constants_gen.rs"); // pub const IC/TC/IRC/DIAG: [u32; N] canonical words, from b1 poseidon2.hpp

pub const P: u32 = 2013265921; // 15*2^27+1
pub const MM: u32 = 0x77FFFFFF; // -P^-1 mod 2^32 (additive REDC)
pub const R2: u32 = 1172168163; // 2^64 mod P
pub const HALF_P1: u32 = 1006632961; // (P+1)/2
/// MMIX golden-ratio constant used by the conformance-v1 merkle vectors.
pub const MERKLE_SEED: u64 = 0x9e3779b97f4a7c15;
pub const MERKLE_LEAVES: usize = 4096;
pub const MERKLE_ROOT_HEX: &str =
    "1264a0fd266c3e7b5a4cac98234e55a12e2a472a058f37432ecd8aa05387a53c";
pub const MERKLE_ROOT_SHA256: &str =
    "e561ffc54da3384d00013c69da0596292980f2388eb0dfece045fcab315b1917";

#[inline]
pub fn mont_mul(a: u32, b: u32) -> u32 {
    let t = (a as u64) * (b as u64);
    let m = (t as u32).wrapping_mul(MM);
    let u = (m as u64) * (P as u64);
    let s = t + u; // < 2^64 (t < 2^63, u < 2^63)
    let mut r = (s >> 32) as u32;
    if r >= P {
        r -= P;
    }
    r
}
#[inline]
pub fn encode(a: u32) -> u32 {
    mont_mul(a, R2)
}
#[inline]
pub fn decode(a: u32) -> u32 {
    mont_mul(a, 1)
}
#[inline]
pub fn fadd(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= P {
        s - P
    } else {
        s
    }
}
#[inline]
pub fn fsub(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        a + (P - b)
    }
}
#[inline]
fn halve(x: u32) -> u32 {
    (x >> 1) + (x & 1) * HALF_P1
}
#[inline]
fn dbl(x: u32) -> u32 {
    fadd(x, x)
}
#[inline]
fn d7(x: u32) -> u32 {
    let x2 = mont_mul(x, x);
    let x4 = mont_mul(x2, x2);
    let x6 = mont_mul(x4, x2);
    mont_mul(x6, x)
}

fn enc_const(c: &[u32]) -> Vec<u32> {
    c.iter().map(|&x| encode(x)).collect()
}

fn m_ext(c: &mut [u32; 16]) {
    let mut ts = [0u32; 4];
    for i in 0..4 {
        let (x0, x1, x2, x3) = (c[i * 4], c[i * 4 + 1], c[i * 4 + 2], c[i * 4 + 3]);
        let t01 = fadd(x0, x1);
        let t23 = fadd(x2, x3);
        let t0123 = fadd(t01, t23);
        let t01123 = fadd(t0123, x1);
        let t01233 = fadd(t0123, x3);
        c[i * 4 + 3] = fadd(t01233, dbl(x0));
        c[i * 4 + 1] = fadd(t01123, dbl(x2));
        c[i * 4] = fadd(t01123, t01);
        c[i * 4 + 2] = fadd(t01233, t23);
        for j in 0..4 {
            ts[j] = fadd(ts[j], c[i * 4 + j]);
        }
    }
    for i in 0..16 {
        c[i] = fadd(c[i], ts[i % 4]);
    }
}

fn internal_layer(c: &mut [u32; 16], diag: &[u32; 16]) {
    let mut part_sum = c[1];
    for i in 2..16 {
        part_sum = fadd(part_sum, c[i]);
    }
    let sum = fadd(part_sum, c[0]);
    c[0] = fsub(part_sum, c[0]);
    c[1] = fadd(sum, c[1]);
    c[2] = fadd(sum, dbl(c[2]));
    c[3] = fadd(sum, halve(c[3]));
    c[4] = fadd(sum, fadd(dbl(c[4]), c[4]));
    c[5] = fadd(sum, dbl(dbl(c[5])));
    c[6] = fsub(sum, halve(c[6]));
    c[7] = fsub(sum, fadd(dbl(c[7]), c[7]));
    c[8] = fsub(sum, dbl(dbl(c[8])));
    c[9] = fadd(sum, mont_mul(c[9], diag[9]));
    c[10] = fadd(sum, halve(halve(c[10])));
    c[11] = fadd(sum, mont_mul(c[11], diag[11]));
    c[12] = fadd(sum, mont_mul(c[12], diag[12]));
    c[13] = fadd(sum, mont_mul(c[13], diag[13]));
    c[14] = fadd(sum, mont_mul(c[14], diag[14]));
    c[15] = fadd(sum, mont_mul(c[15], diag[15]));
}

pub struct Poseidon2 {
    pub ic: [u32; 64],  // Montgomery-encoded
    pub tc: [u32; 64],
    pub irc: [u32; 13],
    pub diag: [u32; 16],
}

/// The GPU constants buffer layout (WGSL offsets): IC(64) | TC(64) | IRC(13) | DIAG(16),
/// all Montgomery-encoded raw words.
pub fn constants_buffer_mont() -> Vec<u32> {
    let mut v = enc_const(&IC);
    v.extend(enc_const(&TC));
    v.extend(enc_const(&IRC));
    v.extend(enc_const(&DIAG));
    v
}

impl Poseidon2 {
    pub fn new() -> Self {
        let ic: [u32; 64] = enc_const(&IC).try_into().unwrap();
        let tc: [u32; 64] = enc_const(&TC).try_into().unwrap();
        let irc: [u32; 13] = enc_const(&IRC).try_into().unwrap();
        let diag: [u32; 16] = enc_const(&DIAG).try_into().unwrap();
        Self { ic, tc, irc, diag }
    }
    /// in/out: raw Montgomery words (state domain of the WGSL kernel)
    pub fn mix(&self, c: &mut [u32; 16]) {
        m_ext(c);
        for r in 0..4 {
            for i in 0..16 {
                c[i] = fadd(c[i], self.ic[r * 16 + i]);
            }
            for i in 0..16 {
                c[i] = d7(c[i]);
            }
            m_ext(c);
        }
        for r in 0..13 {
            c[0] = fadd(c[0], self.irc[r]);
            c[0] = d7(c[0]);
            internal_layer(c, &self.diag);
        }
        for r in 0..4 {
            for i in 0..16 {
                c[i] = fadd(c[i], self.tc[r * 16 + i]);
            }
            for i in 0..16 {
                c[i] = d7(c[i]);
            }
            m_ext(c);
        }
    }
    /// canonical in -> canonical out (golden domain)
    pub fn perm_canon(&self, inp: &[u32; 16]) -> [u32; 16] {
        let mut c = [0u32; 16];
        for i in 0..16 {
            c[i] = encode(inp[i]);
        }
        self.mix(&mut c);
        let mut o = [0u32; 16];
        for i in 0..16 {
            o[i] = decode(c[i]);
        }
        o
    }
}

// ---- golden-vector parsing (b1 vectors.inc) ----
fn parse_hex_state(h: &str) -> [u32; 16] {
    let mut o = [0u32; 16];
    for i in 0..16 {
        o[i] = u32::from_str_radix(&h[i * 8..i * 8 + 8], 16).unwrap();
    }
    o
}
pub fn to_hex(s: &[u32]) -> String {
    let mut out = String::with_capacity(128);
    for w in s {
        out.push_str(&format!("{:08x}", w));
    }
    out
}
pub fn parse_vectors(src: &str) -> Vec<(String, [u32; 16], [u32; 16])> {
    let mut v = Vec::new();
    for line in src.lines() {
        let l = line.trim();
        if l.starts_with("{ \"") {
            let parts: Vec<&str> = l.split('"').collect();
            v.push((
                parts[1].to_string(),
                parse_hex_state(parts[3]),
                parse_hex_state(parts[5]),
            ));
        }
    }
    v
}

pub fn u32s_to_bytes(v: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for w in v {
        b.extend_from_slice(&w.to_le_bytes());
    }
    b
}
pub fn bytes_to_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let h = Sha256::digest(data);
    let mut s = String::new();
    for b in h {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
