//! W2C: transcript-chain replay + WHIR fold/grind CPU reference (fork-C gate goldens).
//! Domain law: sponge/poseidon2 operate on RAW Montgomery words (field.rs mix domain);
//! stream events record CANONICAL words (p3 recorder: as_canonical_u64).
use crate::field::{decode, encode, fadd, fsub, mont_mul, sha256_hex, Poseidon2, P};

pub const WIDTH: usize = 16;
pub const RATE: usize = 8;
#[inline]
pub fn mask_ok(w_canon: u32, bits: usize) -> bool {
    w_canon & (((1u64 << bits) - 1) as u32) == 0
}

// ---------------- EF4 BabyBear (X^4 - 11), raw Montgomery limb words ----------------
pub type Ef = [u32; 4];
#[inline]
pub fn ef_zero() -> Ef {
    [0, 0, 0, 0]
}
#[inline]
pub fn ef_add(a: &Ef, b: &Ef) -> Ef {
    [fadd(a[0], b[0]), fadd(a[1], b[1]), fadd(a[2], b[2]), fadd(a[3], b[3])]
}
#[inline]
pub fn ef_sub(a: &Ef, b: &Ef) -> Ef {
    [fsub(a[0], b[0]), fsub(a[1], b[1]), fsub(a[2], b[2]), fsub(a[3], b[3])]
}
/// beta = raw(11): X^4 = 11
pub fn beta_raw() -> u32 {
    encode(11)
}
pub fn ef_mul(a: &Ef, b: &Ef) -> Ef {
    let be = beta_raw();
    let c0d = mont_mul(a[0], b[0]);
    let c0f = fadd(fadd(mont_mul(a[1], b[3]), mont_mul(a[2], b[2])), mont_mul(a[3], b[1]));
    let c1d = fadd(mont_mul(a[0], b[1]), mont_mul(a[1], b[0]));
    let c1f = fadd(mont_mul(a[2], b[3]), mont_mul(a[3], b[2]));
    let c2d = fadd(fadd(mont_mul(a[0], b[2]), mont_mul(a[1], b[1])), mont_mul(a[2], b[0]));
    let c2f = mont_mul(a[3], b[3]);
    let c3d = fadd(
        fadd(mont_mul(a[0], b[3]), mont_mul(a[1], b[2])),
        fadd(mont_mul(a[2], b[1]), mont_mul(a[3], b[0])),
    );
    [
        fadd(c0d, mont_mul(c0f, be)),
        fadd(c1d, mont_mul(c1f, be)),
        fadd(c2d, mont_mul(c2f, be)),
        c3d,
    ]
}
/// multiply EF by base-field scalar given as canonical word
#[inline]
pub fn ef_scale(a: &Ef, x_canon: u32) -> Ef {
    if x_canon == 1 {
        return *a;
    }
    let xr = encode(x_canon);
    [mont_mul(a[0], xr), mont_mul(a[1], xr), mont_mul(a[2], xr), mont_mul(a[3], xr)]
}
/// openvm ExpPowers2: [z, z^2, z^4, z^8, ...] (no leading 1)
pub fn exp_powers_of_2(z: &Ef, n: usize) -> Vec<Ef> {
    let mut v = Vec::with_capacity(n);
    if n == 0 {
        return v;
    }
    // openvm ExpPowers2: sequence starts at z (no leading 1): z, z^2, z^4, ...
    v.push(*z);
    let mut cur = *z;
    for _ in 1..n {
        cur = ef_mul(&cur, &cur);
        v.push(cur);
    }
    v
}

// ---------------- Duplex challenger (p3 DuplexChallenger semantics) ----------------
pub struct Duplex<'a> {
    pub st: [u32; 16],
    pub ib: Vec<u32>,
    pub ob: Vec<u32>,
    pub p2: &'a Poseidon2,
}
impl<'a> Duplex<'a> {
    pub fn new(p2: &'a Poseidon2) -> Self {
        Self { st: [0u32; 16], ib: Vec::new(), ob: Vec::new(), p2 }
    }
    pub fn duplexing(&mut self) {
        for (i, v) in self.ib.drain(..).enumerate() {
            self.st[i] = v;
        }
        self.p2.mix(&mut self.st);
        self.ob.clear();
        self.ob.extend_from_slice(&self.st[..RATE]);
    }
    /// observe canonical word (as the recorder sees it)
    pub fn observe(&mut self, w_canon: u32) {
        self.ib.push(encode(w_canon));
        self.ob.clear();
        if self.ib.len() == RATE {
            self.duplexing();
        }
    }
    pub fn sample(&mut self) -> u32 {
        if !self.ib.is_empty() || self.ob.is_empty() {
            self.duplexing();
        }
        let v = self.ob.pop().unwrap();
        decode(v)
    }
    pub fn sample_bits(&mut self, bits: usize) -> usize {
        let v = self.sample();
        (v as u64 & ((1u64 << bits) - 1)) as usize
    }
    /// base state for grind scan: [input_buffer || sponge_state]
    pub fn grind_base(&self) -> ([u32; 16], usize) {
        let mut base = self.st;
        for (i, v) in self.ib.iter().enumerate() {
            base[i] = *v;
        }
        (base, self.ib.len())
    }
    /// grind check word (canonical) for candidate witness: perm(base with w at idx)[7]
    pub fn grind_check(p2: &Poseidon2, base: &[u32; 16], idx: usize, w_canon: u32) -> u32 {
        let mut st = *base;
        st[idx] = encode(w_canon);
        p2.mix(&mut st);
        decode(st[RATE - 1])
    }
    /// deterministic serial min-winner scan
    pub fn grind_min(&self, p2: &Poseidon2, bits: usize) -> (u32, u32, usize) {
        let (base, idx) = self.grind_base();
        let mut w = 0u32;
        loop {
            let c = Self::grind_check(p2, &base, idx, w);
            if mask_ok(c, bits) {
                return (w, c, idx);
            }
            w += 1;
            assert!(w < P, "grind: no winner < P");
        }
    }
}

// ---------------- merkle (WHIR EF-tree recipe) ----------------
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T: Copy> SendPtr<T> {
    #[inline]
    fn put(&self, i: usize, v: T) {
        unsafe { *self.0.add(i) = v };
    }
}
/// leaf: PaddingFreeSponge(16,8,8) over row words (raw), zero-padded.
pub fn merkle_leaf(p2: &Poseidon2, row: &[u32]) -> [u32; 8] {
    let mut st = [0u32; 16];
    st[..row.len()].copy_from_slice(row);
    p2.mix(&mut st);
    let mut d = [0u32; 8];
    d.copy_from_slice(&st[..8]);
    d
}
/// compress: TruncatedPermutation<2,8,16>: perm([L||R])[..8] (raw)
pub fn merkle_compress(p2: &Poseidon2, l: &[u32; 8], r: &[u32; 8]) -> [u32; 8] {
    let mut st = [0u32; 16];
    st[..8].copy_from_slice(l);
    st[8..].copy_from_slice(r);
    p2.mix(&mut st);
    let mut d = [0u32; 8];
    d.copy_from_slice(&st[..8]);
    d
}

/// Full WHIR EF merkle over rows (each 4 raw words). rows.len() pow2, rpq pow2.
/// Returns (leaf digests, root).
pub fn merkle_ef_rows(rows: &[[u32; 4]], rpq: usize) -> (Vec<[u32; 8]>, [u32; 8]) {
    let n = rows.len();
    assert!(n.is_power_of_two() && rpq.is_power_of_two() && rpq <= n);
    let nt = std::thread::available_parallelism().map(|x| x.get().min(16)).unwrap_or(8);
    let mut layer: Vec<[u32; 8]> = vec![[0u32; 8]; n];
    {
        let chunk = n / nt;
        std::thread::scope(|s| {
            let ptr = SendPtr(layer.as_mut_ptr());
            let mut handles = Vec::new();
            for t in 0..nt {
                let lo = t * chunk;
                let hi = if t + 1 == nt { n } else { (t + 1) * chunk };
                let rows = &rows[..];
                handles.push(s.spawn(move || {
                    let p2 = Poseidon2::new();
                    for i in lo..hi {
                        let d = merkle_leaf(&p2, &rows[i]);
                        ptr.put(i, d);
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
    }
    let leafdig = layer.clone();
    // phase-1: query-stride-paired layers (exactly log2(rpq)), qs constant = n/rpq
    let qs = n / rpq;
    let mut size = n;
    for _ in 0..rpq.trailing_zeros() {
        let nsz = size / 2;
        let mut next: Vec<[u32; 8]> = vec![[0u32; 8]; nsz];
        let nt2 = nt.min(nsz.max(1));
        let chunk = nsz / nt2;
        let prev = std::mem::replace(&mut layer, Vec::new());
        std::thread::scope(|s| {
            let nn = SendPtr(next.as_mut_ptr());
            let mut handles = Vec::new();
            for t in 0..nt2 {
                let lo = t * chunk;
                let hi = if t + 1 == nt2 { nsz } else { (t + 1) * chunk };
                let prev = &prev[..];
                handles.push(s.spawn(move || {
                    let p2 = Poseidon2::new();
                    for i in lo..hi {
                        let x = i / qs;
                        let y = i % qs;
                        let d = merkle_compress(&p2, &prev[2 * x * qs + y], &prev[(2 * x + 1) * qs + y]);
                        nn.put(i, d);
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
        layer = next;
        size = nsz;
    }
    // phase-2: adjacent-pair binary layers
    while size > 1 {
        let nsz = size / 2;
        let mut next: Vec<[u32; 8]> = vec![[0u32; 8]; nsz];
        let nt2 = nt.min(nsz);
        let chunk = nsz / nt2;
        let prev = std::mem::replace(&mut layer, Vec::new());
        std::thread::scope(|s| {
            let nn = SendPtr(next.as_mut_ptr());
            let mut handles = Vec::new();
            for t in 0..nt2 {
                let lo = t * chunk;
                let hi = if t + 1 == nt2 { nsz } else { (t + 1) * chunk };
                let prev = &prev[..];
                handles.push(s.spawn(move || {
                    let p2 = Poseidon2::new();
                    for i in lo..hi {
                        let d = merkle_compress(&p2, &prev[2 * i], &prev[2 * i + 1]);
                        nn.put(i, d);
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });
        layer = next;
        size = nsz;
    }
    let root = layer[0];
    (leafdig, root)
}

// ---------------- exact field helpers ----------------
fn pow_canon(mut base: u64, mut e: u64) -> u64 {
    let mut r = 1u64;
    base %= P as u64;
    while e > 0 {
        if e & 1 == 1 {
            r = r * base % (P as u64);
        }
        base = base * base % (P as u64);
        e >>= 1;
    }
    r
}
fn pow_raw(base_raw: u32, mut e: u64) -> u32 {
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
/// primitive 2^log_n-th root, raw domain (g=31)
pub fn root_of_unity_raw(log_n: u32) -> u32 {
    assert!(log_n <= 27);
    // p3 two_adic_generator: 31^(15 * 2^(27-log_n)) - full (p-1)/2^n exponent
    let e = 15u64 * ((1u64 << 27) >> log_n);
    encode(pow_canon(31, e) as u32)
}
/// in-place radix-2 DIT forward DFT: a = coefficients (raw) -> evaluations at <w_n>
pub fn fft_inplace(a: &mut [u32]) {
    let n = a.len();
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - log_n);
        let j = j as usize;
        if j > i {
            a.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let wm = pow_raw(root_of_unity_raw(log_n), (n / len) as u64);
        let half = len / 2;
        let mut wlen = encode(1);
        let mut k = 0usize;
        while k < half {
            let wr = wlen;
            let mut j = k;
            while j < n {
                let u = a[j];
                let t = mont_mul(a[j + half], wr);
                a[j] = fadd(u, t);
                a[j + half] = fsub(u, t);
                j += len;
            }
            wlen = mont_mul(wlen, wm);
            k += 1;
        }
        len *= 2;
    }
}
/// inverse DFT (coeffs <- evals), raw domain: forward DFT with w^{-1} / n
pub fn ifft_inplace(a: &mut [u32]) {
    let n = a.len();
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - log_n);
        let j = j as usize;
        if j > i {
            a.swap(i, j);
        }
    }
    let winv = pow_raw(root_of_unity_raw(log_n), (1u64 << log_n) - 1);
    let mut len = 2usize;
    while len <= n {
        let wm = pow_raw(winv, (n / len) as u64);
        let half = len / 2;
        let mut wlen = encode(1);
        let mut k = 0usize;
        while k < half {
            let wr = wlen;
            let mut j = k;
            while j < n {
                let u = a[j];
                let t = mont_mul(a[j + half], wr);
                a[j] = fadd(u, t);
                a[j + half] = fsub(u, t);
                j += len;
            }
            wlen = mont_mul(wlen, wm);
            k += 1;
        }
        len *= 2;
    }
    let ninv = encode(pow_canon(n as u64, (P as u64) - 2) as u32);
    for x in a.iter_mut() {
        *x = mont_mul(*x, ninv);
    }
}
/// p3 mobius_transform: multilinear evals (on {0,1}^k) -> coeffs
pub fn mobius_inplace(a: &mut [u32]) {
    let n = a.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let lo = a[j];
                let hi = a[j + h];
                a[j + h] = fsub(hi, lo);
            }
        }
        h *= 2;
    }
}
pub fn sha_u32(v: &[u32]) -> String {
    sha256_hex(&crate::field::u32s_to_bytes(v))
}
