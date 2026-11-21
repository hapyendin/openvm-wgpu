//! W2C grs probe: compare my fold->mobius->pad->fft vs WHIR_DUMP grs; test bitrev convention.
use wgpu_backend::field::{decode, encode, Poseidon2};
use wgpu_backend::w2c::*;

fn read_u32s(p: &str) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn raw_ef(c: &[u32]) -> Vec<Ef> {
    c.chunks_exact(4).map(|x| [encode(x[0]), encode(x[1]), encode(x[2]), encode(x[3])]).collect()
}

fn canon_ef1(e: &[u32]) -> Vec<u32> { e.to_vec() }
fn read_u32s_raw_u64(p: &str) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    (0..b.len()/8).map(|i| u32::from_le_bytes(b[8*i..8*i+4].try_into().unwrap())).collect()
}
fn powu64(mut b: u64, mut e: u64) -> u64 {
    let mut r = 1u64;
    while e > 0 { if e & 1 == 1 { r = r * b % 2013265921; } b = b * b % 2013265921; e >>= 1; }
    r
}


// corrected two-adic root: 31^(15*2^(27-log_n))
fn root_fixed(log_n: u32) -> u32 {
    let e = 15u64 * ((1u64 << 27) >> log_n);
    let mut b = 31u64; let mut r = 1u64; let mut e = e;
    while e > 0 { if e & 1 == 1 { r = r * b % 2013265921; } b = b * b % 2013265921; e >>= 1; }
    encode(r as u32)
}
fn powr(mut b: u32, mut e: u64) -> u32 {
    let mut r = encode(1u32);
    while e > 0 { if e & 1 == 1 { r = wgpu_backend::field::mont_mul(r, b); } b = wgpu_backend::field::mont_mul(b, b); e >>= 1; }
    r
}
fn fft_fixed(a: &mut [u32]) {
    let n = a.len();
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = ((i as u32).reverse_bits() >> (32 - log_n)) as usize;
        if j > i { a.swap(i, j); }
    }
    let wn = root_fixed(log_n);
    let mut len = 2usize;
    while len <= n {
        let wm = powr(wn, (n / len) as u64);
        let half = len / 2;
        let mut wlen = encode(1u32);
        let mut k = 0usize;
        while k < half {
            let wr = wlen;
            let mut j = k;
            while j < n {
                let u = a[j];
                let t = wgpu_backend::field::mont_mul(a[j + half], wr);
                a[j] = wgpu_backend::field::fadd(u, t);
                a[j + half] = wgpu_backend::field::fsub(u, t);
                j += len;
            }
            wlen = wgpu_backend::field::mont_mul(wlen, wm);
            k += 1;
        }
        len *= 2;
    }
}
fn main() {
    let d = "data/b9/r15/w2c-r1/whir";
    let tag = std::env::args().nth(1).unwrap_or("0".into());
    let round = std::env::args().nth(2).unwrap_or("0".into());
    let k_whir: usize;
    let log_rs0: usize;
    let meta = std::fs::read_to_string(format!("{d}/w{tag}-meta.txt")).unwrap();
    let mp = |k: &str| -> usize {
        meta.split_whitespace().find(|t| t.starts_with(&format!("{k}="))).unwrap()[k.len() + 1..].parse().unwrap()
    };
    k_whir = mp("k_whir");
    log_rs0 = mp("log_rs0");
    let _ = mp("m");
    let f = raw_ef(&read_u32s(&format!("{d}/w{tag}-f0.bin")));
    let alphas_c = read_u32s(&format!("{d}/w{tag}-alphas.bin"));
    let alphas = raw_ef(&alphas_c);
    let r_i: usize = round.parse().unwrap();
    let mut f = f;
    for i in 0..(r_i + 1) * k_whir {
        let a = alphas[i];
        let half = f.len() / 2;
        let mut nf = Vec::with_capacity(half);
        for y in 0..half {
            let f0 = f[2 * y];
            let df = ef_sub(&f[2 * y + 1], &f0);
            nf.push(ef_add(&f0, &ef_mul(&df, &a)));
        }
        f = nf;
    }
    for l in 0..4 {
        let mut col: Vec<u32> = f.iter().map(|e| e[l]).collect();
        mobius_inplace(&mut col);
        for (i, v) in col.iter().enumerate() {
            f[i][l] = *v;
        }
    }
    let n_fft = 1usize << (log_rs0 - 1 - r_i);
    println!("dim {} -> fft {}", f.len(), n_fft);
    let logn_pre = n_fft.trailing_zeros();
    let mut f_pre = f.clone();
    f_pre.resize(n_fft, ef_zero());
    let _ = &f_pre;
    f = f_pre.clone();
    for l in 0..4 {
        let mut col: Vec<u32> = f.iter().map(|e| e[l]).collect();
        fft_inplace(&mut col);
        for (i, v) in col.iter().enumerate() {
            f[i][l] = *v;
        }
    }
    // dump gcoeffs (padded, canonical) sample-compare
    let gcco = read_u32s(&format!("{d}/w{tag}-r{round}-gcoeffs.bin"));
    let c0 = |i: usize| -> [u32; 4] { let e = f_pre[i]; [decode(e[0]), decode(e[1]), decode(e[2]), decode(e[3])] };
    let mut cid = 0; let mut cbr = 0;
    for t in 0..4096usize {
        let i = (t * 9301 + 49297) % n_fft;
        let g = [gcco[4*i], gcco[4*i+1], gcco[4*i+2], gcco[4*i+3]];
        if c0(i) == g { cid += 1; }
        let br = ((i as u32).reverse_bits() >> (32 - logn_pre)) as usize;
        if c0(br) == g { cbr += 1; }
    }
    println!("coeffs: identity={cid} bitrev={cbr}");
    // dump grs (canonical words)
    let gcan = read_u32s(&format!("{d}/w{tag}-r{round}-grs.bin"));
    // variant: ifft with/without 1/n
    let mut fi = f_pre.clone();
    for l in 0..4 {
        let mut col: Vec<u32> = fi.iter().map(|e| e[l]).collect();
        ifft_inplace(&mut col);
        for (i, v) in col.iter().enumerate() { fi[i][l] = *v; }
    }
    let nraw = encode(n_fft as u32);
    let fi_scale = |i: usize| -> [u32; 4] {
        let e = fi[i];
        [wgpu_backend::field::mont_mul(e[0], nraw), wgpu_backend::field::mont_mul(e[1], nraw),
         wgpu_backend::field::mont_mul(e[2], nraw), wgpu_backend::field::mont_mul(e[3], nraw)]
    };
    let mut id_i = 0; let mut id_s = 0; let mut br_i = 0;
    let mut f2 = f_pre.clone();
    for l in 0..4 {
        let mut col: Vec<u32> = f2.iter().map(|e| e[l]).collect();
        fft_fixed(&mut col);
        for (i, v) in col.iter().enumerate() { f2[i][l] = *v; }
    }
    let mut fixid = 0usize; let mut fixbr = 0usize;
    for t in 0..4096usize {
        let i = (t * 9301 + 49297) % n_fft;
        let g = [gcan[4*i], gcan[4*i+1], gcan[4*i+2], gcan[4*i+3]];
        let e = f2[i]; let a = [decode(e[0]), decode(e[1]), decode(e[2]), decode(e[3])];
        if a == g { fixid += 1; }
        let br = ((i as u32).reverse_bits() >> (32 - logn_pre)) as usize;
        let e = f2[br]; let a = [decode(e[0]), decode(e[1]), decode(e[2]), decode(e[3])];
        if a == g { fixbr += 1; }
    }
    println!("FIXED root: identity={fixid} bitrev={fixbr}");
    let mut ratios = std::collections::BTreeMap::new();
    for t in 0..512usize {
        let i = (t * 9301 + 49297) % n_fft;
        let g = [gcan[4*i], gcan[4*i+1], gcan[4*i+2], gcan[4*i+3]];
        let a = fi[i]; let a = [decode(a[0]), decode(a[1]), decode(a[2]), decode(a[3])];
        if a == g { id_i += 1; }
        let b = fi_scale(i);
        if b == g { id_s += 1; }
        if g[0] != 0 { let mine0 = decode(f[i][0]); if mine0 != 0 {
            let inv = powu64(mine0 as u64, 2013265919);
            let r = ((g[0] as u64) * inv % 2013265921) as u32;
            *ratios.entry(r).or_insert(0u32) += 1; } }
    }
    println!("ifft: id={id_i} id_scaled={id_s} ratios_top={:?}", ratios.into_iter().collect::<Vec<_>>().into_iter().take(6).collect::<Vec<_>>());
    let mine = |i: usize| -> [u32; 4] { [decode(f[i][0]), decode(f[i][1]), decode(f[i][2]), decode(f[i][3])] };
    let logn = n_fft.trailing_zeros();
    let mut ident = 0usize;
    let mut bitrev = 0usize;
    let mut revord = 0usize;
    let mut sampled = 0usize;
    for t in 0..4096usize {
        let i = (t * 9301 + 49297) % n_fft; // pseudo-random sample
        let br = (i as u32).reverse_bits() >> (32 - logn);
        let br = br as usize;
        sampled += 1;
        let g = [gcan[4 * i], gcan[4 * i + 1], gcan[4 * i + 2], gcan[4 * i + 3]];
        if mine(i) == g { ident += 1; }
        if mine(br) == g { bitrev += 1; }
        if mine((n_fft - i) % n_fft) == g { revord += 1; }
    }
    println!("sampled={sampled} identity={ident} bitrev={bitrev} revord={revord}");
    // OOD direct test: z0 at stream ev, y at ev+4 (app r0: ev=13793)
    if tag == "0" {
        let sd = "data/b9/w2c/streams-r1";
        let sv = read_u32s_raw_u64(&format!("{sd}/app-segments_0.prove.bin"));
        let evz = 13793usize;
        let z0 = [encode(sv[evz]), encode(sv[evz+1]), encode(sv[evz+2]), encode(sv[evz+3])];
        let dimc = 20usize;
        // exp powers of 2 of z0 (EF)
        let mut zp: Vec<Ef> = Vec::with_capacity(dimc);
        zp.push([encode(1),0,0,0]);
        zp.push(z0);
        let mut cur = z0;
        for _ in 2..dimc { cur = ef_mul(&cur,&cur); zp.push(cur); }
        let mut poly: Vec<Ef> = f_pre.iter().take(1<<dimc).cloned().collect();
        for j in 0..dimc {
            let h = 1usize << j;
            for i2 in (0..poly.len()).step_by(h*2) {
                for k2 in i2..i2+h {
                    let c0 = poly[k2];
                    let d = ef_sub(&poly[k2+h], &c0);
                    poly[k2] = ef_add(&c0, &ef_mul(&d, &zp[j]));
                }
            }
        }
        let y = canon_ef1(&poly[0]);
        let ycan = canon_ef1(&[decode(z0[0]),decode(z0[1]),decode(z0[2]),decode(z0[3])]);
        println!("z0 canon {:?}", &ycan[..]);
        println!("OOD mine {y:?} stream {:?}", &sv[evz+4..evz+8]);
    }
    {
        use std::collections::HashMap;
        let mut map: HashMap<u128, usize> = HashMap::with_capacity(n_fft);
        for i in 0..n_fft {
            let e = f[i];
            let k = (e[0] as u128) | ((e[1] as u128) << 31) | ((e[2] as u128) << 62) | ((e[3] as u128) << 93);
            map.entry(k).or_insert(i);
        }
        for j in 0..8usize {
            let g = ([gcan[4*j], gcan[4*j+1], gcan[4*j+2], gcan[4*j+3]]);
            let e = [encode(g[0]) as u128, encode(g[1]) as u128, encode(g[2]) as u128, encode(g[3]) as u128];
            let k = e[0] | (e[1] << 31) | (e[2] << 62) | (e[3] << 93);
            println!("dump[{j}] -> mine idx {:?}", map.get(&k));
        }
    }
    let g0 = [gcan[0], gcan[1], gcan[2], gcan[3]];
    println!("mine[0]={:?} dump[0]={:?} mine[1]={:?} dump[br(1)]={:?}", mine(0), g0, mine(1), [gcan[4 * ((1usize).reverse_bits() >> (32 - logn))], gcan[4 * ((1usize).reverse_bits() >> (32 - logn)) + 1], gcan[4 * ((1usize).reverse_bits() >> (32 - logn)) + 2], gcan[4 * ((1usize).reverse_bits() >> (32 - logn)) + 3]]);
}
