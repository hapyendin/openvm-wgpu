// brute: for candidate z0 positions/orders, evaluate dump gcoeffs at z0 and compare stream y
use wgpu_backend::field::{decode, encode};
use wgpu_backend::w2c::{ef_add, ef_mul, ef_sub, Ef};

fn canon_ef(rows: &[Ef]) -> Vec<u32> {
    rows.iter().flat_map(|r| r.map(decode)).collect()
}
fn read_ef(p: &str) -> Vec<[u32; 4]> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(16).map(|c| [
        u32::from_le_bytes(c[0..4].try_into().unwrap()),
        u32::from_le_bytes(c[4..8].try_into().unwrap()),
        u32::from_le_bytes(c[8..12].try_into().unwrap()),
        u32::from_le_bytes(c[12..16].try_into().unwrap()),
    ]).collect()
}

fn stream_words(p: &str) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    (0..b.len() / 8).map(|i| u32::from_le_bytes(b[8 * i..8 * i + 4].try_into().unwrap())).collect()
}

fn eval(gc: &[[u32; 4]], z: &[u32; 4], revzp: bool) -> [u32; 4] {
    // gc canon rows; z canon EF limbs
    let n = gc.len();
    let nv = n.trailing_zeros() as usize;
    let zf = z.map(encode);
    let mut zp: Vec<Ef> = Vec::with_capacity(nv);
    zp.push(zf);
    let mut cur = zf;
    for _ in 1..nv {
        cur = ef_mul(&cur, &cur);
        zp.push(cur);
    }
    if revzp { zp.reverse(); }
    let mut poly: Vec<Ef> = gc.iter().map(|r| r.map(encode)).collect();
    for j in 0..nv {
        let h = 1usize << j;
        for i2 in (0..n).step_by(h * 2) {
            for k2 in i2..i2 + h {
                let c0 = poly[k2];
                let d = ef_sub(&poly[k2 + h], &c0);
                poly[k2] = ef_add(&c0, &ef_mul(&d, &zp[j]));
            }
        }
    }
    let c = canon_ef(&[poly[0]]);
    [c[0], c[1], c[2], c[3]]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let inst = args.get(1).map(|s| s.as_str()).unwrap_or("0");
    let evz: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(13793);
    let stream_file = format!("data/b9/w2c/streams-r1/{}.prove.bin", ["app-segments_0","agg-leaf_0","agg-internal-for-leaf_0","agg-internal-recursive_0","wrap-internal-recursive.1_0"][inst.parse::<usize>().unwrap()]);
    let sv = stream_words(&stream_file);
    let mut gc = read_ef(&format!("data/b9/r15/w2c-r1/whir/w{inst}-r0-gcoeffs.bin"));
    if gc.len() > (1 << 20) { gc.truncate(1 << 20); }
    println!("gc rows {} evz {evz}", gc.len());
    for d in [-16i64, -12, -8, -4, 0, 4, 8, 12] {
        let e = (evz as i64 + d) as usize;
        for ord in 0..4 {
            let mut z = [sv[e], sv[e + 1], sv[e + 2], sv[e + 3]];
            if ord == 1 {
                z.reverse();
            }
            let y = eval(&gc, &z, ord >= 2);
            let t = [sv[evz + 4], sv[evz + 5], sv[evz + 6], evz as u32 + 7];
            let _ = t;
            let ymatch = (0..4).all(|j| sv[e + 4 + j] == y[j]);
            println!("evz {} ord {} z{:?} -> y{:?} {}", e, ord, z, y, if ymatch { "MATCH-STREAM" } else { "" });
        }
    }
}
