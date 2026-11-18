// scan: find stream window z such that Horner(gc, z) == stream words at z+4 (OOD y)
use wgpu_backend::field::{decode, encode};
use wgpu_backend::w2c::{ef_add, ef_mul, Ef};

fn read_ef(p: &str) -> Vec<[u32; 4]> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(16)
        .map(|c| {
            [
                u32::from_le_bytes(c[0..4].try_into().unwrap()),
                u32::from_le_bytes(c[4..8].try_into().unwrap()),
                u32::from_le_bytes(c[8..12].try_into().unwrap()),
                u32::from_le_bytes(c[12..16].try_into().unwrap()),
            ]
        })
        .collect()
}
fn stream_words(p: &str) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    (0..b.len() / 8)
        .map(|i| u32::from_le_bytes(b[8 * i..8 * i + 4].try_into().unwrap()))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let inst: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(0);
    let names = [
        "app-segments_0",
        "agg-leaf_0",
        "agg-internal-for-leaf_0",
        "agg-internal-recursive_0",
        "wrap-internal-recursive.1_0",
    ];
    let sv = stream_words(&format!(
        "/home/matt/appz/openvm-sycl-skel/b9/w2c/streams-r1/{}.prove.bin",
        names[inst]
    ));
    let mut gc = read_ef(&format!(
        "/home/matt/appz/openvm-sycl-skel/b9/r15/w2c-r1/whir/w{inst}-r0-gcoeffs.bin"
    ));
    // true coeffs = first 2^(m-k): derive from meta m,k
    let meta = std::fs::read_to_string(format!(
        "/home/matt/appz/openvm-sycl-skel/b9/r15/w2c-r1/whir/w{inst}-meta.txt"
    ))
    .unwrap();
    let mp = |k: &str| -> usize {
        meta.split_whitespace()
            .find(|t| t.starts_with(&format!("{k}=")))
            .unwrap()
            .split('=')
            .nth(1)
            .unwrap()
            .parse()
            .unwrap()
    };
    let kdim = mp("m") - mp("k_whir");
    gc.truncate(1usize << kdim);
    println!("gc {} meta m={} k={}", gc.len(), mp("m"), mp("k_whir"));
    let gc_raw: Vec<Ef> = gc.iter().map(|r| r.map(encode)).collect();
    let n = sv.len();
    let lo = 100usize;
    let hi = n - 8;
    let nthreads = 8usize;
    let handles: Vec<_> = (0..nthreads)
        .map(|t| {
            let gc = gc_raw.clone();
            let sv = sv.clone();
            std::thread::spawn(move || {
                let mut out: Vec<(usize, [u32; 4])> = Vec::new();
                let mut e = lo + t;
                while e < hi {
                    let z: Ef = [
                        encode(sv[e]),
                        encode(sv[e + 1]),
                        encode(sv[e + 2]),
                        encode(sv[e + 3]),
                    ];
                    // Horner high->low
                    let mut acc = gc[gc.len() - 1];
                    for c in gc[..gc.len() - 1].iter().rev() {
                        acc = ef_add(&ef_mul(&acc, &z), c);
                    }
                    let cc = acc.map(decode);
                    if cc == [sv[e + 4], sv[e + 5], sv[e + 6], sv[e + 7]] {
                        out.push((e, cc));
                    }
                    if e >= 4 && cc == [sv[e - 4], sv[e - 3], sv[e - 2], sv[e - 1]] {
                        out.push((e, cc));
                    }
                    e += nthreads;
                }
                out
            })
        })
        .collect();
    for h in handles {
        for (e, y) in h.join().unwrap() {
            println!("HIT z0@{e} y={y:?}");
        }
    }
    println!("scan done");
}
