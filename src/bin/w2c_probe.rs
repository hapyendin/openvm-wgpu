//! W2C probe: find why app-segments chain diverges at first sample (ev78).
use wgpu_backend::field::{Poseidon2};
use wgpu_backend::w2c::Duplex;

fn load(dir: &str, tag: &str) -> (Vec<u32>, Vec<bool>) {
    let f = tag.replace('/', "_");
    let raw = std::fs::read(format!("{dir}/{f}.prove.bin")).unwrap();
    let fl = std::fs::read(format!("{dir}/{f}.prove.bin.flags")).unwrap();
    let n = raw.len() / 8;
    let vals: Vec<u32> = (0..n).map(|i| u32::from_le_bytes(raw[8 * i..8 * i + 4].try_into().unwrap())).collect();
    let flags: Vec<bool> = (0..n).map(|i| (fl[i >> 3] >> (i & 7)) & 1 == 1).collect();
    (vals, flags)
}

#[derive(Clone)]
enum Edit {
    None,
    Skip(usize),
    Perm(usize),
    Ob0(usize),
    ObDup(usize),
    Flip(usize),
    Glue(Vec<u32>, bool),
    Raw(usize, usize),
}

fn observe_raw(d: &mut Duplex, w: u32) {
    d.ib.push(w);
    d.ob.clear();
    if d.ib.len() == 8 {
        d.duplexing();
    }
}

// replay whole stream with one edit; collect first derived samples at/after ev 70
fn run(p2: &Poseidon2, vals: &[u32], flags: &[bool], edit: &Edit) -> Vec<(usize, u32)> {
    let mut d = Duplex::new(p2);
    if let Edit::Glue(g, r) = edit {
        for g in g {
            if *r {
                observe_raw(&mut d, *g);
            } else {
                d.observe(*g);
            }
        }
    }
    let mut out = Vec::new();
    for i in 0..vals.len() {
        let skip = matches!(edit, Edit::Skip(k) if *k == i);
        let perm = matches!(edit, Edit::Perm(k) if *k == i);
        let ob0 = matches!(edit, Edit::Ob0(k) if *k == i);
        let obd = matches!(edit, Edit::ObDup(k) if *k == i);
        let flip = matches!(edit, Edit::Flip(k) if *k == i);
        let rawr = matches!(edit, Edit::Raw(a, b) if i >= *a && i < *b);
        if perm {
            d.duplexing();
        }
        if skip {
            continue;
        }
        if ob0 {
            d.observe(0);
        }
        if flip {
            let v = d.sample();
            if i >= 70 && out.len() < 4 {
                out.push((i, v));
            }
        } else if flags[i] {
            let v = d.sample();
            if i >= 70 && out.len() < 4 {
                out.push((i, v));
            }
        } else {
            if rawr {
                observe_raw(&mut d, vals[i]);
            } else {
                d.observe(vals[i]);
            }
            if obd {
                d.observe(vals[i]);
            }
        }
    }
    out
}

fn main() {
    let dir = "data/b9/w2c/streams-r1";
    let (av, af) = load(dir, "app-segments_0");
    let (gv, gf) = load(dir, "agg-leaf_0");
    let p2 = Poseidon2::new();
    let target = av[78];
    println!("targets: app e78={target} agg e161={}", gv[161]);
    let b = run(&p2, &gv, &gf, &Edit::None);
    println!("BASE agg first2 {:?}", &b[..2.min(b.len())]);
    let b = run(&p2, &av, &af, &Edit::None);
    println!("BASE app first2 {:?}", &b[..2.min(b.len())]);
    // template alignment: contiguous match app[17..78] inside agg window, app[9..17] (pv8), app[0..8] (pfx)
    for off in 0..200usize {
        if off + 61 <= gv.len() && gv[off..off + 61] == av[17..78] {
            println!("TEMPLATE: app[17..78] == agg[{off}..{}]", off + 61);
        }
        if off + 8 <= gv.len() && gv[off..off + 8] == av[9..17] {
            println!("PV8: app[9..17] == agg[{off}..{}]", off + 8);
        }
        if off + 8 <= gv.len() && gv[off..off + 8] == av[0..8] {
            println!("PFX: app[0..8] == agg[{off}..{}]", off + 8);
        }
    }
    let mut variants: Vec<(String, Edit)> = Vec::new();
    for i in 0..80 {
        variants.push((format!("skip{i}"), Edit::Skip(i)));
        variants.push((format!("perm{i}"), Edit::Perm(i)));
        variants.push((format!("ob0_{i}"), Edit::Ob0(i)));
        variants.push((format!("obdup{i}"), Edit::ObDup(i)));
        variants.push((format!("flip{i}"), Edit::Flip(i)));
    }
    variants.push(("raw0_8".into(), Edit::Raw(0, 8)));
    variants.push(("raw0_78".into(), Edit::Raw(0, 78)));
    variants.push(("raw8_78".into(), Edit::Raw(8, 78)));
    let pvhex = "5ed5d617b86a99669d75acc461de11bfc226b046797a20f8b2282331955344d4";
    let glue_be: Vec<u32> = (0..8).map(|i| u32::from_str_radix(&pvhex[i * 8..i * 8 + 8], 16).unwrap()).collect();
    let glue_le: Vec<u32> = glue_be.iter().map(|w| w.swap_bytes()).collect();
    let vkapp: Vec<u32> = vec![93085625, 540830993, 1888218509, 927761177, 1083260992, 464140180, 402247195, 690334738];
    variants.push(("glueVKapp".into(), Edit::Glue(vkapp, false)));
    variants.push(("glueBE".into(), Edit::Glue(glue_be.clone(), false)));
    variants.push(("glueLE".into(), Edit::Glue(glue_le.clone(), false)));
    variants.push(("glueBEraw".into(), Edit::Glue(glue_be, true)));
    variants.push(("glueLEraw".into(), Edit::Glue(glue_le, true)));
    let mut hits = 0;
    for (name, ed) in variants {
        let r = run(&p2, &av, &af, &ed);
        if r.first().map(|(_, v)| *v == target).unwrap_or(false) {
            println!("HIT {name}: {:?}", &r[..2.min(r.len())]);
            hits += 1;
        }
    }
    println!("single-edit hits: {hits}");
}
