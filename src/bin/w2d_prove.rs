//! GATE W2D: single-process full-GPU prove of app-segments/0 on the W2D-R1 world.
//! One process: duplex transcript engine (live challenges) -> GPU leaf build ->
//! GPU logup GKR rounds -> GPU grinds (serial-min) -> WHIR fold (W2C CPU fold
//! chain + GPU mobius/DIT-FFT/merkle commits + OOD) -> emitted proof stream.
//! Unported regions (prefix commits, constraint batch + FRI-on-quotient, query
//! openings) are passthrough-observed from the recorded same-run stream.
use std::path::{Path, PathBuf};

use wgpu_backend::device::{InitOpts, WgpuDevice};
use wgpu_backend::field::{constants_buffer_mont, decode, encode, fsub, mont_mul, sha256_hex, u32s_to_bytes, Poseidon2};
use wgpu_backend::gkr::{check_expected, check_snapshots, run_instance_gpu, GkrInputs, GkrPipeline};
use wgpu_backend::logup_leaf::{build_leaf, LeafPipeline, LeafWorld, TraceCache};
use wgpu_backend::w2c::*;
use wgpu_backend::limitsbuf::ShardBuf;
use wgpu_backend::whir_fold::{self, WhirFold};
use wgpu_backend::wgsl;

fn read_u32s(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|_| panic!("read {}", p.display()));
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn read_ef(p: &Path) -> Vec<Ef> {
    read_u32s(p).chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}
fn canon_ef(e: &[Ef]) -> Vec<u32> {
    e.iter().flat_map(|x| x.iter().map(|&w| decode(w))).collect()
}
fn raw_ef(c: &[u32]) -> Vec<Ef> {
    c.chunks_exact(4).map(|x| [encode(x[0]), encode(x[1]), encode(x[2]), encode(x[3])]).collect()
}
fn raw_ef4(b: &[u8]) -> Ef {
    [
        encode(u32::from_le_bytes(b[0..4].try_into().unwrap())),
        encode(u32::from_le_bytes(b[4..8].try_into().unwrap())),
        encode(u32::from_le_bytes(b[8..12].try_into().unwrap())),
        encode(u32::from_le_bytes(b[12..16].try_into().unwrap())),
    ]
}
fn read_exact_all<R: std::io::Read>(r: &mut R, buf: &mut [u8]) {
    let mut f = 0usize;
    while f < buf.len() {
        match r.read(&mut buf[f..]) {
            Ok(0) => panic!("unexpected EOF"),
            Ok(n) => f += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => panic!("{e}"),
        }
    }
}
fn flatten(e: &[Ef]) -> Vec<u32> {
    e.iter().flat_map(|r| r.iter().copied()).collect()
}
fn unflatten(w: &[u32]) -> Vec<Ef> {
    w.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}
fn powm(mut b: u32, mut e: u64) -> u32 {
    let mut r = encode(1);
    while e > 0 {
        if e & 1 == 1 { r = mont_mul(r, b); }
        b = mont_mul(b, b);
        e >>= 1;
    }
    r
}
fn vm_hwm_mb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
        / 1024
}

const GRIND_WGSL: &str = r#"
@group(0) @binding(4) var<storage, read_write> amin: array<atomic<u32>>;
@compute @workgroup_size(256u)
fn grind_scan(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  if (w >= uni.a2) { return; }
  var cells: array<u32, 16u>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) { cells[i] = io1[i]; }
  cells[uni.seed] = encw(w);
  perm_mix(&cells);
  let c = decode(cells[7u]);
  if ((c & uni.a1) == 0u) { atomicMin(&amin[0], w); }
}
@compute @workgroup_size(1u)
fn grind_check(@builtin(global_invocation_id) gid: vec3<u32>) {
  var cells: array<u32, 16u>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) { cells[i] = io1[i]; }
  cells[uni.seed] = encw(uni.a2);
  perm_mix(&cells);
  io2[0] = decode(cells[7u]);
}
"#;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut world = PathBuf::from("data/b9/r15/w2d-r1");
    let mut inputs = PathBuf::from("w2d/w2d-inputs");
    let mut streams = PathBuf::from("w2d/streams-w2d");
    let mut out = PathBuf::from("w2d/app-segments_0.prove.bin.w2d");
    let mut report = PathBuf::from("w2d/w2d-report.json");
    let mut stop_at = String::from("all");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--world" => { i += 1; world = args[i].clone().into(); }
            "--inputs" => { i += 1; inputs = args[i].clone().into(); }
            "--streams" => { i += 1; streams = args[i].clone().into(); }
            "--out" => { i += 1; out = args[i].clone().into(); }
            "--report" => { i += 1; report = args[i].clone().into(); }
            "--stop-at" => { i += 1; stop_at = args[i].clone(); }
            other => eprintln!("unknown arg {other}"),
        }
        i += 1;
    }
    let tag = "app-segments_0";
    if let Some(repl) = args.iter().position(|x| x == "--replay") {
        let p = PathBuf::from(args[repl + 1].clone());
        let raw = std::fs::read(&p).unwrap();
        let n = raw.len() / 8;
        let vals: Vec<u32> = (0..n).map(|k| u32::from_le_bytes(raw[8 * k..8 * k + 4].try_into().unwrap())).collect();
        let fl = std::fs::read(format!("{}.flags", p.display())).unwrap();
        let flags: Vec<bool> = (0..n).map(|k| (fl[k >> 3] >> (k & 7)) & 1 == 1).collect();
        let p2 = Poseidon2::new();
        let glue: [u32; 8] = [93085625, 540830993, 1888218509, 927761177, 1083260992, 464140180, 402247195, 690334738];
        let mut d = Duplex::new(&p2);
        for g in glue { d.observe(g); }
        let site0 = [77usize, 13723];
        let mut mism = 0usize;
        let mut ns = 0usize;
        for ev in 0..n {
            if flags[ev] {
                let v = d.sample();
                ns += 1;
                if v != vals[ev] { mism += 1; if mism < 4 { println!("REPLAY-MISM {ev} want {} got {v}", vals[ev]); } }
            } else {
                d.observe(vals[ev]);
            }
        }
        let _ = site0;
        println!("REPLAY {}: samples={ns} mismatches={mism} -> {}", p.display(), if mism == 0 { "GREEN" } else { "RED" });
        if mism != 0 { std::process::exit(2); }
        return;
    }
    let raw = std::fs::read(streams.join(format!("{tag}.prove.bin"))).unwrap();
    let n = raw.len() / 8;
    let vals: Vec<u32> = (0..n).map(|k| u32::from_le_bytes(raw[8 * k..8 * k + 4].try_into().unwrap())).collect();
    let fl = std::fs::read(streams.join(format!("{tag}.prove.bin.flags"))).unwrap();
    let flags: Vec<bool> = (0..n).map(|k| (fl[k >> 3] >> (k & 7)) & 1 == 1).collect();
    let nsamp = flags.iter().filter(|f| **f).count();
    eprintln!("stream loaded n={n} samples={nsamp} vmhwm={}MB", vm_hwm_mb());

    // psites rows (app instances first, global order)
    let mut site_bits: Vec<usize> = Vec::new();
    let mut site_w: Vec<u32> = Vec::new();
    for l in std::fs::read_to_string(world.join("psites/pow_sites.jsonl")).unwrap().lines() {
        let v: serde_json::Value = serde_json::from_str(l).unwrap();
        site_bits.push(v["bits"].as_u64().unwrap() as usize);
        site_w.push(v["witness"].as_u64().unwrap() as u32);
    }
    // WHIR meta
    let meta_s = std::fs::read_to_string(world.join("whir/w0-meta.txt")).unwrap();
    let mp = |k: &str| -> usize {
        meta_s.split_whitespace().find(|t| t.starts_with(&format!("{k}="))).unwrap()[k.len() + 1..].parse().unwrap()
    };
    let k_whir = mp("k_whir");
    let log_rs0 = mp("log_rs0");
    let mu_bits = mp("mu_bits");
    let q_bits = mp("q_bits");
    let ulen = mp("ulen");
    let nq_v: Vec<usize> = meta_s.split("rounds=[").nth(1).unwrap().split(']').next().unwrap()
        .split(',').map(|x| x.trim().parse().unwrap()).collect();

    // mu witness position (single unambiguous scan)
    let mut mu_ev: Option<usize> = None;
    for ev in 6000..13800 {
        if !flags[ev] && flags[ev + 1] && mask_ok(vals[ev + 1], mu_bits) && (ev + 2..ev + 6).all(|x| flags[x]) {
            if mu_ev.is_some() { panic!("mu witness ambiguous"); }
            mu_ev = Some(ev);
        }
    }
    let mu_event = mu_ev.expect("mu witness position");
    // final query pow witness = before the trailing S-run of nq_last+5
    let fin_qpow = n - (nq_v[nq_v.len() - 1] + 5) - 1;
    eprintln!("mu witness event = {mu_event}, final qpow witness = {fin_qpow}");

    // skeleton walk: witness events in stream order
    let mut site_events: Vec<usize> = vec![77, mu_event];
    {
        let mut ev = mu_event + 6; // after witness+check+mu(4)
        for (r, &nq) in nq_v.iter().enumerate() {
            let is_last = r + 1 == nq_v.len();
            for _ in 0..k_whir {
                ev += 8;
                site_events.push(ev);
                ev += 2;
                ev += 4;
            }
            if !is_last {
                ev += 8 + 4 + 4; // commit, z0, ood
                if q_bits > 0 { site_events.push(ev); ev += 2; }
                ev += nq + 4;
            } else {
                ev += 4 * ulen; // finalcoeffs
                ev = fin_qpow;  // + query openings passthrough
                if q_bits > 0 { site_events.push(ev); ev += 2; }
                ev += nq + 4;
            }
        }
        eprintln!("fold skeleton end_ev={ev} (n={n})");
        assert_eq!(ev, n, "fold skeleton must consume the full stream");
    }
    assert_eq!(site_events.len(), 22);
    for (i2, e) in site_events.iter().enumerate() {
        assert!(!flags[*e], "site {i2} witness not O at {e}");
        assert_eq!(vals[*e], site_w[i2], "site {i2} witness != psites at {e}");
    }

    // ---------------- PASS 1: duplex transcript engine ----------------
    let p2 = Poseidon2::new();
    let glue: [u32; 8] = [93085625, 540830993, 1888218509, 927761177, 1083260992, 464140180, 402247195, 690334738];
    let mut d = Duplex::new(&p2);
    for g in glue { d.observe(g); }
    let mut derived = vec![0u32; n];
    let mut sample_mism: Vec<(usize, u32, u32)> = Vec::new();
    let mut bases: Vec<([u32; 16], usize)> = Vec::new();
    let mut si = 0usize;
    for ev in 0..n {
        if si < site_events.len() && site_events[si] == ev {
            bases.push(d.grind_base());
            si += 1;
            d.observe(vals[ev]);
        } else if flags[ev] {
            let v = d.sample();
            derived[ev] = v;
            if v != vals[ev] && sample_mism.len() < 10 { sample_mism.push((ev, vals[ev], v)); }
        } else {
            d.observe(vals[ev]);
        }
    }
    assert_eq!(si, site_events.len());
    let chain_ok = sample_mism.is_empty();
    eprintln!("CHAIN samples={nsamp} mismatches={} {:?}", sample_mism.len(), &sample_mism[..sample_mism.len().min(3)]);
    if stop_at == "transcript" {
        println!("W2D transcript-only: chain_ok={chain_ok} vmhwm={}MB", vm_hwm_mb());
        return;
    }

    // ---------------- GPU init ----------------
    let dev = WgpuDevice::init(InitOpts { require_intel_arc: true, allow_u32_only: false });
    eprintln!("ADAPTER {} vmhwm={}MB", dev.report.name, vm_hwm_mb());
    let pipe = GkrPipeline::new(&dev);
    let lp = LeafPipeline::new(&dev);
    let wf = WhirFold::new(&dev);

    // ---------------- GPU leaf build ----------------
    let world_lw: LeafWorld = serde_json::from_str(&std::fs::read_to_string(inputs.join("leafworld.json")).unwrap()).unwrap();
    let traces_dir = world.join("traces/app-segments_0");
    let mut cache = TraceCache::new();
    let (leaf_bufs, leaf_sha) = build_leaf(&dev, &lp, &world_lw, &inputs, &traces_dir, &mut cache,
        wgpu_backend::gkr::leaf_chunks(&dev, world_lw.log_stack).1);
    let leaf_exp = {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(inputs.join("world-info.json")).unwrap()).unwrap();
        j["leaf_sha256_raw_dump"].as_str().unwrap().to_string()
    };
    let leaf_match = leaf_sha == leaf_exp;
    eprintln!("LEAF gpu={leaf_sha} exp={leaf_exp} match={leaf_match} vmhwm={}MB", vm_hwm_mb());
    if stop_at == "leaf" {
        println!("W2D leaf-only: match={leaf_match} vmhwm={}MB", vm_hwm_mb());
        return;
    }

    // ---------------- GPU GKR rounds ----------------
    let mut gin: GkrInputs = serde_json::from_str(&std::fs::read_to_string(inputs.join("gkr-inputs-w2d.json")).unwrap()).unwrap();
    let mut gkr_const_mism = 0usize;
    {
        let inst = &mut gin.instances[0];
        let mut mus_live: Vec<[u32; 4]> = vec![[derived[107], derived[108], derived[109], derived[110]]];
        let mut c = 111usize;
        let mut lam_live = Vec::new();
        let mut r_live: Vec<Vec<[u32; 4]>> = Vec::new();
        for j in 1..inst.t {
            lam_live.push([derived[c], derived[c + 1], derived[c + 2], derived[c + 3]]); c += 4;
            let mut rs = Vec::new();
            for _ in 0..j {
                c += 12;
                rs.push([derived[c], derived[c + 1], derived[c + 2], derived[c + 3]]); c += 4;
            }
            c += 16;
            mus_live.push([derived[c], derived[c + 1], derived[c + 2], derived[c + 3]]); c += 4;
            r_live.push(rs);
        }
        for j in 1..inst.t {
            let want: Vec<u64> = std::iter::once(mus_live[j as usize - 1])
                .chain(if j >= 2 { r_live[j as usize - 2].iter().copied() } else { [].iter().copied() })
                .flat_map(|q: [u32; 4]| q.into_iter().map(|w| w as u64)).collect();
            if inst.per_round_xi.get(&j).cloned().unwrap_or_default() != want { gkr_const_mism += 1; }
            inst.per_round_xi.insert(j, want);
            let lamw = lam_live[j as usize - 1].iter().map(|&w| w as u64).collect::<Vec<_>>();
            if inst.rounds[j as usize - 1].lambda != lamw { gkr_const_mism += 1; }
            inst.rounds[j as usize - 1].lambda = lamw;
            for k in 0..j as usize {
                let rw = r_live[j as usize - 1][k].iter().map(|&w| w as u64).collect::<Vec<_>>();
                if inst.rounds[j as usize - 1].subs[k].r != rw { gkr_const_mism += 1; }
                inst.rounds[j as usize - 1].subs[k].r = rw;
            }
        }
    }
    eprintln!("GKR constants live-vs-recorded mismatches={gkr_const_mism}");
    let mut snaps = Vec::new();
    let gout = run_instance_gpu(&dev, &pipe, &gin.instances[0], leaf_bufs, leaf_sha.clone(), &world, &mut snaps);
    let val_mism = check_expected(&gin.instances[0], &gout);
    let snap_mism = check_snapshots(&gin.instances[0], &gout, &world);
    eprintln!("GKR valmism={} snapmism={} root_q={:?} vmhwm={}MB", val_mism.len(), snap_mism.len(), gout.root_q, vm_hwm_mb());
    for x in val_mism.iter().take(16) { eprintln!("VALMISM {x}"); }
    let mut plane = std::collections::HashMap::<String, usize>::new();
    for x in val_mism.iter() {
        let k = x.split(|c| c == '[' || c == ']').next().unwrap_or("?").to_string();
        *plane.entry(k).or_default() += 1;
    }
    eprintln!("VALMISM-TAGS {plane:?}");
    if stop_at == "gkr" {
        println!("W2D gkr: valmism={} snapmism={}", val_mism.len(), snap_mism.len());
        return;
    }

    // ---------------- GPU grinds ----------------
    let wgrind = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grind"),
        source: wgpu::ShaderSource::Wgsl(format!("{}{}\n{}", wgsl::mul_prelude(dev.has_shader_int64), wgsl::COMMON, GRIND_WGSL).into()),
    });
    let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("grind-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let layout = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
    let mkp = |name: &str| dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name), layout: Some(&layout), module: &wgrind, entry_point: Some(name), compilation_options: Default::default(), cache: None,
    });
    let p_scan = mkp("grind_scan");
    let p_check = mkp("grind_check");
    let kc = wf.buf(&dev, &constants_buffer_mont(), "kc");
    let io1 = wf.buf(&dev, &[0u32; 16], "bases");
    let io2 = wf.buf(&dev, &[0u32; 8], "check");
    let amin = wf.buf(&dev, &[0xFFFFFFFFu32; 4], "amin");
    let uni = wf.uni16(&dev);
    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: kc.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: io1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: io2.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: amin.as_entire_binding() },
        ],
        label: None,
    });
    let mut wmin_all: Vec<u32> = Vec::new();
    let mut grind_rows: Vec<serde_json::Value> = Vec::new();
    let mut grind_ok = true;
    for i2 in 0..site_events.len() {
        let ev = site_events[i2];
        let bits = site_bits[i2];
        let wrec = site_w[i2];
        let (base, idx) = bases[i2];
        let limit = ((wrec + 1).next_power_of_two() << 1).max(256);
        let mask = (1u64 << bits) - 1;
        dev.queue.write_buffer(&io1, 0, &u32s_to_bytes(&base));
        dev.queue.write_buffer(&amin, 0, &u32s_to_bytes(&[0xFFFFFFFFu32; 4]));
        WhirFold::set_uni(&dev, &uni, idx as u32, mask as u32, limit, 0);
        WhirFold::dispatch(&dev, &p_scan, &bg, limit as usize);
        let wmin = WhirFold::read_back(&dev, &amin, 1)[0];
        WhirFold::set_uni(&dev, &uni, idx as u32, mask as u32, wmin, 0);
        WhirFold::dispatch(&dev, &p_check, &bg, 1);
        let check_gpu = WhirFold::read_back(&dev, &io2, 1)[0];
        let check_rec = vals[ev + 1];
        let check_cpu = Duplex::grind_check(&p2, &base, idx, wmin);
        let ok = wmin == wrec && check_gpu == check_rec && check_cpu == check_rec && mask_ok(check_gpu, bits);
        grind_ok &= ok;
        wmin_all.push(wmin);
        grind_rows.push(serde_json::json!({"site": i2, "event": ev, "bits": bits, "w_rec": wrec, "w_gpu": wmin, "check_rec": check_rec, "check_gpu": check_gpu, "check_cpu": check_cpu, "ok": ok}));
        eprintln!("GRIND {i2} ev={ev} bits={bits} w_gpu={wmin} rec={wrec} ok={ok} vmhwm={}MB", vm_hwm_mb());
    }
    if stop_at == "grind" {
        println!("W2D grinds: ok={grind_ok}");
        return;
    }

    // ---------------- WHIR fold: W2C CPU chain + GPU mobius/fft/merkle ----------------
    // W2d lowmem: w0-f0.bin/w0-w0.bin (256 MiB each) are NOT loaded as host
    // Vecs — the (raw_ef(read_u32s(..)) pattern cost 2x256 MiB per file as a
    // transient). Round-0 first sub-round streams both files in one sequential
    // pass that computes the s_eval accumulator AND emits the folded halves.
    let dump_alphas = read_u32s(&world.join("whir/w0-alphas.bin"));
    let f0_path = world.join("whir/w0-f0.bin");
    let w0_path = world.join("whir/w0-w0.bin");
    let n_f0 = (std::fs::metadata(&f0_path).unwrap().len() / 16) as usize;
    let mut my_f: Vec<Ef> = Vec::new();
    let mut my_w: Vec<Ef> = Vec::new();
    let mut log_rs = log_rs0;
    let mut aidx = 0usize;
    let mut fold_rows: Vec<serde_json::Value> = Vec::new();
    let mut fold_ok = true;
    let mut ev = mu_event + 6;
    {
        let mu_c = read_u32s(&world.join("whir/w0-mu.bin"));
        let live_mu: Vec<u32> = (mu_event + 2..mu_event + 6).map(|x| derived[x]).collect();
        if live_mu != mu_c { fold_ok = false; eprintln!("FOLD mu live != dump live={live_mu:?} dump={mu_c:?}"); }
    }
    let mut fold_commit_words = vec![0u32; n]; // computed O words by position
    let mut fold_ood_words = vec![0u32; n];
    let mut fold_seval_pos: Vec<(usize, Vec<u32>)> = Vec::new(); // (start, 8 words) computed s_evals
    let mut fold_fc: Vec<u32> = Vec::new();
    for (r, &nq) in nq_v.iter().enumerate() {
        let is_last = r + 1 == nq_v.len();
        for kq in 0..k_whir {
            let streamed = r == 0 && kq == 0;
            let half = if streamed { n_f0 / 2 } else { my_f.len() / 2 };
            // the fold alpha lives at ev+10 (after sv(8) + witness + check) and
            // is duplex-derived in pass 1, so the streamed first pass can fold
            // in the same sequential pass that accumulates s_evals.
            let a = raw_ef(&[derived[ev + 10], derived[ev + 11], derived[ev + 12], derived[ev + 13]])[0];
            let mut acc1 = ef_zero();
            let mut acc2 = ef_zero();
            if streamed {
                let mut nf: Vec<Ef> = Vec::with_capacity(half);
                let mut nw: Vec<Ef> = Vec::with_capacity(half);
                let mut ff = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&f0_path).unwrap());
                let mut fw = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&w0_path).unwrap());
                const PC: usize = 1 << 16; // rows (pairs) per chunk: 2 MiB per file
                let mut fraw = vec![0u8; PC * 32];
                let mut wraw = vec![0u8; PC * 32];
                let mut done = 0usize;
                while done < half {
                    let cnt = PC.min(half - done);
                    read_exact_all(&mut ff, &mut fraw[..cnt * 32]);
                    read_exact_all(&mut fw, &mut wraw[..cnt * 32]);
                    for y in 0..cnt {
                        let f0v = raw_ef4(&fraw[y * 32..y * 32 + 16]);
                        let f1v = raw_ef4(&fraw[y * 32 + 16..y * 32 + 32]);
                        let w0v = raw_ef4(&wraw[y * 32..y * 32 + 16]);
                        let w1v = raw_ef4(&wraw[y * 32 + 16..y * 32 + 32]);
                        let df = ef_sub(&f1v, &f0v);
                        let dw = ef_sub(&w1v, &w0v);
                        acc1 = ef_add(&acc1, &ef_mul(&ef_add(&f0v, &df), &ef_add(&w0v, &dw)));
                        acc2 = ef_add(&acc2, &ef_mul(&ef_add(&f0v, &ef_scale(&df, 2)), &ef_add(&w0v, &ef_scale(&dw, 2))));
                        nf.push(ef_add(&f0v, &ef_mul(&df, &a)));
                        nw.push(ef_add(&w0v, &ef_mul(&dw, &a)));
                    }
                    done += cnt;
                }
                my_f = nf;
                my_w = nw;
            } else {
                for y in 0..half {
                    let f0v = my_f[2 * y];
                    let df = ef_sub(&my_f[2 * y + 1], &f0v);
                    let w0v = my_w[2 * y];
                    let dw = ef_sub(&my_w[2 * y + 1], &w0v);
                    acc1 = ef_add(&acc1, &ef_mul(&ef_add(&f0v, &df), &ef_add(&w0v, &dw)));
                    acc2 = ef_add(&acc2, &ef_mul(&ef_add(&f0v, &ef_scale(&df, 2)), &ef_add(&w0v, &ef_scale(&dw, 2))));
                }
            }
            let sv = canon_ef(&[acc1, acc2]);
            for j in 0..8 {
                if vals[ev + j] != sv[j] || flags[ev + j] { fold_ok = false; eprintln!("FOLD r{r} s{kq} SEVALS mismatch ev {}", ev + j); }
            }
            fold_seval_pos.push((ev, sv));
            ev += 8;
            ev += 2; // pow witness + check (site-verified)
            if derived[ev..ev + 4] != dump_alphas[aidx * 4..aidx * 4 + 4] {
                fold_ok = false; eprintln!("FOLD r{r} alpha {aidx} live != dump at {ev}");
            }
            ev += 4;
            if !streamed {
                // fold in place (W2d lowmem: the old nf/nw Vecs doubled the
                // resident my_f/my_w peak)
                for y in 0..half {
                    let f0v = my_f[2 * y];
                    let df = ef_sub(&my_f[2 * y + 1], &f0v);
                    my_f[y] = ef_add(&f0v, &ef_mul(&df, &a));
                    let w0v = my_w[2 * y];
                    let dw = ef_sub(&my_w[2 * y + 1], &w0v);
                    my_w[y] = ef_add(&w0v, &ef_mul(&dw, &a));
                }
                my_f.truncate(half);
                my_w.truncate(half);
                // W2d lowmem: truncate keeps capacity; give the freed halves
                // back so the 256 MiB f/w slabs don't ride through the
                // mobius/fft/merkle stage (peak was there, not at build time).
                my_f.shrink_to_fit();
                my_w.shrink_to_fit();
            }
            aidx += 1;
        }
        let coeffs = gpu_mobius(&dev, &wf, &my_f);
        let mut row = serde_json::json!({"round": r, "nq": nq});
        if !is_last {
            let n_fft = 1usize << (log_rs - 1);
            let root = gpu_fft_merkle_stream(&dev, &wf, &coeffs, (log_rs - 1) as u32, n_fft, k_whir);
            let stream_ok = (0..8).all(|j| !flags[ev + j] && vals[ev + j] == root[j]);
            if !stream_ok { fold_ok = false; eprintln!("FOLD r{r} COMMIT stream mismatch at {ev}: mine {root:?} stream {:?}", &vals[ev..ev + 8]); }
            for j in 0..8 { fold_commit_words[ev + j] = root[j]; }
            row["commit_event"] = serde_json::json!(ev);
            row["commit_match_stream"] = serde_json::json!(stream_ok);
            ev += 8;
            let z0 = raw_ef(&[derived[ev], derived[ev + 1], derived[ev + 2], derived[ev + 3]]);
            ev += 4;
            let yv = {
                let mut acc = coeffs[coeffs.len() - 1];
                for cc in coeffs[..coeffs.len() - 1].iter().rev() {
                    acc = ef_add(&ef_mul(&acc, &z0[0]), cc);
                }
                acc
            };
            let y_c = canon_ef(&[yv]);
            let ood_ok = (0..4).all(|j| !flags[ev + j] && vals[ev + j] == y_c[j]);
            if !ood_ok { fold_ok = false; eprintln!("FOLD r{r} OOD mismatch at {ev}: mine {y_c:?} stream {:?}", &vals[ev..ev + 4]); }
            for j in 0..4 { fold_ood_words[ev + j] = y_c[j]; }
            row["ood_match_stream"] = serde_json::json!(ood_ok);
            ev += 4;
            if q_bits > 0 { ev += 2; }
            let d_bits = log_rs - k_whir;
            let qmask = (1u64 << d_bits) - 1;
            let qidx: Vec<u64> = (0..nq).map(|j| (derived[ev + j] as u64) & qmask).collect();
            ev += nq;
            let gamma = raw_ef(&[derived[ev], derived[ev + 1], derived[ev + 2], derived[ev + 3]]);
            ev += 4;
            {
                let dim = my_w.len().trailing_zeros() as usize;
                let n_w = my_w.len();
                let zpow = exp_powers_of_2(&z0[0], dim);
                let mut eq: Vec<Ef> = vec![ef_zero(); n_w];
                eq[0] = gamma[0];
                for (j, zj) in zpow.iter().enumerate() {
                    let one_m = ef_sub(&[encode(1), 0, 0, 0], zj);
                    let step = 1usize << j;
                    for i2 in (0..2usize << j).step_by(step * 2) {
                        for k2 in 0..step {
                            let prev = eq[i2 + k2];
                            eq[i2 + k2] = ef_mul(&prev, &one_m);
                            eq[i2 + k2 + step] = ef_mul(&prev, zj);
                        }
                    }
                }
                for i2 in 0..n_w { my_w[i2] = ef_add(&my_w[i2], &eq[i2]); }
                let omega = root_of_unity_raw(d_bits as u32);
                let mut gpow = ef_mul(&gamma[0], &gamma[0]);
                for qi in 0..nq {
                    let zc = powm(omega, qidx[qi]);
                    let dim = my_w.len().trailing_zeros() as usize;
                    let mut zp2: Vec<u32> = Vec::with_capacity(dim);
                    let mut cur = zc;
                    for _ in 0..dim { zp2.push(cur); cur = mont_mul(cur, cur); }
                    let n_w = my_w.len();
                    let mut eq1: Vec<u32> = vec![0u32; n_w];
                    eq1[0] = encode(1);
                    for (j, zj) in zp2.iter().enumerate() {
                        let om = fsub(encode(1), *zj);
                        let step = 1usize << j;
                        for i2 in (0..2usize << j).step_by(step * 2) {
                            for k2 in 0..step {
                                let prev = eq1[i2 + k2];
                                eq1[i2 + k2] = mont_mul(prev, om);
                                eq1[i2 + k2 + step] = mont_mul(prev, *zj);
                            }
                        }
                    }
                    for i2 in 0..n_w {
                        let add = ef_scale(&gpow, decode(eq1[i2]));
                        my_w[i2] = ef_add(&my_w[i2], &add);
                    }
                    gpow = ef_mul(&gpow, &gamma[0]);
                }
            }
            log_rs -= 1;
        } else {
            let fc = canon_ef(&coeffs);
            fold_fc = fc.clone();
            let nfc = fc.len();
            let fin_ok = (0..nfc).all(|j| !flags[ev + j] && vals[ev + j] == fc[j]);
            if !fin_ok { fold_ok = false; eprintln!("FOLD final coeffs mismatch at {ev} len {nfc}"); }
            row["finalcoeffs_match_stream"] = serde_json::json!(fin_ok);
            let fc_c: Vec<u32> = read_ef(&world.join("whir/w0-finalcoeffs.bin")).into_iter().flatten().collect();
            row["finalcoeffs_match_dump"] = serde_json::json!(fc == fc_c);
            if fc != fc_c { fold_ok = false; }
            ev += nfc;
            ev = fin_qpow;
            if q_bits > 0 { ev += 2; }
            ev += nq + 4;
        }
        fold_rows.push(row);
        eprintln!("FOLD r{r} done ev={ev} fold_ok={fold_ok} vmhwm={}MB", vm_hwm_mb());
    }
    eprintln!("FOLD ok={fold_ok} end_ev={ev}");

    // ---------------- assembly: emitted stream ----------------
    // GPU/computed regions:
    //  [87..5911): logup GKR from gout (O positions)
    //  fold s_evals/commit/ood/finalcoeffs computed words by position
    //  site witnesses: GPU grind winners; S words: derived
    //  everything else O: recorded passthrough
    let mut gkr_words = vec![0u32; n];
    {
        let mut c = 87usize;
        for w in gout.root_q.iter() { gkr_words[c] = *w; c += 1; }
        for w in &gout.claims0 { gkr_words[c] = *w; c += 1; }
        c += 4; // mu1 sits at 107..110 (S word)
        for rr in &gout.rounds {
            c += 4; // lam S
            for sm in &rr.subs {
                for w in sm { gkr_words[c] = *w; c += 1; }
                c += 4; // r S
            }
            for w in &rr.claims { gkr_words[c] = *w; c += 1; }
            c += 4; // mu S
        }
        assert_eq!(c, 5911, "gkr word cursor");
    }
    let mut computed = vec![0u32; n];
    let mut computed_src: Vec<Option<&'static str>> = vec![None; n];
    for k in 87..5911 {
        if !flags[k] { computed[k] = gkr_words[k]; computed_src[k] = Some("logup-gpu"); }
    }
    for (st, sv) in &fold_seval_pos {
        for j in 0..8 { computed[st + j] = sv[j]; computed_src[st + j] = Some("fold-sevals"); }
    }
    // commit/ood words: mark positions where the fold walk placed computed words
    // (need the exact commit/ood position list: recompute cheaply)
    {
        let mut c = mu_event + 6;
        for (r, &nq) in nq_v.iter().enumerate() {
            let is_last = r + 1 == nq_v.len();
            c += k_whir * 14;
            if !is_last {
                for j in 0..8 { computed[c + j] = fold_commit_words[c + j]; computed_src[c + j] = Some("fold-commit-gpu"); }
                c += 8 + 4;
                for j in 0..4 { computed[c + j] = fold_ood_words[c + j]; computed_src[c + j] = Some("fold-ood"); }
                c += 4 + 2 + nq + 4;
            }
        }
    }
    {
        // finalcoeffs position: after r3 subs
        let mut c = mu_event + 6;
        for (r, &nq) in nq_v.iter().enumerate() {
            let is_last = r + 1 == nq_v.len();
            c += k_whir * 14;
            if !is_last { c += 8 + 4 + 4 + 2 + nq + 4; }
            else {
                for j in 0..fold_fc.len() { computed[c + j] = fold_fc[j]; computed_src[c + j] = Some("fold-finalcoeffs"); }
            }
        }
    }
    let mut emit: Vec<u32> = Vec::with_capacity(n);
    let mut emit_mism: Vec<(usize, u32, u32, &'static str)> = Vec::new();
    for k in 0..n {
        let (w, src) = if flags[k] {
            (derived[k], "sample-live")
        } else if let Some(pos) = site_events.iter().position(|e| *e == k) {
            (wmin_all[pos], "grind-gpu")
        } else if let Some(s) = computed_src[k] {
            (computed[k], s)
        } else {
            (vals[k], "recorded-passthrough")
        };
        if w != vals[k] && emit_mism.len() < 50 { emit_mism.push((k, vals[k], w, src)); }
        emit.push(w);
    }
    let mut of = Vec::with_capacity(n * 8);
    for w in &emit { of.extend_from_slice(&(*w as u64).to_le_bytes()); }
    std::fs::write(&out, &of).unwrap();
    std::fs::copy(streams.join(format!("{tag}.prove.bin.flags")), format!("{}.flags", out.display())).ok();
    let rec_sha = sha256_hex(&raw);
    let gpu_sha = sha256_hex(&of);
    let equal = rec_sha == gpu_sha;
    let first_mismatch = emit.iter().enumerate().find(|(k, w)| **w != vals[*k]).map(|(k, w)| (k, vals[k], *w));
    let n_pass = (0..n).filter(|k| !flags[*k] && computed_src[*k].is_none() && !site_events.contains(k)).count();
    assert_eq!(ev, n, "fold walk must consume stream");

    let green = chain_ok && leaf_match && gkr_const_mism == 0 && val_mism.is_empty() && snap_mism.is_empty()
        && grind_ok && fold_ok && equal && emit_mism.is_empty();
    let rep = serde_json::json!({
        "world": world.to_str().unwrap(),
        "adapter": dev.report.name,
        "chain_samples": nsamp, "chain_ok": chain_ok, "chain_mismatches": sample_mism,
        "leaf_match": leaf_match, "leaf_sha_gpu": leaf_sha, "leaf_sha_expected": leaf_exp,
        "gkr_const_mism": gkr_const_mism, "gkr_value_mismatches": val_mism, "gkr_snap_mismatches": snap_mism,
        "grind_ok": grind_ok, "grind_rows": grind_rows,
        "fold_ok": fold_ok, "fold_rows": fold_rows,
        "stream_equal": equal, "rec_sha256": rec_sha, "gpu_sha256": gpu_sha,
        "first_mismatch": first_mismatch, "emit_mismatches": emit_mism,
        "passthrough_words": n_pass,
        "vm_hwm_mb": vm_hwm_mb(),
        "green": green,
    });
    std::fs::write(&report, serde_json::to_string_pretty(&rep).unwrap()).unwrap();
    eprintln!("W2D {}: equal={} passthrough_words={n_pass} vmhwm={}MB report={}", if green { "GREEN" } else { "RED" }, equal, vm_hwm_mb(), report.display());
    if !green { std::process::exit(2); }
}

// ---------------- GPU fold helpers (mirrors w2c_fold, lowmem uploads) ----------------
fn read_back_off(dev: &WgpuDevice, b: &wgpu::Buffer, off_words: u64, nwords: usize) -> Vec<u32> {
    let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rb-off"),
        size: (nwords * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(b, off_words * 4, &staging, 0, (nwords * 4) as u64);
    dev.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    dev.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    rx.recv().unwrap().unwrap();
    let mut out = vec![0u32; nwords];
    {
        let d = slice.get_mapped_range().unwrap();
        for (i, w) in out.iter_mut().enumerate() {
            *w = u32::from_le_bytes(d[i * 4..i * 4 + 4].try_into().unwrap());
        }
    }
    drop(slice);
    staging.unmap();
    out
}
fn gpu_mobius(dev: &WgpuDevice, wf: &WhirFold, f: &[Ef]) -> Vec<Ef> {
    let n = f.len();
    let data = whir_fold::data_shard(dev, n, "mob in");
    data.write_slice(dev, 0, &u32s_to_bytes(&flatten(f)));
    whir_fold::whir_stages(dev, wf, &data, n, 1, &[]);
    unflatten(&data.read_words(dev, 0, (n * 4) as u64))
}
fn gpu_fft_merkle_stream(dev: &WgpuDevice, wf: &WhirFold, coeffs: &[Ef], log_n: u32, n: usize, k_whir: usize) -> [u32; 8] {
    // W2d lowmem + Phase-2 sharding: FFT runs entirely on a sharded device
    // buffer; the result is streamed to the merkle rows buffer in 16 MiB
    // read-back chunks (all binds <= device binding cap).
    let data = whir_fold::data_shard(dev, n, "fft-in");
    const UC: usize = 1 << 18; // upload rows per chunk (4 MiB)
    let mut bv: Vec<u8> = Vec::with_capacity(UC * 16);
    let mut j0 = 0usize;
    while j0 < n {
        let cnt = UC.min(n - j0);
        bv.clear();
        for jj in 0..cnt {
            let j = j0 + jj;
            let i = ((j as u32).reverse_bits() >> (32 - log_n)) as usize;
            if i < coeffs.len() {
                bv.extend_from_slice(&u32s_to_bytes(&coeffs[i]));
            } else {
                bv.extend_from_slice(&[0u8; 16]);
            }
        }
        data.write_slice(dev, (j0 * 16) as u64, &bv);
        j0 += cnt;
    }
    drop(bv);
    let mut tws: Vec<wgpu::Buffer> = Vec::new();
    let mut half = 1usize;
    while half < n {
        let wm = powm(root_of_unity_raw(log_n), (n / (half * 2)) as u64);
        let mut tw = vec![0u32; half];
        tw[0] = encode(1);
        for k in 1..half { tw[k] = mont_mul(tw[k - 1], wm); }
        tws.push(wf.buf(dev, &tw, "tw"));
        half *= 2;
    }
    whir_fold::whir_stages(dev, wf, &data, n, 0, &tws);
    drop(tws);
    // stream fft result -> merkle rows (16 MiB words -> 32 MiB rows per chunk)
    let rows_b = ShardBuf::new(
        dev,
        "m rows",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        (n * 32) as u64,
    );
    const DC: usize = 1 << 20; // rows per download chunk (16 MiB words / 32 MiB rows)
    let mut bv: Vec<u8> = Vec::with_capacity(DC * 32);
    let mut r0 = 0usize;
    while r0 < n {
        let cnt = DC.min(n - r0);
        let words = data.read_words(dev, (r0 * 4) as u64, (cnt * 4) as u64);
        bv.clear();
        for row in words.chunks_exact(4) {
            bv.extend_from_slice(&u32s_to_bytes(row));
            bv.extend_from_slice(&[0u8; 16]);
        }
        rows_b.write_slice(dev, (r0 * 32) as u64, &bv);
        r0 += cnt;
    }
    drop(data);
    drop(bv);
    whir_fold::merkle_from_rows(dev, wf, &rows_b, n, k_whir)
}
