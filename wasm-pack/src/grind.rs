//! Phase 3b: W2C PoW grind IN THE BROWSER.
//!
//! Port of src/bin/w2c_grind.rs to the await-based wasm tier using the same
//! native u32 design: one Poseidon2 perm per invocation (w == gid.x),
//! atomicMin winner index (NO u64 CAS), lowest-global-invocation-id tie-break
//! (total because w == gid.x), then a 1-thread grind_check kernel that
//! recomputes the winner's check word. CPU cross-check via w2c::Duplex.
//! Inputs (bases / sites table / GREEN goldens) are embedded at build time;
//! the run-hash must equal the recorded golden e69fddee611e56ab over the same
//! canonical JSON rows the native binary hashes.
use std::cell::RefCell;
use std::rc::Rc;

use wgpu_backend::field::{constants_buffer_mont, sha256_hex, u32s_to_bytes, Poseidon2};
use wgpu_backend::w2c::{mask_ok, Duplex};
use wgpu_backend::wgsl;

use crate::atlas::{Bind, ReadBatch};
use crate::gatefast::mkbuf;
use crate::ctx;

const BASES: &[u8] = include_bytes!("../../w2c-grind-bases.bin");
const SITES_SRC: &str = include_str!("../../w2c/pow_sites_r1.jsonl");
const GOLD_SRC: &str = include_str!("../../w2c-results-r1-GREEN.json");
const GOLDEN_HASH: &str = "e69fddee611e56ab";

/// Dawn enforces minStorageBufferOffsetAlignment = 256 (WebGPU spec floor),
/// so every per-site slot inside a shared buffer is 256-byte strided.
const SLOT: u64 = 256;

/// Verbatim from wgpu-backend/src/bin/w2c_grind.rs (no u64 CAS anywhere).
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

pub struct Site {
    inst: u32,
    site: u32,
    bits: u32,
    idx: u32,
    w_rec: u32,
    check_rec: u32,
    base: [u32; 16],
}

fn limit_for(w_rec: u32) -> u32 {
    ((w_rec + 1).next_power_of_two() << 1).max(256)
}

fn sites() -> Vec<Site> {
    let bw: Vec<u32> = BASES
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(bw.len() % 22, 0, "bases file must be 22 words/site");
    bw.chunks_exact(22)
        .map(|r| Site {
            inst: r[0],
            site: r[1],
            bits: r[2],
            idx: r[3],
            w_rec: r[4],
            check_rec: r[5],
            base: r[6..22].try_into().unwrap(),
        })
        .collect()
}

/// (site id, witness, bits) per pow_sites_r1.jsonl + min_equals_recorded count
/// from w2c-results-r1-GREEN.json — the native bin's cross-checks.
fn cross_check(sites: &[Site]) -> (bool, usize) {
    let mut ok = true;
    let tbl: Vec<serde_json::Value> = SITES_SRC
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    if tbl.len() != sites.len() {
        ok = false;
    }
    for (j, s) in sites.iter().enumerate() {
        let t = &tbl[j];
        if t["site"].as_u64().unwrap() as u32 != s.site
            || t["witness"].as_u64().unwrap() as u32 != s.w_rec
            || t["bits"].as_u64().unwrap() as u32 != s.bits
        {
            ok = false;
        }
    }
    let goldv: serde_json::Value = serde_json::from_str(GOLD_SRC).unwrap();
    let mut minrec = 0usize;
    for (_k, instv) in goldv["instances"].as_object().unwrap() {
        if let Some(rows) = instv["grind"].as_array() {
            minrec += rows
                .iter()
                .filter(|r| r["min_equals_recorded"] == true)
                .count();
        }
    }
    (ok, minrec)
}

struct Setup {
    p_scan: wgpu::ComputePipeline,
    p_check: wgpu::ComputePipeline,
    kc: wgpu::Buffer,
    io1: wgpu::Buffer,  // 90 slots x 16 words
    io2: wgpu::Buffer,  // 90 slots x 1 word
    amin: wgpu::Buffer, // 90 slots x 1 word
    unis: Vec<wgpu::Buffer>,
    bgs: Vec<wgpu::BindGroup>,
}

thread_local! {
    static SETUP: RefCell<Option<Rc<Setup>>> = const { RefCell::new(None) };
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn setup(dev_rc: &Rc<crate::Ctx>, n: usize) -> Rc<Setup> {
    SETUP.with(|slot| {
        if let Some(s) = slot.borrow().clone() {
            return s;
        }
        let dev = &dev_rc.dev.device;
        let src = format!(
            "{}{}\n{}",
            wgsl::mul_prelude(dev_rc.dev.has_shader_int64),
            wgsl::COMMON,
            GRIND_WGSL
        );
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grind"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(4, false),
            ],
        });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let mk = |name: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let p_scan = mk("grind_scan");
        let p_check = mk("grind_check");
        let kc_mont = constants_buffer_mont();
        let kc = mkbuf(dev, "grind-kc", (kc_mont.len() * 4) as u64);
        dev_rc
            .dev
            .queue
            .write_buffer(&kc, 0, &u32s_to_bytes(&kc_mont));
        let io1 = mkbuf(dev, "grind-io1", n as u64 * SLOT);
        let io2 = mkbuf(dev, "grind-io2", (n as u64 * SLOT).max(16));
        let amin = mkbuf(dev, "grind-amin", (n as u64 * SLOT).max(16));
        let mut unis = Vec::with_capacity(n);
        let mut bgs = Vec::with_capacity(n);
        for s in 0..n {
            let uni = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("grind-uni"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let binds = [
                Bind::whole(&kc),
                Bind::slice(&io1, (s as u64) * SLOT, 64),
                Bind::slice_to_end(&io2, (s as u64) * SLOT),
                Bind::whole(&uni),
                Bind::slice_to_end(&amin, (s as u64) * SLOT),
            ];
            let entries: Vec<wgpu::BindGroupEntry> = binds
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: match b {
                        Bind::Whole(bb) => bb.as_entire_binding(),
                        Bind::Slice(bb, off, len) => wgpu::BindingResource::Buffer(
                            wgpu::BufferBinding {
                                buffer: bb,
                                offset: *off,
                                size: len.and_then(std::num::NonZeroU64::new),
                            },
                        ),
                    },
                })
                .collect();
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("grind-site"),
                layout: &bgl,
                entries: &entries,
            });
            unis.push(uni);
            bgs.push(bg);
        }
        let st = Rc::new(Setup {
            p_scan,
            p_check,
            kc,
            io1,
            io2,
            amin,
            unis,
            bgs,
        });
        *slot.borrow_mut() = Some(st.clone());
        st
    })
}

fn set_uni(queue: &wgpu::Queue, uni: &wgpu::Buffer, seed: u32, mask: u32, a2: u32) {
    let mut x = [0u8; 32];
    x[0..4].copy_from_slice(&seed.to_le_bytes());
    x[4..8].copy_from_slice(&mask.to_le_bytes());
    x[8..12].copy_from_slice(&a2.to_le_bytes());
    queue.write_buffer(uni, 0, &x);
}

/// One full 90-site grind sweep. batched=false reproduces the native
/// per-site sequencing (2 mapAsync per site); batched=true rides all 90
/// scans + all 90 checks in two submits with ONE mapAsync each.
async fn grind_sweep(
    ctxr: &Rc<crate::Ctx>,
    st: &Rc<Setup>,
    sites: &[Site],
    p2: &Poseidon2,
    batched: bool,
) -> Result<(String, bool, u32), String> {
    let dev = &ctxr.dev.device;
    let queue = &ctxr.dev.queue;
    let n = sites.len();
    let maps: u32;
    let (wmins, checks): (Vec<u32>, Vec<u32>) = if batched {
        // bases -> io1 slots (one write, 256 B slot stride)
        let mut img: Vec<u32> = vec![0u32; n * (SLOT as usize / 4)];
        for (i, s) in sites.iter().enumerate() {
            img[i * (SLOT as usize / 4)..i * (SLOT as usize / 4) + 16].copy_from_slice(&s.base);
        }
        queue.write_buffer(&st.io1, 0, &u32s_to_bytes(&img));
        // reset all winner slots
        let mut fill: Vec<u32> = vec![0u32; n * (SLOT as usize / 4)];
        for i in 0..n {
            fill[i * (SLOT as usize / 4)] = 0xFFFFFFFFu32;
        }
        queue.write_buffer(&st.amin, 0, &u32s_to_bytes(&fill));
        // per-site scan uniforms
        for (s, uni) in sites.iter().zip(st.unis.iter()) {
            set_uni(queue, uni, s.idx, ((1u64 << s.bits) - 1) as u32, limit_for(s.w_rec));
        }
        {
            let mut enc =
                dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut p = enc.begin_compute_pass(&Default::default());
                p.set_pipeline(&st.p_scan);
                for (i, s) in sites.iter().enumerate() {
                    p.set_bind_group(0, &st.bgs[i], &[]);
                    p.dispatch_workgroups(limit_for(s.w_rec) / 256, 1, 1);
                }
            }
            queue.submit(Some(enc.finish()));
        }
        let mut rb = ReadBatch::new();
        for i in 0..n {
            rb.push(&st.amin, (i as u64) * SLOT, 1);
        }
        let wmins: Vec<u32> = rb
            .resolve(dev, queue)
            .await?
            .into_iter()
            .map(|mut w| w.pop().unwrap())
            .collect();
        // winner check: per-site uni carries wmin, one pass, one submit
        for (i, s) in sites.iter().enumerate() {
            set_uni(queue, &st.unis[i], s.idx, ((1u64 << s.bits) - 1) as u32, wmins[i]);
        }
        {
            let mut enc =
                dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut p = enc.begin_compute_pass(&Default::default());
                p.set_pipeline(&st.p_check);
                for i in 0..n {
                    p.set_bind_group(0, &st.bgs[i], &[]);
                    p.dispatch_workgroups(1, 1, 1);
                }
            }
            queue.submit(Some(enc.finish()));
        }
        let mut rb = ReadBatch::new();
        for i in 0..n {
            rb.push(&st.io2, (i as u64) * SLOT, 1);
        }
        let checks: Vec<u32> = rb
            .resolve(dev, queue)
            .await?
            .into_iter()
            .map(|mut c| c.pop().unwrap())
            .collect();
        maps = 2;
        (wmins, checks)
    } else {
        // native sequencing per site: 2 mapAsync round-trips each
        let mut wmins = Vec::with_capacity(n);
        let mut checks = Vec::with_capacity(n);
        for (i, s) in sites.iter().enumerate() {
            queue.write_buffer(&st.io1, (i as u64) * SLOT, &u32s_to_bytes(&s.base));
            queue.write_buffer(&st.amin, (i as u64) * SLOT, &u32s_to_bytes(&[0xFFFFFFFFu32]));
            set_uni(queue, &st.unis[i], s.idx, ((1u64 << s.bits) - 1) as u32, limit_for(s.w_rec));
            {
                let mut enc =
                    dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&st.p_scan);
                    p.set_bind_group(0, &st.bgs[i], &[]);
                    p.dispatch_workgroups(limit_for(s.w_rec) / 256, 1, 1);
                }
                queue.submit(Some(enc.finish()));
            }
            let mut rb = ReadBatch::new();
            rb.push(&st.amin, (i as u64) * SLOT, 1);
            let wmin = rb.resolve(dev, queue).await?[0][0];
            set_uni(queue, &st.unis[i], s.idx, ((1u64 << s.bits) - 1) as u32, wmin);
            {
                let mut enc =
                    dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut p = enc.begin_compute_pass(&Default::default());
                    p.set_pipeline(&st.p_check);
                    p.set_bind_group(0, &st.bgs[i], &[]);
                    p.dispatch_workgroups(1, 1, 1);
                }
                queue.submit(Some(enc.finish()));
            }
            let mut rb = ReadBatch::new();
            rb.push(&st.io2, (i as u64) * SLOT, 1);
            let c = rb.resolve(dev, queue).await?[0][0];
            wmins.push(wmin);
            checks.push(c);
        }
        maps = (2 * n) as u32;
        (wmins, checks)
    };

    // CPU leg + canonical rows (byte-identical construction to w2c_grind.rs)
    let mut all_ok = true;
    let mut rows: Vec<serde_json::Value> = Vec::with_capacity(n);
    for (i, s) in sites.iter().enumerate() {
        let wmin = wmins[i];
        let check_gpu = checks[i];
        let check_cpu = Duplex::grind_check(p2, &s.base, s.idx as usize, wmin);
        let ok = wmin == s.w_rec
            && check_gpu == s.check_rec
            && check_cpu == s.check_rec
            && mask_ok(check_gpu, s.bits as usize);
        all_ok &= ok;
        rows.push(serde_json::json!({
            "site": s.site, "inst": s.inst, "bits": s.bits,
            "w_rec": s.w_rec, "w_min_gpu": wmin,
            "check_rec": s.check_rec, "check_gpu": check_gpu, "check_cpu": check_cpu,
            "mask_ok": mask_ok(check_gpu, s.bits as usize), "ok": ok,
        }));
    }
    let canon = serde_json::to_string(&rows).unwrap();
    Ok((sha256_hex(canon.as_bytes()), all_ok, maps))
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

pub async fn run_all() -> Result<String, String> {
    let ctxr = ctx()?;
    let sites_v = sites();
    let n = sites_v.len();
    let st = setup(&ctxr, n);
    let p2 = Poseidon2::new();
    let (xok, minrec) = cross_check(&sites_v);
    let mut out = String::new();
    out.push_str(&format!("GRIND_SITES: {n} cross_check={} gold_minrec={minrec}\n", xok));

    let t0 = now_ms();
    let (h_seq, ok_seq, maps_seq) = grind_sweep(&ctxr, &st, &sites_v, &p2, false).await?;
    let t1 = now_ms();
    let (h_b1, ok_b1, maps_b1) = grind_sweep(&ctxr, &st, &sites_v, &p2, true).await?;
    let t2 = now_ms();
    let (h_b2, ok_b2, _) = grind_sweep(&ctxr, &st, &sites_v, &p2, true).await?;
    let t3 = now_ms();

    out.push_str(&format!(
        "GRIND_SEQ hash={}.. ok={} maps={maps_seq} ms={:.1}\n",
        &h_seq[..16],
        ok_seq,
        t1 - t0
    ));
    out.push_str(&format!(
        "GRIND_BATCH hash={}.. ok={} maps={maps_b1} ms={:.1}\n",
        &h_b1[..16],
        ok_b1,
        t2 - t1
    ));
    out.push_str(&format!(
        "GRIND_BATCH2 hash={}.. ok={} ms={:.1}\n",
        &h_b2[..16],
        ok_b2,
        t3 - t2
    ));
    let det = h_b1 == h_b2;
    let same = h_seq == h_b1;
    let golden = h_b1.starts_with(GOLDEN_HASH);
    out.push_str(&format!("GRIND_DETERMINISM: {}\n", if det { "PASS" } else { "FAIL" }));
    out.push_str(&format!(
        "GRIND_SEQ_EQ_BATCH: {}\n",
        if same { "PASS" } else { "FAIL" }
    ));
    out.push_str(&format!(
        "GRIND_GOLDEN({GOLDEN_HASH}..): {}\n",
        if golden { "PASS" } else { "FAIL" }
    ));
    let pass = xok && minrec == n && ok_seq && ok_b1 && ok_b2 && det && same && golden;
    out.push_str(&format!("GRIND_WGPU: {}\n", if pass { "PASS" } else { "FAIL" }));
    Ok(out)
}
