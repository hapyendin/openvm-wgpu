//! Phase 3b: gate_c1 with bind-group atlas + single batched readback.
//! Same GPU legs and byte-exact same output lines as the classic per-dispatch
//! gate_c1 (lib.rs run_gate_c1); the differences are:
//!   * buffers + bind groups are persistent (created once, reused per call),
//!   * every dispatch of every leg rides in ONE command encoder / ONE submit,
//!   * the three readbacks (perm outputs, merkle root, NTT outputs) resolve
//!     through a single mapAsync (ReadBatch) instead of three.
use std::cell::RefCell;
use std::rc::Rc;

use wgpu_backend::field::{
    decode, encode, parse_vectors, sha256_hex, to_hex, u32s_to_bytes, Poseidon2, MERKLE_LEAVES,
    MERKLE_ROOT_HEX, MERKLE_ROOT_SHA256, MERKLE_SEED, P,
};

use crate::atlas::{Atlas, Bind, ReadBatch};
use crate::{ctx, gen_lcg, ntt_dit, twiddles, N_NTT};

const VEC_SRC: &str = include_str!("../../wgpu-backend/goldens/vectors.inc");

pub(crate) fn mkbuf(dev: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

struct Bufs {
    kc: wgpu::Buffer,
    uni: wgpu::Buffer,
    in_buf: wgpu::Buffer,
    rows: wgpu::Buffer,
    ab: [wgpu::Buffer; 2],
    n_buf: wgpu::Buffer,
    tw_buf: wgpu::Buffer,
    ntt_out: wgpu::Buffer,
    n_perm: usize,
    n_tw: usize,
}

thread_local! {
    static ATLAS: Atlas = Atlas::new();
    static BUFS: RefCell<Option<Rc<Bufs>>> = const { RefCell::new(None) };
}

fn seed_row(queue: &wgpu::Queue, bufs: &Bufs, seed_canon: u32, rowbase: u32) {
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&seed_canon.to_le_bytes());
    b[4..8].copy_from_slice(&rowbase.to_le_bytes());
    queue.write_buffer(&bufs.uni, 0, &b);
}

fn bufs(dev_rc: &Rc<crate::Ctx>) -> Rc<Bufs> {
    BUFS.with(|slot| {
        if let Some(b) = slot.borrow().clone() {
            return b;
        }
        let dev = &dev_rc.dev.device;
        let kc_mont = wgpu_backend::field::constants_buffer_mont();
        let kc = mkbuf(dev, "gf-kc", (kc_mont.len() * 4) as u64);
        dev_rc
            .dev
            .queue
            .write_buffer(&kc, 0, &u32s_to_bytes(&kc_mont));
        let uni = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gf-uni"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vectors = parse_vectors(VEC_SRC);
        let n_perm = vectors.len();
        let nl = MERKLE_LEAVES as u64;
        let inp = gen_lcg(N_NTT, 0x5eed1234abcd0001u64);
        let tw = twiddles(N_NTT, 0x4c734715u32);
        let n_tw = tw.len();
        let b = Rc::new(Bufs {
            kc,
            uni,
            in_buf: mkbuf(dev, "gf-perm-in", (n_perm * 16 * 4) as u64),
            rows: mkbuf(dev, "gf-rows", nl * 8 * 4),
            ab: [mkbuf(dev, "gf-a", nl * 8 * 4), mkbuf(dev, "gf-b", nl * 8 * 4)],
            n_buf: mkbuf(dev, "gf-ntt-in", (N_NTT * 4) as u64),
            tw_buf: mkbuf(dev, "gf-tw", (n_tw * 4) as u64),
            ntt_out: mkbuf(dev, "gf-ntt-out", (N_NTT * 4) as u64),
            n_perm,
            n_tw,
        });
        *slot.borrow_mut() = Some(b.clone());
        b
    })
}

pub async fn run_fast() -> Result<String, String> {
    let ctxr = ctx()?;
    let dev = &ctxr.dev.device;
    let queue = &ctxr.dev.queue;
    let mut out = String::new();
    let mut pass = true;

    // ---- CPU reference self-check (identical to classic) ----
    let vectors = parse_vectors(VEC_SRC);
    let p2 = Poseidon2::new();
    let mut cpu_ok = true;
    for (name, inp, exp) in &vectors {
        let got = p2.perm_canon(inp);
        if &to_hex(&got) != &to_hex(exp) {
            out.push_str(&format!("CPU_PERM {name}: FAIL\n"));
            cpu_ok = false;
        }
    }
    out.push_str(&format!(
        "CPU_REFERENCE_GOLDEN: {}\n",
        if cpu_ok { "PASS" } else { "FAIL" }
    ));
    if !cpu_ok {
        return Ok(out);
    }

    let bufs = bufs(&ctxr);
    ATLAS.with(|a| a.reset_counters());

    // ---- inputs ----
    let mut flat: Vec<u32> = Vec::new();
    for (_, inp, _) in &vectors {
        for i in 0..16 {
            flat.push(encode(inp[i]));
        }
    }
    queue.write_buffer(&bufs.in_buf, 0, &u32s_to_bytes(&flat));
    let inp = gen_lcg(N_NTT, 0x5eed1234abcd0001u64);
    let inp_mont: Vec<u32> = inp.iter().map(|&x| encode(x)).collect();
    let omega = 0x4c734715u32;
    let tw = twiddles(N_NTT, omega);
    queue.write_buffer(&bufs.n_buf, 0, &u32s_to_bytes(&inp_mont));
    queue.write_buffer(&bufs.tw_buf, 0, &u32s_to_bytes(&tw));

    let seed_f = (MERKLE_SEED % P as u64) as u32;
    let nl = MERKLE_LEAVES as u64;
    seed_row(queue, &bufs, seed_f, 0);

    // ---- one encoder, every dispatch of every leg, ONE submit ----
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gate-c1-fast"),
    });
    // perm leg (in-place on in_buf, io2 = dummy — lib semantics)
    {
        let bg = ATLAS.with(|a| a.bind(
            dev,
            &ctxr.ps.bgl,
            "gate-perm",
            &[
                Bind::whole(&bufs.kc),
                Bind::whole(&bufs.in_buf),
                Bind::whole(&ctxr.dummy),
                Bind::whole(&bufs.uni),
            ],
        ));
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&ctxr.ps.perm_batch);
        p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(bufs.n_perm.div_ceil(32) as u32, 1, 1);
    }
    // derive rows (single window: win() >> 4096*32 B)
    {
        let bg = ATLAS.with(|a| a.bind(
            dev,
            &ctxr.ps.bgl,
            "gate-derive",
            &[
                Bind::whole(&bufs.kc),
                Bind::slice(&bufs.rows, 0, nl * 32),
                Bind::whole(&ctxr.dummy),
                Bind::whole(&bufs.uni),
            ],
        ));
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&ctxr.ps.derive_row);
        p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups((nl as usize).div_ceil(32) as u32, 1, 1);
    }
    // merkle init -> ab[0]
    {
        let bg = ATLAS.with(|a| a.bind(
            dev,
            &ctxr.ps.bgl,
            "gate-init",
            &[
                Bind::whole(&bufs.kc),
                Bind::slice(&bufs.rows, 0, nl * 32),
                Bind::slice(&bufs.ab[0], 0, nl * 32),
                Bind::whole(&bufs.uni),
            ],
        ));
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&ctxr.ps.merkle_init);
        p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups((nl as usize).div_ceil(32) as u32, 1, 1);
    }
    // reduce levels ping-ponging ab[0]/ab[1]; atlas reuses level-pair groups
    // across the two slots whenever (size, buffers) repeat.
    let mut cur = 0usize;
    let mut cur_n = MERKLE_LEAVES;
    let mut lvl = 0;
    while cur_n > 1 {
        let out_nodes = cur_n / 2;
        let bg = ATLAS.with(|a| a.bind(
            dev,
            &ctxr.ps.bgl,
            "gate-lvl",
            &[
                Bind::whole(&bufs.kc),
                Bind::slice(&bufs.ab[cur], 0, (out_nodes * 64) as u64),
                Bind::slice(&bufs.ab[1 - cur], 0, (out_nodes * 32) as u64),
                Bind::whole(&bufs.uni),
            ],
        ));
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&ctxr.ps.merkle_compress);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(out_nodes.div_ceil(32) as u32, 1, 1);
        }
        cur = 1 - cur;
        cur_n = out_nodes;
        lvl += 1;
    }
    // NTT leg
    {
        let bg = ATLAS.with(|a| a.bind(
            dev,
            &ctxr.ntt_ps.bgl,
            "gate-ntt",
            &[
                Bind::whole(&bufs.n_buf),
                Bind::whole(&bufs.tw_buf),
                Bind::whole(&bufs.ntt_out),
            ],
        ));
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&ctxr.ntt_ps.fwd);
        p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(1, 1, 1);
    }
    let _ = lvl;
    queue.submit(Some(enc.finish()));

    // ---- ONE batched readback for all three legs ----
    let mut rb = ReadBatch::new();
    rb.push(&bufs.in_buf, 0, (bufs.n_perm * 16) as u64);
    rb.push(&bufs.ab[cur], 0, 8);
    rb.push(&bufs.ntt_out, 0, N_NTT as u64);
    let got = rb.resolve(dev, queue).await?;
    let (hits, misses) = ATLAS.with(|a| a.counters());
    out.push_str(&format!(
        "ATLAS bind hits={hits} misses={misses} maps={}\n",
        rb.nmaps()
    ));

    // ---- comparisons (identical output lines to classic) ----
    let got_words = &got[0];
    let mut out_canon: Vec<u32> = Vec::new();
    for (vi, (_, _, exp)) in vectors.iter().enumerate() {
        let mut g = [0u32; 16];
        for j in 0..16 {
            g[j] = decode(got_words[vi * 16 + j]);
        }
        let ok = to_hex(&g) == to_hex(exp);
        out.push_str(&format!(
            "GPU_PERM {}: {}\n",
            vectors[vi].0,
            if ok { "PASS" } else { "FAIL" }
        ));
        pass &= ok;
        out_canon.extend_from_slice(&g);
    }
    out.push_str(&format!(
        "GPU_PERM_OUTPUTS_SHA256: {}\n",
        sha256_hex(&u32s_to_bytes(&out_canon))
    ));

    let root_mont = &got[1];
    let mut root = [0u32; 8];
    for i in 0..8 {
        root[i] = decode(root_mont[i]);
    }
    let root_hex = to_hex(&root);
    let root_sha = sha256_hex(&u32s_to_bytes(&root));
    let ok_root = root_hex == MERKLE_ROOT_HEX;
    let ok_sha = root_sha == MERKLE_ROOT_SHA256;
    out.push_str(&format!(
        "GPU_MERKLE root={root_hex} exp={MERKLE_ROOT_HEX} [{}]\n",
        if ok_root { "PASS" } else { "FAIL" }
    ));
    out.push_str(&format!(
        "GPU_MERKLE sha256={root_sha} exp={MERKLE_ROOT_SHA256} [{}]\n",
        if ok_sha { "PASS" } else { "FAIL" }
    ));
    pass &= ok_root && ok_sha;

    const NTT_FIRST8: &str = "08407c0c33095dae29413c166d2c846a41695163328bd2fd00f11c08568aebab";
    const NTT_LAST8: &str = "481851ba74ad8f6f516167e7414acf100ca25fad5cd153ce3e6583f51a4f3fa0";
    const NTT_GOLDEN_SHA: &str = "a11b282970362f31b0d16a443e93a9171d5744ef657358da628b902e9835be0e";
    let out_cpu = ntt_dit(&inp, &tw);
    let cpu_first = to_hex(&out_cpu[0..8]);
    let cpu_last = to_hex(&out_cpu[N_NTT - 8..N_NTT]);
    let cpu_ok_ntt = cpu_first == NTT_FIRST8 && cpu_last == NTT_LAST8;
    out.push_str(&format!(
        "CPU_NTT first8={} match={} \n",
        cpu_first,
        if cpu_ok_ntt { "PASS" } else { "FAIL" }
    ));
    pass &= cpu_ok_ntt;

    let got_mont = &got[2];
    let got_canon: Vec<u32> = got_mont.iter().map(|&x| decode(x)).collect();
    let gpu_first = to_hex(&got_canon[0..8]);
    let gpu_last = to_hex(&got_canon[N_NTT - 8..N_NTT]);
    let mut nbad = 0usize;
    for i in 0..N_NTT {
        if got_canon[i] != out_cpu[i] {
            nbad += 1;
        }
    }
    let full_sha = sha256_hex(&u32s_to_bytes(&got_canon));
    let sha_ok = full_sha == NTT_GOLDEN_SHA;
    out.push_str(&format!(
        "GPU_NTT first8={} last8={}\n",
        gpu_first, gpu_last
    ));
    out.push_str(&format!(
        "GPU_NTT sha256={full_sha} exp={NTT_GOLDEN_SHA} [{}]\n",
        if sha_ok { "PASS" } else { "FAIL" }
    ));
    out.push_str(&format!(
        "GPU_NTT full correctness: {} ({} bad words vs CPU)\n",
        if nbad == 0 { "PASS" } else { "FAIL" },
        nbad
    ));
    pass &= sha_ok && nbad == 0;

    out.push_str(&format!(
        "GATE_C1_BATCHED: {}\n",
        if pass { "PASS" } else { "FAIL" }
    ));
    Ok(out)
}
