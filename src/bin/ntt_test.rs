// NTT test: verify 4096-pt NTT against b2 golden (a11b2829...)
// Uses the wgpu-probe golden values from b2 README
use wgpu_backend::{field, pipeline, NttPipelineSet, InitOpts, WgpuBackend};
use wgpu_backend::field::{encode, decode, to_hex, mont_mul, fadd};

const N: usize = 4096;

// LCG sequence generator (same as wgpu-probe)
fn gen_lcg(n: usize, seed: u64) -> Vec<u32> {
    let mut v = Vec::with_capacity(n);
    let mut s = seed;
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        v.push((s >> 32) as u32);
    }
    v
}

// Fast DIT NTT (bitrev input gather, natural output) - mirrors WGSL kernel
fn ntt_dit(inp: &[u32], tw: &[u32]) -> Vec<u32> {
    let n = inp.len();
    let lg = n.trailing_zeros() as usize;
    let mut data = vec![0u32; n];
    // bit-reverse input order
    for k in 0..n {
        let brk = br_bits(k as u32, lg as u32);
        data[brk as usize] = encode(inp[k]);
    }
    let mut off = 0usize;
    for st in 1..=lg {
        let half = 1usize << st;
        let q1 = half >> 1;
        let cnt = 1usize << (st - 1);
        for i in 0..(n / 2) {
            let b = i & (q1 - 1);
            let blk = i >> (st - 1);
            let pp = blk * half + b;
            let qq = pp + q1;
            let w = tw[off + b];
            let t = mont_mul(data[qq], w);
            let u = data[pp];
            data[pp] = fadd(u, t);
            data[qq] = fsub(u, t);
        }
        off += cnt;
    }
    data.iter().map(|&x| decode(x)).collect()
}

fn br_bits(x: u32, bits: u32) -> u32 {
    let mut r = 0u32;
    for i in 0..bits {
        if (x >> i) & 1 == 1 {
            r |= 1 << (bits - 1 - i);
        }
    }
    r
}

fn fsub(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { a + (2013265921u32 - b) }
}

// Twiddle table (Montgomery words)
fn twiddles(n: usize, omega_canon: u32) -> Vec<u32> {
    let lg = n.trailing_zeros() as usize;
    let w_m = encode(omega_canon);
    let mut out = Vec::with_capacity(n - 1);
    for st in 1..=lg {
        let step = n >> st;
        let mut wpow = vec![encode(1); step * (1 << (st - 1))];
        for k in 1..wpow.len() {
            wpow[k] = mont_mul(wpow[k - 1], w_m);
        }
        for b in 0..(1usize << (st - 1)) {
            out.push(wpow[b * step]);
        }
    }
    out
}

fn main() {
    // Input: LCG sequence
    let inp = gen_lcg(N, 0x5eed1234abcd0001u64);
    let inp_mont: Vec<u32> = inp.iter().map(|&x| encode(x)).collect();
    
    // Omega: TWO_ADIC_GENERATORS[12] = 0x4c734715 (canonical)
    let omega = 0x4c734715u32;
    let tw = twiddles(N, omega);
    
    // Compute CPU reference
    let out_cpu = ntt_dit(&inp, &tw);
    
    // Golden values from b2 README: a11b282970362f31b0d16a443e93a9171d5744ef657358da628b902e9835be0e
    const NTT_FIRST8: &str = "08407c0c33095dae29413c166d2c846a41695163328bd2fd00f11c08568aebab";
    const NTT_LAST8: &str = "481851ba74ad8f6f516167e7414acf100ca25fad5cd153ce3e6583f51a4f3fa0";
    
    let got_first8 = to_hex(&out_cpu[0..8]);
    let got_last8 = to_hex(&out_cpu[N-8..N]);
    println!("CPU_NTT first8={} last8={}", got_first8, got_last8);
    println!("CPU_NTT first8 match: {} | last8 match: {}",
        if got_first8 == NTT_FIRST8 { "PASS" } else { "FAIL" },
        if got_last8 == NTT_LAST8 { "PASS" } else { "FAIL" });
    
    if got_first8 != NTT_FIRST8 || got_last8 != NTT_LAST8 {
        println!("CPU_NTT: FAILED - golden mismatch");
        std::process::exit(1);
    }
    
    // GPU test
    let opts = InitOpts {
        require_intel_arc: true,
        allow_u32_only: false,
        device_substring: wgpu_backend::adapters::device_substring_from_env(),
    };
    
    let backend = WgpuBackend::new(opts);
    println!("GPU NTT test starting...");
    
    let ntt_ps = NttPipelineSet::new(&backend.dev, backend.dev.has_shader_int64);
    
    let n_buf = backend.dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ntt_in"),
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let tw_buf = backend.dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ntt_tw"),
        size: (tw.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let out_buf = backend.dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ntt_out"),
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let staging = backend.dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ntt_staging"),
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    backend.dev.queue.write_buffer(&n_buf, 0, &field::u32s_to_bytes(&inp_mont));
    backend.dev.queue.write_buffer(&tw_buf, 0, &field::u32s_to_bytes(&tw));
    
    let bg = backend.dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &ntt_ps.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: n_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tw_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
        ],
    });
    
    pipeline::dispatch_ntt(&backend.dev, &ntt_ps.fwd, &bg);
    
    {
        let mut enc = backend.dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (N * 4) as u64);
        backend.dev.queue.submit(Some(enc.finish()));
    }
    
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    backend.dev.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    let data = slice.get_mapped_range().unwrap();
    let got_mont: Vec<u32> = data.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    drop(data);
    staging.unmap();
    
    // Decode to canonical
    let got_canon: Vec<u32> = got_mont.iter().map(|&x| decode(x)).collect();
    
    let got_first8_gpu = to_hex(&got_canon[0..8]);
    let got_last8_gpu = to_hex(&got_canon[N-8..N]);
    
    println!("GPU_NTT first8={} last8={}", got_first8_gpu, got_last8_gpu);
    println!("GPU_NTT first8 match: {} | last8 match: {}",
        if got_first8_gpu == NTT_FIRST8 { "PASS" } else { "FAIL" },
        if got_last8_gpu == NTT_LAST8 { "PASS" } else { "FAIL" });
    
    let mut bad = 0;
    for i in 0..N {
        if got_canon[i] != out_cpu[i] {
            bad += 1;
        }
    }
    
    let full_sha = field::sha256_hex(&field::u32s_to_bytes(&got_canon));
    const NTT_GOLDEN_SHA: &str = "a11b282970362f31b0d16a443e93a9171d5744ef657358da628b902e9835be0e";
    println!("GPU_NTT sha256={}", full_sha);
    println!("GPU_NTT sha256 match: {}",
        if full_sha == NTT_GOLDEN_SHA { "PASS" } else { "FAIL" });
    
    println!("GPU_NTT full correctness: {} ({} bad words vs CPU)",
        if bad == 0 { "PASS" } else { "FAIL" }, bad);
    
    if bad == 0 && full_sha == NTT_GOLDEN_SHA {
        println!("NTT_TEST: PASS");
        std::process::exit(0);
    } else {
        println!("NTT_TEST: FAIL");
        std::process::exit(1);
    }
}
