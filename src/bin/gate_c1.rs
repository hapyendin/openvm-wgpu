// GATE C1: b1 golden vectors through the wgpu backend's trait-shaped APIs.
// Correctness-only (miner may be running; no timing claims).
use wgpu_backend::field::{
    decode, encode, parse_vectors, sha256_hex, to_hex, Poseidon2, MERKLE_LEAVES,
    MERKLE_ROOT_HEX, MERKLE_ROOT_SHA256, MERKLE_SEED, P,
};
use wgpu_backend::traits::{
    DeviceDataTransporterShim, MatrixDimensions, Poseidon2ChipShim, ProverBackendShim,
    ProverDeviceShim,
};
use wgpu_backend::{InitOpts, WgpuBackend};

const VEC_SRC: &str = include_str!("../../goldens/vectors.inc");

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let farm_mode = argv.iter().any(|a| a == "--farm") || std::env::var("OPENVM_WGPU_FARM").is_ok();
    let opts = InitOpts {
        require_intel_arc: !farm_mode,
        allow_u32_only: farm_mode,
    };
    println!("MODE: {}", if farm_mode { "farm (any discrete GPU, u32 fallback ok)" } else { "b70 (Intel Arc assert, int64 path)" });

    // ---- 0) CPU reference self-check against goldens (must pass first) ----
    let vectors = parse_vectors(VEC_SRC);
    let p2 = Poseidon2::new();
    let mut cpu_ok = true;
    for (name, inp, out) in &vectors {
        let got = p2.perm_canon(inp);
        if &to_hex(&got) != &to_hex(out) {
            println!("CPU_PERM {name}: FAIL");
            cpu_ok = false;
        }
    }
    println!("CPU_REFERENCE_GOLDEN: {}", if cpu_ok { "PASS" } else { "FAIL" });
    if !cpu_ok {
        std::process::exit(1);
    }

    // ---- 1) device + adapter/feature/limits dump ----
    let backend = WgpuBackend::new(opts);
    println!(
        "REPORT: {:?}",
        &backend.dev.report
    );
    // ---- 2) u64 atomic probe (grind CAS / sumcheck add feasibility) ----
    match backend.dev.probe_u64_atomics() {
        None => println!(
            "U64_ATOMICS: UNAVAILABLE (feature SHADER_INT64_ATOMIC={}, SHADER_INT64={})",
            backend.dev.has_int64_atomic_feature, backend.dev.has_shader_int64
        ),
        Some((winners, lo, hi)) => {
            let v = ((hi as u64) << 32) | lo as u64;
            let base = 0xDEADBEEFCAFE1234u64;
            let ok = winners == 1 && v >= base + 224 && v <= base + 224 + 31;
            println!(
                "U64_ATOMICS: {} (cas_winners={winners} sum=0x{v:016x} expect base+winner+224)",
                if ok { "WORKING" } else { "WRONG-RESULTS" }
            );
        }
    }

    let mut pass = true;

    // ---- 3) GPU 3 golden perms through trait API ----
    let mut flat: Vec<u32> = Vec::new();
    for (_, inp, _) in &vectors {
        for i in 0..16 {
            flat.push(encode(inp[i]));
        }
    }
    let n = vectors.len();
    let mut m = <WgpuBackend as DeviceDataTransporterShim<WgpuBackend>>::transport_matrix_to_device(
        &backend, &flat, 16,
    );
    println!("MAT h={} w={} chunks={} len_words={}", m.height(), m.width(), m.chunks.len(), m.len_words);
    <WgpuBackend as Poseidon2ChipShim>::perm_batch(&backend, &mut m, n);
    let got_words =
        <WgpuBackend as DeviceDataTransporterShim<WgpuBackend>>::transport_matrix_from_device_to_host(
            &backend, &m,
        );
    let mut out_canon: Vec<u32> = Vec::new();
    for (vi, (_, _, exp)) in vectors.iter().enumerate() {
        let mut got = [0u32; 16];
        for j in 0..16 {
            got[j] = decode(got_words[vi * 16 + j]);
        }
        let hex = to_hex(&got);
        let ok = hex == to_hex(exp);
        println!("GPU_PERM {}: {}", vectors[vi].0, if ok { "PASS" } else { "FAIL" });
        pass &= ok;
        out_canon.extend_from_slice(&got);
    }
    println!("GPU_PERM_OUTPUTS_SHA256: {}", sha256_hex(&flat_bytes(&out_canon)));

    // ---- 4) 4096-leaf merkle root through trait API ----
    let seed_f = (MERKLE_SEED % P as u64) as u32;
    let root = wgpu_backend::traits::MerkleHasherShim::merkle_root_from_seed(&backend, seed_f, MERKLE_LEAVES);
    let root_hex = to_hex(&root);
    let root_sha = sha256_hex(&flat_bytes(&root));
    let ok_root = root_hex == MERKLE_ROOT_HEX;
    let ok_sha = root_sha == MERKLE_ROOT_SHA256;
    println!("GPU_MERKLE root={root_hex} exp={MERKLE_ROOT_HEX} [{}]", if ok_root { "PASS" } else { "FAIL" });
    println!("GPU_MERKLE sha256={root_sha} exp={MERKLE_ROOT_SHA256} [{}]", if ok_sha { "PASS" } else { "FAIL" });
    pass &= ok_root && ok_sha;

    // device_ctx trait touch (keeps trait honest)
    println!("DEVICE_CTX adapter={:?}", wgpu_backend::traits::ProverDeviceShim::device_ctx(&backend).adapter_name);

    println!("GATE_C1: {}", if pass { "PASS" } else { "FAIL" });
    std::process::exit(if pass { 0 } else { 2 });
}

fn flat_bytes(v: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for w in v {
        b.extend_from_slice(&w.to_le_bytes());
    }
    // sha256 over raw LE bytes (matches b1 merkle_test.cpp print_verdict)
    b
}
