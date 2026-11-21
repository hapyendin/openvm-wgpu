# C3-APPLY report — fri_commit_test v6 on b70 (Arc Pro B70 / BMG G31)

Base report authored on an NVIDIA test box (validated there first); this file records
the Arc stamp of the same patch on b70 under kernel 7.2.6+deb13-amd64.

## ARC-STAMP — 2026-11-15 23:15–23:23 PST (b70, Mesa 26.1.6-1~bpo13+1, wgpu 30.0.2, Vulkan path)

**VERDICT: PASS — full golden bar met on Arc. Tree-of-record is now v6.**

| Item | Result |
|------|--------|
| File applied | `src/bin/fri_commit_test.rs` (v6, sha256 `21405a58ed796c420c94df5c163a9851643816f66e48075fa8858d705b153389`, 424 lines) |
| Backup of prior file | `wgpu-backend/backup/fri_commit_test.rs.pre-v6-20261115T2310` (sha256 `d46d56bc47f9…`, 312 lines) + existing `src/bin/fri_commit_test.rs.orig` |
| Build | `cargo build --release -j4` (rustup rustc 1.100.0), incremental 0.89s, **zero API deltas** — no get_mapped_range/write_buffer adaptation needed |
| Adapter | `Intel(R) Graphics (BMG G31)` vendor 0x8086 device 0xe223, Mesa 26.1.6, SHADER_INT64 enabled |
| GOLDEN_99FC8DF2_CPU | PASS (`99fc8df2bfc5e70f…616c1fda`) |
| GOLDEN_99FC8DF2_GPU | PASS (same sha) |
| roots0-5 (CPU + GPU) | 12/12 PASS vs embedded goldens |
| GPU_CPU_PARITY | PASS — 0/3008 words bad |
| Regression gate_c1 | PASS — GPU_MERKLE root `1264a0fd266c3e7b…` == expected; GATE_C1: PASS |
| Regression ntt_test | PASS — GPU_NTT sha `a11b282970362f31…` == expected; 0 bad words vs CPU |
| Sustained Vulkan load | 421 cycles x 3 runs = 1,263 device runs / 300s, 0 failures, xe=0 (concurrent with SYCL load block) |
| Kernel load-duty | xe-error lines in all leg windows: 0 (see prove/LOAD-DUTY.md; kernel uptime ~13.4k s at this point) |

## Notes / deltas vs task brief

1. Brief said tree pins wgpu **26.0.1**; actual `wgpu-backend/Cargo.toml` pins **=30.0.2**
   (= same minor v6 was authored against on the NVIDIA box) → zero port deltas. Nothing to adapt.
2. System `/usr/bin/cargo` (1.85.1) is too old for wgpu 30 (needs rustc >= 1.87); the tree
   builds with the host rustup toolchain (`~/.cargo/bin`, rustc 1.100.0). Workspace root is
   `~/appz/openvm-wgpu` (binaries land in `../target/release` relative to wgpu-backend).
3. `SHADER_INT64_ATOMIC_*` unavailable on this driver — tests correctly detect and skip that
   feature (gate_c1 prints U64_ATOMICS: UNAVAILABLE); no impact on the goldens.
4. All C3 runs logged as GPU legs with journalctl xe-counts: logs at
   `~/appz/openvm-sycl-skel/prove/logs/{c3v6-fri-vulkan,c3reg-gate_c1,c3reg-ntt,c3v6-vulkan-sustained}.log`.
