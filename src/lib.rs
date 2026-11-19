pub mod adapters;
pub mod device;
pub mod field;
pub mod gkr;
pub mod gkr_wgsl;
pub mod limitsbuf;
pub mod logup_leaf;
pub mod mat;
pub mod pipeline;
pub mod traits;
pub mod wgsl;
pub mod w2c;
pub mod whir_fold;

pub use device::{InitOpts, WgpuDevice};
pub use mat::WgpuMat;
pub use pipeline::{KernelCtx, NttPipelineSet, PipelineSet};

use crate::field::{constants_buffer_mont, decode, Poseidon2, MERKLE_SEED};
use crate::traits::*;

/// BabyBear backend over wgpu (FORK-C). Analogous to GenericGpuBackend<HS>
/// + GpuDevice pair in openvm-cuda-backend, collapsed since the hash scheme
/// is fixed to Poseidon2-BabyBear in C2.
pub struct WgpuBackend {
    pub dev: WgpuDevice,
    pub ps: PipelineSet,
    pub kctx: KernelCtx,
    pub dummy: wgpu::Buffer,
    pub cpu: Poseidon2, // CPU reference (golden checks)
    pub ntt_ps: NttPipelineSet,
}

#[derive(Clone)]
pub struct WgpuCtx {
    pub adapter_name: String,
}

pub struct DummyCommitter;

impl WgpuBackend {
    pub fn new(opts: InitOpts) -> WgpuBackend {
        let dev = WgpuDevice::init(opts);
        let u64_mul = dev.has_shader_int64;
        let ps = PipelineSet::new(&dev, u64_mul);
        let kctx = KernelCtx::new(&dev, &constants_buffer_mont());
        let dummy = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ntt_ps = NttPipelineSet::new(&dev, u64_mul);
        WgpuBackend {
            dev,
            ps,
            kctx,
            dummy,
            cpu: Poseidon2::new(),
            ntt_ps,
        }
    }

    fn dispatch_chunked(&self, pipe: &wgpu::ComputePipeline, m: &WgpuMat, words: usize) {
        // Phase-2: multi-chunk dispatch. Every 16-word thread is fully
        // independent (perm_batch guards on arrayLength(io1), so per-chunk
        // binds need no offset math); chunks are row-aligned pow2.
        let _ = words;
        for (ci, c) in m.chunks.iter().enumerate() {
            let take = m.chunk_len(ci);
            if take == 0 {
                break;
            }
            let bg = self.kctx.bind(&self.dev, &self.ps, c, &self.dummy);
            let threads = take / 16; // perm_batch: 16 words/thread
            pipeline::dispatch(&self.dev, pipe, &bg, threads.max(1));
        }
    }
    
    /// FRI commit 8-col layer (Layer 0): hash each 8-col row
    /// Input: n_rows x 8 words (row-major)
    /// Output: n_rows x 8-word digests (canonical)
    pub fn fri_commit_8col(&self, input: &WgpuMat, output: &WgpuMat) {
        assert_eq!(input.width, 8, "fri_commit_8col requires 8-col input");
        assert_eq!(output.width, 8, "fri_commit_8col produces 8-word digests");
        assert_eq!(input.height, output.height, "input/output row count mismatch");
        let n_rows = input.height;
        let bg = self
            .kctx
            .bind(&self.dev, &self.ps, &input.chunks[0], &output.chunks[0]);
        pipeline::dispatch(&self.dev, &self.ps.fri_commit_8col, &bg, n_rows);
    }
    
    /// FRI commit 1-col layer (Layer 1-5): hash 1-col padded to 8
    /// Input: n_rows x 1 word (column-major, padded to 8)
    /// Output: n_rows x 8-word digests (first word = hash, rest = 0)
    pub fn fri_commit_1col(&self, input: &WgpuMat, output: &WgpuMat) {
        assert_eq!(input.width, 1, "fri_commit_1col requires 1-col input");
        assert_eq!(output.width, 8, "fri_commit_1col produces 8-word digests");
        assert_eq!(input.height, output.height, "input/output row count mismatch");
        let n_rows = input.height;
        let bg = self
            .kctx
            .bind(&self.dev, &self.ps, &input.chunks[0], &output.chunks[0]);
        pipeline::dispatch(&self.dev, &self.ps.fri_commit_1col, &bg, n_rows);
    }
}

impl ProverBackendShim for WgpuBackend {
    type Val = u32;
    type Challenge = u32;
    type Commitment = Vec<u32>;
    type Matrix = WgpuMat;
    type PcsData = ();
}

impl MatrixDimensions for WgpuMat {
    fn height(&self) -> usize {
        self.height
    }
    fn width(&self) -> usize {
        self.width
    }
}

impl ProverDeviceShim for WgpuBackend {
    type DeviceCtx = WgpuCtx;
    fn device_ctx(&self) -> &WgpuCtx {
        // leaked-once static: fine for a singleton device per process
        static CTX: std::sync::OnceLock<WgpuCtx> = std::sync::OnceLock::new();
        CTX.get_or_init(|| WgpuCtx {
            adapter_name: self.dev.report.name.clone(),
        })
    }
}

impl DeviceDataTransporterShim<WgpuBackend> for WgpuBackend {
    fn transport_matrix_to_device(&self, host: &[u32], width: usize) -> WgpuMat {
        WgpuMat::from_host(&self.dev, host, width)
    }
    fn transport_matrix_from_device_to_host(&self, m: &WgpuMat) -> Vec<u32> {
        m.copy_out(&self.dev)
    }
}

impl Poseidon2ChipShim for WgpuBackend {
    fn perm_batch(&self, m: &mut WgpuMat, _n_states: usize) {
        self.dispatch_chunked(&self.ps.perm_batch, m, m.len_words);
    }
}

impl MerkleHasherShim for WgpuBackend {
    /// GPU legs: derive_row -> merkle_init -> merkle_compress (WGSL paths
    /// from wgpu-probe + b1 ported kernels). Phase-2: buffers are ShardBufs
    /// and every bind is a pow2-aligned window <= device binding cap, so
    /// the pass runs under WebGPU spec-floor limits unchanged.
    fn merkle_root_from_seed(&self, seed_canon: u32, num_leaves: usize) -> [u32; 8] {
        use crate::limitsbuf::ShardBuf;
        assert_eq!(seed_canon, (MERKLE_SEED % crate::field::P as u64) as u32);
        let n = num_leaves;
        let us = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let dev = &self.dev;
        // rows: n x 8 mont words (zero-init not needed: derive writes all)
        let rows = ShardBuf::new(dev, "merkle-rows", us, (n * 8 * 4) as u64);
        let wr = (dev.win() / 32).min(n as u64).max(1); // rows per derive/init window
        for w0 in (0..n as u64).step_by(wr as usize) {
            let cnt = wr.min(n as u64 - w0);
            self.kctx.set_seed_row(dev, seed_canon, w0 as u32);
            let bg = self.kctx.bind_res(
                dev,
                &self.ps,
                rows.bind(dev, w0 * 32, cnt * 32),
                self.dummy.as_entire_binding(),
            );
            pipeline::dispatch(dev, &self.ps.derive_row, &bg, cnt as usize);
        }
        // leaves = init(rows)
        let mut cur = ShardBuf::new(dev, "merkle-leaves", us, (n * 8 * 4) as u64);
        for w0 in (0..n as u64).step_by(wr as usize) {
            let cnt = wr.min(n as u64 - w0);
            let bg = self.kctx.bind_res(
                dev,
                &self.ps,
                rows.bind(dev, w0 * 32, cnt * 32),
                cur.bind(dev, w0 * 32, cnt * 32),
            );
            pipeline::dispatch(dev, &self.ps.merkle_init, &bg, cnt as usize);
        }
        // compress levels: window of out-rows W: read 2W rows => W <= win/64
        let mut cur_n = n;
        while cur_n > 1 {
            let out_nodes = cur_n / 2;
            let nxt = ShardBuf::new(dev, "merkle-nxt", us, (out_nodes * 8 * 4) as u64);
            let wo = (dev.win() / 64).min(out_nodes as u64).max(1);
            for w0 in (0..out_nodes as u64).step_by(wo as usize) {
                let cnt = wo.min(out_nodes as u64 - w0);
                let bg = self.kctx.bind_res(
                    dev,
                    &self.ps,
                    cur.bind(dev, w0 * 64, cnt * 64),
                    nxt.bind(dev, w0 * 32, cnt * 32),
                );
                pipeline::dispatch(dev, &self.ps.merkle_compress, &bg, cnt as usize);
            }
            cur = nxt;
            cur_n = out_nodes;
        }
        let root_mont = cur.read_words(dev, 0, 8);
        let mut root = [0u32; 8];
        for i in 0..8 {
            root[i] = decode(root_mont[i]);
        }
        root
    }
}

impl TraceCommitterShim<WgpuBackend> for DummyCommitter {
    fn commit(&self, _traces: &[&WgpuMat]) -> (Vec<u32>, ()) {
        // LDE-NTT + row-hash commit = b2/b3-equivalent milestones; not in C2.
        unimplemented!("commit path scheduled for the b7-equivalent WGSL milestone")
    }
}
