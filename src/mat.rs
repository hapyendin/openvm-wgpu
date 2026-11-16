// DeviceData analogue: u32 field-word matrices over chunked storage buffers.
// Rationale (C1-ARCH §a): CUDA backend uses D<T>/DMatrix over cudaMallocAsync
// pool + VMM; wgpu has no VMM, but storage buffers are flat + cheap — we
// replace the pool with direct buffers and the VMM contiguity trick with
// explicit CHUNKING, since max_storage_buffer_binding_size caps a single
// binding at ~2 GiB (measured 2147483644 on ANV).
use crate::device::WgpuDevice;
use crate::field::u32s_to_bytes;

/// Hard ceiling per binding; keep 250 MiB headroom below the adapter cap for
/// future binding-header growth and alignment.
pub const CHUNK_HEADROOM_BYTES: u64 = 250 * 1024 * 1024;

pub struct WgpuMat {
    pub height: usize,
    pub width: usize, // cells per row (field elements)
    pub chunks: Vec<wgpu::Buffer>,
    pub words_per_chunk: usize,
    pub len_words: usize,
}

impl WgpuMat {
    fn chunk_words(dev: &WgpuDevice, len_words: usize) -> usize {
        let cap = dev
            .report
            .limits
            .max_storage_buffer_binding_size
            .saturating_sub(CHUNK_HEADROOM_BYTES)
            / 4;
        (cap as usize).min(len_words.max(1)).max(1024)
    }

    fn chunk_len(&self, i: usize) -> usize {
        (self.len_words - i * self.words_per_chunk).min(self.words_per_chunk)
    }

    fn mk(dev: &WgpuDevice, words_per_chunk: usize, len_words: usize) -> Vec<wgpu::Buffer> {
        let nchunks = len_words.div_ceil(words_per_chunk).max(1);
        (0..nchunks)
            .map(|i| {
                let take = (len_words - i * words_per_chunk).min(words_per_chunk);
                dev.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("mat chunk {i}")),
                    size: (take.max(1) * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect()
    }

    /// zeros(h, w): like CUDA-side zeros matrices; no host round-trip.
    pub fn zeros(dev: &WgpuDevice, height: usize, width: usize) -> WgpuMat {
        let len_words = height * width;
        let wpc = Self::chunk_words(dev, len_words);
        WgpuMat {
            height,
            width,
            chunks: Self::mk(dev, wpc, len_words),
            words_per_chunk: wpc,
            len_words,
        }
    }

    /// transport_matrix_to_device analogue (host col-major words -> device).
    pub fn from_host(dev: &WgpuDevice, words: &[u32], width: usize) -> WgpuMat {
        let mut m = Self::zeros(dev, words.len() / width.max(1), width);
        m.copy_in(dev, words);
        m
    }

    pub fn copy_in(&mut self, dev: &WgpuDevice, words: &[u32]) {
        assert!(words.len() <= self.len_words);
        let mut off = 0usize;
        for (ci, c) in self.chunks.iter().enumerate() {
            let take = (words.len() - off).min(self.words_per_chunk);
            if take == 0 {
                break;
            }
            let bytes = &words[off..off + take];
            // chunk capacity >= words_per_chunk; pad last write to chunk size?
            // write only `take` words.
            dev.queue.write_buffer(c, 0, &u32s_to_bytes(bytes));
            let _ = ci;
            off += take;
        }
        assert_eq!(off, words.len());
    }

    /// transport_matrix_from_device_to_host analogue: full readback
    /// (single chunk fast path; generic multi-chunk loop).
    pub fn copy_out(&self, dev: &WgpuDevice) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.len_words);
        for (ci, c) in self.chunks.iter().enumerate() {
            let take = self.chunk_len(ci);
            if take == 0 {
                break;
            }
            let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("mat out staging {ci}")),
                size: (take * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            {
                let mut enc = dev
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                enc.copy_buffer_to_buffer(c, 0, &staging, 0, (take * 4) as u64);
                dev.queue.submit(Some(enc.finish()));
            }
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| ());
            dev.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .expect("poll");
            let data = slice.get_mapped_range().expect("map");
            out.extend(data.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
            drop(data);
            staging.unmap();
        }
        out
    }
}
