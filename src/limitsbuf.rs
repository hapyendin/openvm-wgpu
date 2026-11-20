// ShardBuf: one logical GPU buffer, split into device-limit-legal shards.
//
// Under OPENVM_WGPU_WEBGPU_LIMITS=1 every storage binding must be <= 128
// MiB and every buffer <= 256 MiB, so the big prove-path buffers (GKR
// levels/pool/eq, WHIR gcoeffs, merkle rows, logup leaf regions) are
// built as Vec<Buffer> shards. Windows bound into bind groups are
// pow2-sized and pow2-offset (all row counts/strides here are pow2), so
// a window of <= win() bytes starting at a pow2-aligned offset never
// straddles a shard (shard sizes are pow2 >= win()). The host asserts
// this on every bind — a straddle is a hard panic, not silent damage.
use std::num::NonZeroU64;

use crate::device::{pow2_floor, WgpuDevice};

pub struct ShardBuf {
    pub shards: Vec<wgpu::Buffer>,
    pub shard_bytes: u64,
    pub size: u64,
}

fn shard_size(dev: &WgpuDevice, size: u64) -> u64 {
    // pow2 floor of min(size-rounded-up, device buffer cap, 1 GiB)
    let cap = pow2_floor(dev.buf_cap()).min(1u64 << 30);
    if size <= cap {
        // pow2 CEIL so any buffer <= cap is ONE shard: pow2-floor shards made
        // small buffers (e.g. 4476B table blobs -> 4096B shards) unbindable
        // for full-range binds (w2d lowmem tables air).
        (size.max(4)).next_power_of_two()
    } else {
        cap
    }
}

impl ShardBuf {
    pub fn new(dev: &WgpuDevice, label: &str, usage: wgpu::BufferUsages, size: u64) -> ShardBuf {
        assert!(size > 0, "ShardBuf {label}: zero size");
        let shard_bytes = shard_size(dev, size);
        let n = size.div_ceil(shard_bytes);
        let mut shards = Vec::with_capacity(n as usize);
        for i in 0..n {
            let sz = shard_bytes.min(size - i * shard_bytes);
            shards.push(dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}/s{i}")),
                size: sz,
                usage,
                mapped_at_creation: false,
            }));
        }
        ShardBuf { shards, shard_bytes, size }
    }

    pub fn nshards(&self) -> usize {
        self.shards.len()
    }

    /// Locate (shard index, offset in shard) for a logical byte offset.
    pub fn loc(&self, off: u64) -> (usize, u64) {
        (
            (off / self.shard_bytes) as usize,
            off % self.shard_bytes,
        )
    }

    /// Binding for the logical range [off, off+len). Requires:
    /// len <= device binding cap and full containment in one shard.
    pub fn bind(&self, dev: &WgpuDevice, off: u64, len: u64) -> wgpu::BindingResource<'_> {
        assert!(len > 0, "bind len 0");
        assert!(
            len <= dev.report.limits.max_storage_buffer_binding_size,
            "bind {len} bytes exceeds binding cap {}",
            dev.report.limits.max_storage_buffer_binding_size
        );
        assert!(off + len <= self.size, "bind range past end");
        let (i, o) = self.loc(off);
        assert!(
            o + len <= self.shards[i].size(),
            "bind [{off},{}) straddles shard boundary (shard_bytes={}, shard {i} size={})",
            off + len,
            self.shard_bytes,
            self.shards[i].size()
        );
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.shards[i],
            offset: o,
            size: Some(unsafe { NonZeroU64::new_unchecked(len) }),
        })
    }

    /// Whole-shard binding (for small shard 0 etc.).
    pub fn bind_shard(&self, i: usize) -> wgpu::BindingResource<'_> {
        self.shards[i].as_entire_binding()
    }

    /// Queue-write bytes at a logical offset, splitting across shards.
    pub fn write_slice(&self, dev: &WgpuDevice, off: u64, bytes: &[u8]) {
        assert!(off + bytes.len() as u64 <= self.size, "write past end");
        let mut done = 0usize;
        while done < bytes.len() {
            let o = off + done as u64;
            let (i, so) = self.loc(o);
            let take = bytes.len() - done;
            let take = (take as u64).min(self.shards[i].size() - so) as usize;
            dev.queue.write_buffer(&self.shards[i], so, &bytes[done..done + take]);
            done += take;
        }
    }

    /// Copy [word_off, word_off+nwords) u32 words from a (sub)range into
    /// this buffer at logical byte offset 0. Source may be any shard range.
    pub fn copy_words_from_shard(
        &self,
        dev: &WgpuDevice,
        src: &ShardBuf,
        src_word_off: u64,
        nwords: u64,
        dst_word_off: u64,
    ) {
        let mut done = 0u64;
        while done < nwords {
            let so = (src_word_off + done) * 4;
            let doff = (dst_word_off + done) * 4;
            let (i, sso) = src.loc(so);
            let (j, dso) = self.loc(doff);
            let take = (nwords - done)
                .min((src.shards[i].size() - sso) / 4)
                .min((self.shards[j].size() - dso) / 4);
            if take == 0 {
                break;
            }
            let mut enc = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("shard-copy"),
                });
            enc.copy_buffer_to_buffer(&src.shards[i], sso, &self.shards[j], dso, take * 4);
            dev.queue.submit(Some(enc.finish()));
            done += take;
        }
        assert_eq!(done, nwords, "copy_words_from_shard stalled");
    }

    /// Device-copy u32 words from one raw buffer into this buffer.
    pub fn copy_words_from_buf(
        &self,
        dev: &WgpuDevice,
        src: &wgpu::Buffer,
        src_word_off: u64,
        nwords: u64,
        dst_word_off: u64,
    ) {
        let mut done = 0u64;
        while done < nwords {
            let so = (src_word_off + done) * 4;
            let doff = (dst_word_off + done) * 4;
            let (j, dso) = self.loc(doff);
            let take = (nwords - done).min((src.size() - so) / 4).min((self.shards[j].size() - dso) / 4);
            if take == 0 {
                break;
            }
            let mut enc = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("shard-copy-buf"),
                });
            enc.copy_buffer_to_buffer(src, so, &self.shards[j], dso, take * 4);
            dev.queue.submit(Some(enc.finish()));
            done += take;
        }
        assert_eq!(done, nwords, "copy_words_from_buf stalled");
    }

    /// Read nwords u32 words starting at a logical word offset (blocking).
    pub fn read_words(&self, dev: &WgpuDevice, word_off: u64, nwords: u64) -> Vec<u32> {
        let mut out = vec![0u32; nwords as usize];
        let mut done = 0u64;
        while done < nwords {
            let o = (word_off + done) * 4;
            let (i, so) = self.loc(o);
            let take = (nwords - done).min((self.shards[i].size() - so) / 4);
            if take == 0 {
                break;
            }
            let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shard-read"),
                size: take * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&self.shards[i], so, &staging, 0, take * 4);
            dev.queue.submit(Some(enc.finish()));
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| ());
            dev.device
                .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                .expect("shard read poll");
            {
                let data = slice.get_mapped_range().expect("map");
                for k in 0..take as usize {
                    out[(done + k as u64) as usize] =
                        u32::from_le_bytes(data[4 * k..4 * k + 4].try_into().unwrap());
                }
            }
            staging.unmap();
            done += take;
        }
        assert_eq!(done, nwords, "read_words stalled");
        out
    }

    pub fn read_all(&self, dev: &WgpuDevice) -> Vec<u32> {
        self.read_words(dev, 0, self.size / 4)
    }

    /// W2A-BROWSER: awaitable twin of read_words. Native branch identical to
    /// the sync body (map_async + poll(Wait)); wasm32 awaits the mapAsync
    /// callback via a oneshot (device.poll is a NO-OP on the WebGPU backend).
    pub async fn read_words_async(
        &self,
        dev: &WgpuDevice,
        word_off: u64,
        nwords: u64,
    ) -> Vec<u32> {
        let mut out = vec![0u32; nwords as usize];
        let mut done = 0u64;
        while done < nwords {
            let o = (word_off + done) * 4;
            let (i, so) = self.loc(o);
            let take = (nwords - done).min((self.shards[i].size() - so) / 4);
            if take == 0 {
                break;
            }
            let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shard-read"),
                size: take * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&self.shards[i], so, &staging, 0, take * 4);
            dev.queue.submit(Some(enc.finish()));
            let slice = staging.slice(..);
            #[cfg(not(target_arch = "wasm32"))]
            {
                slice.map_async(wgpu::MapMode::Read, |_| ());
                dev.device
                    .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                    .expect("shard read poll");
            }
            #[cfg(target_arch = "wasm32")]
            {
                let (tx, rx) =
                    futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                rx.await.map_err(|e| e.to_string()).and_then(|r| r.map_err(|e| e.to_string()))
                    .expect("shard read map");
            }
            {
                let data = slice.get_mapped_range().expect("map");
                for k in 0..take as usize {
                    out[(done + k as u64) as usize] =
                        u32::from_le_bytes(data[4 * k..4 * k + 4].try_into().unwrap());
                }
            }
            staging.unmap();
            done += take;
        }
        assert_eq!(done, nwords, "read_words_async stalled");
        out
    }

    pub async fn read_all_async(&self, dev: &WgpuDevice) -> Vec<u32> {
        self.read_words_async(dev, 0, self.size / 4).await
    }
}
