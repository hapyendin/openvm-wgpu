//! W2C WHIR-fold leg IN-BROWSER (Phase 3d): per-instance CPU fold-plan +
//! GPU coeffs (pre-mobius on CPU, as the native bin plans) -> DIT-FFT grs ->
//! GPU merkle commit — the exact op order of src/bin/w2c_fold.rs,
//! reproducing the native goldens: agg-leaf_0 r0 grs sha
//! 5b91a9e38bfa47fb633ce4ee564a0175f6f360c1e7d8b0e3f7e607e98547777c / r1
//! 28948222729453e28e950f1dd4c6c2435c1d77521cd4dd88266ea521b744c1e8
//! byte-exact, plus every per-round commit root from
//! w2c-results-r1-GREEN.json.
//!
//! The page owns the file:// XHRs (async; see §9 quirks) and slices the
//! gcoeffs/finalcoeffs dumps down to the needed prefix before pushing bytes
//! into these resumable exports. grs goldens are verified by sha: the page
//! hashes the grs dump with crypto.subtle and compares to the sha reported
//! here (both are sha256 over the CANON little-endian word stream, which is
//! exactly the dump file's byte content).

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use wgpu_backend::device::WgpuDevice;
use wgpu_backend::field::{decode, encode, mont_mul, sha256_hex, u32s_to_bytes};
use wgpu_backend::limitsbuf::ShardBuf;
use wgpu_backend::w2c::*;
use wgpu_backend::readfix::ReadFix;
use wgpu_backend::whir_fold::{self, WhirFold};

use crate::ctx;

thread_local! {
    static WFS: RefCell<Option<Rc<WhirFold>>> = const { RefCell::new(None) };
    static RFIX: RefCell<Option<Rc<ReadFix>>> = const { RefCell::new(None) };
    static ST: RefCell<Option<InstState>> = const { RefCell::new(None) };
}

struct InstState {
    k_whir: usize,
    nr: usize,
    log_ns: Vec<u32>,
    coeffs: Vec<Vec<Ef>>,
}

fn rfix() -> Result<Rc<ReadFix>, JsValue> {
    RFIX.with(|p| {
        let mut b = p.borrow_mut();
        if b.is_none() {
            let c = ctx().map_err(|e| JsValue::from_str(&e))?;
            *b = Some(Rc::new(ReadFix::new(&c.dev)));
        }
        Ok(b.clone().unwrap())
    })
}

fn wf() -> Result<Rc<WhirFold>, JsValue> {
    WFS.with(|p| {
        let mut b = p.borrow_mut();
        if b.is_none() {
            let c = ctx().map_err(|e| JsValue::from_str(&e))?;
            *b = Some(Rc::new(WhirFold::new(&c.dev)));
        }
        Ok(b.clone().unwrap())
    })
}

fn to_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn raw_ef(c: &[u32]) -> Vec<Ef> {
    c.chunks_exact(4)
        .map(|x| [encode(x[0]), encode(x[1]), encode(x[2]), encode(x[3])])
        .collect()
}
fn flatten(e: &[Ef]) -> Vec<u32> {
    e.iter().flat_map(|r| r.iter().copied()).collect()
}
fn unflatten(w: &[u32]) -> Vec<Ef> {
    w.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}
fn canon_ef(e: &[Ef]) -> Vec<u32> {
    e.iter().flat_map(|x| x.iter().map(|&w| decode(w))).collect()
}
fn powm(mut b: u32, mut e: u64) -> u32 {
    let mut r = encode(1);
    while e > 0 {
        if e & 1 == 1 {
            r = mont_mul(r, b);
        }
        b = mont_mul(b, b);
        e >>= 1;
    }
    r
}

/// Awaitable gpu_fft: CPU pad + bitrev permutation, GPU DIT stages, then
/// AWAITED readback (device.poll is a no-op on the web backend).
async fn gpu_fft_async(dev: &WgpuDevice, wf: &WhirFold, rf: &ReadFix, coeffs: &[Ef], log_n: u32) -> Vec<Ef> {
    let n = 1usize << log_n;
    let mut rows: Vec<Ef> = Vec::with_capacity(n);
    rows.extend_from_slice(coeffs);
    rows.resize(n, ef_zero());
    let mut perm: Vec<Ef> = vec![ef_zero(); n];
    for i in 0..n {
        let j = ((i as u32).reverse_bits() >> (32 - log_n)) as usize;
        perm[j] = rows[i];
    }
    let data = whir_fold::data_shard(dev, n, "fft in");
    data.write_slice(dev, 0, &u32s_to_bytes(&flatten(&perm)));
    let mut tws: Vec<wgpu::Buffer> = Vec::new();
    let mut half = 1usize;
    while half < n {
        let wm = powm(root_of_unity_raw(log_n), (n / (half * 2)) as u64);
        let mut tw = vec![0u32; half];
        tw[0] = encode(1);
        for k in 1..half {
            tw[k] = mont_mul(tw[k - 1], wm);
        }
        tws.push(wf.buf(dev, &tw, "tw"));
        half *= 2;
    }
    whir_fold::whir_stages(dev, wf, &data, n, 0, &tws);
    // §7c: COPY right after queue.write_buffer sees PRE-write bytes on the
    // web backend; dispatch-copy first (ReadFix), then map the result.
    let words = rf.read_words_after_write(dev, &data, 0, (n * 4) as u64).await;
    unflatten(&words)
}

/// Awaitable gpu_merkle: rows [row4|0^4] -> commit root (8 canonical u32).
async fn gpu_merkle_async(
    dev: &WgpuDevice,
    wf: &WhirFold,
    grs: &[Ef],
    k_whir: usize,
) -> [u32; 8] {
    let n = grs.len();
    let mut rows8: Vec<u32> = Vec::with_capacity(n * 8);
    for r in grs {
        rows8.extend_from_slice(r);
        rows8.extend_from_slice(&[0u32; 4]);
    }
    let rows_b = ShardBuf::new(
        dev,
        "m rows",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        (n * 32) as u64,
    );
    rows_b.write_slice(dev, 0, &u32s_to_bytes(&rows8));
    whir_fold::merkle_from_rows_async(dev, wf, &rows_b, n, k_whir).await
}

/// Begin one instance: CPU fold-plan (exact twin of the native bin's plan
/// loop: k_whir folds per whir round, then post-mobius) and return the
/// per-round table the page uses to slice golden dumps + feed rounds.
#[wasm_bindgen(js_name = "w2cInstBegin")]
pub fn w2c_inst_begin(
    k_whir: u32,
    log_rs0: u32,
    nr: u32,
    f0: Box<[u8]>,
    alphas: Box<[u8]>,
) -> Result<String, JsValue> {
    let k = k_whir as usize;
    let nr = nr as usize;
    let mut f = raw_ef(&to_u32s(&f0));
    let alphas = raw_ef(&to_u32s(&alphas));
    let mut aidx = 0usize;
    let mut coeffs: Vec<Vec<Ef>> = Vec::new();
    for _r in 0..nr {
        for _s in 0..k {
            let a = *alphas
                .get(aidx)
                .ok_or_else(|| JsValue::from_str("alphas exhausted"))?;
            aidx += 1;
            let half = f.len() / 2;
            let mut nf = vec![ef_zero(); half];
            for y in 0..half {
                let e0 = f[2 * y];
                let d = ef_sub(&f[2 * y + 1], &e0);
                nf[y] = ef_add(&e0, &ef_mul(&d, &a));
            }
            f = nf;
        }
        let mut c = f.clone();
        for l in 0..4 {
            let mut col: Vec<u32> = c.iter().map(|e| e[l]).collect();
            mobius_inplace(&mut col);
            for (j, x) in col.iter().enumerate() {
                c[j][l] = *x;
            }
        }
        coeffs.push(c);
    }
    let log_ns: Vec<u32> = (0..nr).map(|r| log_rs0.saturating_sub(r as u32 + 1)).collect();
    let mut tbl = String::from("{\"rounds\":[");
    for r in 0..nr {
        if r > 0 {
            tbl.push(',');
        }
        let rows = coeffs[r].len();
        let is_final = r + 1 == nr;
        tbl.push_str(&format!(
            "{{\"rows\":{rows},\"log_n\":{},\"prefix_bytes\":{},\"is_final\":{is_final}}}",
            log_ns[r],
            rows * 16
        ));
    }
    tbl.push_str("]}");
    ST.with(|s| *s.borrow_mut() = Some(InstState { k_whir: k, nr, log_ns, coeffs }));
    Ok(tbl)
}

/// One whir round. `gc_prefix` = the golden gcoeffs/finalcoeffs dump sliced
/// to rows*16 bytes; `grs_sha_exp` = page-side sha256 hex of the golden grs
/// dump ("" on the final round). Runs the GPU FFT + merkle (non-final) with
/// awaited readbacks and reports PASS/FAIL + first mismatch word.
#[wasm_bindgen(js_name = "w2cRound")]
pub async fn w2c_round(r: u32, gc_prefix: Box<[u8]>, grs_exp: Box<[u8]>) -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let wf = wf()?;
    let (k_whir, nr, log_n, coeffs_r) = ST.with(|s| {
        let st = s.borrow();
        let st = st.as_ref().ok_or_else(|| JsValue::from_str("w2cInstBegin first"))?;
        let r = r as usize;
        if r >= st.nr {
            return Err(JsValue::from_str("round out of range"));
        }
        Ok::<(usize, usize, u32, Vec<Ef>), JsValue>((
            st.k_whir,
            st.nr,
            st.log_ns[r],
            st.coeffs[r].clone(),
        ))
    })?;
    let is_final = r as usize + 1 == nr;
    let cc = canon_ef(&coeffs_r);
    let cc_bytes = u32s_to_bytes(&cc);
    let gc_ok = gc_prefix.len() >= cc_bytes.len() && gc_prefix[..cc_bytes.len()] == cc_bytes[..];
    let mut rep = format!("W2C_R{r} GCOEFFS_OK {gc_ok}");
    if !gc_ok {
        let d = gc_prefix
            .iter()
            .zip(cc_bytes.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(usize::MAX);
        rep.push_str(&format!(" GCOEFFS_FIRSTBAD_BYTE {d}"));
    }
    if is_final {
        rep.push_str(" FINAL 1");
        ST.with(|s| s.borrow_mut().take());
        return Ok(rep);
    }
    let t0 = js_sys::Date::now();
    let rf = rfix()?;
    let mut fftin = coeffs_r;
    fftin.resize(1usize << log_n, ef_zero());
    let grs_gpu = gpu_fft_async(&c.dev, &wf, &rf, &fftin, log_n).await;
    let gc_canon = canon_ef(&grs_gpu);
    let grs_sha = sha256_hex(&u32s_to_bytes(&gc_canon));
    let expw = to_u32s(&grs_exp);
    let (grs_ok, mismatch) = if grs_exp.is_empty() {
        (false, String::new())
    } else {
        let bad = gc_canon.iter().zip(expw.iter()).position(|(a, b)| a != b);
        let msg = match bad {
            Some(d) => format!(
                " FIRSTBAD {d} gpu={:08x} exp={:08x} gpu0=[{:08x},{:08x},{:08x},{:08x}] exp0=[{:08x},{:08x},{:08x},{:08x}]",
                gc_canon[d], expw[d], gc_canon[0], gc_canon[1], gc_canon[2], gc_canon[3],
                expw[0], expw[1], expw[2], expw[3]),
            None if gc_canon.len() != expw.len() =>
                format!(" LEN gpu={} exp={}", gc_canon.len(), expw.len()),
            None => String::new(),
        };
        (bad.is_none() && gc_canon.len() == expw.len(), msg)
    };
    let root = gpu_merkle_async(&c.dev, &wf, &grs_gpu, k_whir).await;
    let ms = js_sys::Date::now() - t0;
    rep.push_str(&format!(
        " GRSSHA {grs_sha} GRS_OK {grs_ok}{mismatch} ROOT [{}] ROUND_MS {ms:.0}",
        root.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(",")
    ));
    if r as usize + 1 == nr {
        ST.with(|s| s.borrow_mut().take());
    }
    Ok(rep)
}

/// In-browser twin of the native bin's --selftest: known coeffs -> CPU
/// column-FFT vs GPU DIT stages, for log_n in {3, 10, 14, 16}.
#[wasm_bindgen(js_name = "w2cSelftest")]
pub async fn w2c_selftest() -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let wf = wf()?;
    let mut rep = String::new();
    for &ln in &[3u32, 10, 14, 16] {
        let n = 1usize << ln;
        let raw: Vec<u32> = (0..n * 4).map(|i| (i as u32).wrapping_mul(2654435761) | 1).collect();
        let coeffs = raw_ef(&raw);
        let mut cpu_flat = flatten(&coeffs);
        let mut cols: Vec<Vec<u32>> = vec![vec![0u32; n]; 4];
        for (r, e) in coeffs.iter().enumerate() {
            for l in 0..4 {
                cols[l][r] = e[l];
            }
        }
        for l in 0..4 {
            fft_inplace(&mut cols[l]);
            for r in 0..n {
                cpu_flat[r * 4 + l] = cols[l][r];
            }
        }
        let rf = rfix()?;
        // ECHO: same shard, write_slice then dispatch-copy readback, NO
        // stages -> isolates write-visibility from stage math.
        let n = 1usize << ln;
        let echoshard = whir_fold::data_shard(&c.dev, n, "echo");
        echoshard.write_slice(&c.dev, 0, &u32s_to_bytes(&flatten(&coeffs)));
        let echo = rf.read_words_after_write(&c.dev, &echoshard, 0, (n * 4) as u64).await;
        let enbad = echo.iter().zip(flatten(&coeffs).iter()).filter(|(a, b)| a != b).count();
        // MOBIUS: mode-1 stages (no twiddles, only add/sub) -> isolates
        // uni+stage addressing from the mul/twiddle path.
        let mshard = whir_fold::data_shard(&c.dev, n, "mobt");
        mshard.write_slice(&c.dev, 0, &u32s_to_bytes(&flatten(&coeffs)));
        whir_fold::whir_stages(&c.dev, &wf, &mshard, n, 1, &[]);
        let mwords = rf.read_words_after_write(&c.dev, &mshard, 0, (n * 4) as u64).await;
        let mut mcpu = flatten(&coeffs);
        {
            let mut rows: Vec<Ef> = unflatten(&mcpu);
            for l in 0..4 {
                let mut col: Vec<u32> = rows.iter().map(|e| e[l]).collect();
                mobius_inplace(&mut col);
                for (r, x) in col.iter().enumerate() {
                    rows[r][l] = *x;
                }
            }
            mcpu = flatten(&rows);
        }
        let mbad = mwords.iter().zip(mcpu.iter()).filter(|(a, b)| a != b).count();
        let gpu = flatten(&gpu_fft_async(&c.dev, &wf, &rf, &coeffs, ln).await);
        let nbad = gpu.iter().zip(cpu_flat.iter()).filter(|(a, b)| a != b).count();
        let firstbad = gpu.iter().zip(cpu_flat.iter()).position(|(a, b)| a != b);
        rep.push_str(&format!("SELFTEST ln={ln} echobad={enbad} mbad={mbad} m0=[{:08x}] mc0=[{:08x}] nbad={nbad} firstbad={firstbad:?} gpu0=[{:08x},{:08x}] cpu0=[{:08x},{:08x}]\n",
            mwords[0], mcpu[0], gpu[0], gpu[1], cpu_flat[0], cpu_flat[1]));
    }
    Ok(rep)
}
