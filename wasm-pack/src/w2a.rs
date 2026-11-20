//! W2A GKR leg IN-BROWSER (Phase 3c): the recorded gkr-inputs.json instance
//! + JS-XHR-fed leaf bytes drive the cfg-gated ASYNC GKR driver
//! (wgpu_backend::gkr::run_instance_async) inside headless Chromium WebGPU,
//! reproducing the native run-hash byte-exact (golden agg-leaf_0 =
//! fd53a33e82d7319b551a97c0710e5e75fac26980c0b6c1b02bd79bef0929f691).
//!
//! Inputs arrive via JS (file:// XHR, see mkpage_w2a.py); the driver's
//! fs/env debug knobs are naturally inert in the browser, so every readback
//! goes through the await-based mapAsync path (the ONLY browser-specific
//! machinery — same oneshot pattern as lib.rs::read_words).
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use wgpu_backend::gkr::{check_expected, run_instance_async, GkrInputs, GkrPipeline};

use crate::ctx;

thread_local! {
    static PIPE: RefCell<Option<Rc<GkrPipeline>>> = const { RefCell::new(None) };
}

/// Run one recorded GKR instance in the browser.
/// json: full gkr-inputs.json text (the same file the native w2a_gkr bin reads).
/// inst_name: which instance (e.g. "agg-leaf_0").
/// leaf: raw leaf bytes (dbg_<tag>_leaves_cpu.bin) from JS XHR.
/// Returns a W2A_* marker report; the JS page asserts the goldens.
#[wasm_bindgen(js_name = "w2aRun")]
pub async fn w2a_run(
    json: String,
    inst_name: String,
    leaf: Box<[u8]>,
) -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let dev = &c.dev;
    let pipe = match PIPE.with(|p| p.borrow().clone()) {
        Some(p) => p,
        None => {
            let fresh = Rc::new(GkrPipeline::new(dev));
            PIPE.with(|pp| *pp.borrow_mut() = Some(fresh.clone()));
            fresh
        }
    };
    let inputs: GkrInputs = serde_json::from_str(&json)
        .map_err(|e| JsValue::from_str(&format!("gkr-inputs parse: {e}")))?;
    let inst = inputs
        .instances
        .iter()
        .find(|i| i.name == inst_name)
        .ok_or_else(|| JsValue::from_str(&format!("instance {inst_name} not in json")))?;
    let exp_rounds = inst.rounds.len();
    let exp_subs: usize = inst.rounds.iter().map(|r| r.subs.len()).sum();
    let t0 = js_sys::Date::now();
    let mut snaps = Vec::new();
    let out =
        run_instance_async(dev, &pipe, inst, &leaf, std::path::Path::new("."), &mut snaps).await;
    let ms = js_sys::Date::now() - t0;
    let mismatches = check_expected(inst, &out);
    let rounds_done = out.rounds.len();
    let mut rep = String::new();
    rep.push_str(&format!(
        "W2A_INSTANCE {} t={} rounds={}/{} subs={}\n",
        out.name, out.t, rounds_done, exp_rounds, exp_subs
    ));
    rep.push_str(&format!("W2A_LEAF_BYTES {}\n", leaf.len()));
    rep.push_str(&format!("W2A_LEAF_SHA256 {}\n", out.leaf_sha256));
    rep.push_str(&format!("W2A_CLAIM_SHA256 {}\n", out.out_sha256));
    rep.push_str(&format!("W2A_CLAIM_LEN {}\n", out.out_sha256.len()));
    rep.push_str(&format!("W2A_VALUE_MISMATCHES {}\n", mismatches.len()));
    for m in mismatches.iter().take(5) {
        rep.push_str(&format!("W2A_MISMATCH {m}\n"));
    }
    rep.push_str(&format!("W2A_MS {ms:.1}\n"));
    Ok(rep)
}
