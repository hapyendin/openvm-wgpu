//! W2B logup-leaf leg IN-BROWSER (Phase 3d): the cpu9 leaf world (log_stack
//! 26, 23 trace files ~0.93 GB) + gkr-inputs-w2b.json drive the resumable
//! leaf builder pieces (wgpu_backend::logup_leaf::{leaf_begin,air_start,
//! air_feed_tables,air_feed_trace,air_run,leaf_sha_async}) and the ASYNC GKR
//! driver (run_instance_gpu_async) inside headless Chromium WebGPU,
//! reproducing the native goldens BYTE-EXACT:
//!   leaf sha f5ecc63de1646b064ff943f9bd59cbea0b2ec799a73dc5c33cafe3342d4a965f
//!   out sha  8c06f8ac2b4070d68b46f2a7d2112855d354c91584b3edeeb374e84a8e370a63
//!
//! File I/O quirk (§7c + 3d): responseType='arraybuffer' is ILLEGAL for
//! SYNCHRONOUS XHR from a document and Chromium IGNORES overrideMimeType
//! x-user-defined (decodes as UTF-8 -> bytes mangled). So the page does the
//! proven ASYNC file:// XHRs and feeds one file at a time into these
//! resumable exports; the per-air pipeline state lives in a thread_local.

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use wgpu_backend::field::sha256_hex;
use wgpu_backend::gkr::{check_expected, leaf_chunks, run_instance_gpu_async, GkrInputs, GkrPipeline};
use wgpu_backend::logup_leaf::{
    self, air_feed_tables, air_feed_trace, air_run, air_start, LeafPipeline, LeafWorld,
};

use crate::ctx;

thread_local! {
    static PIPES: RefCell<Option<Rc<(GkrPipeline, LeafPipeline)>>> = const { RefCell::new(None) };
    static WORLD: RefCell<Option<Rc<LeafWorld>>> = const { RefCell::new(None) };
    static BUILD: RefCell<Option<logup_leaf::LeafBuildState>> = const { RefCell::new(None) };
    static LEAF_SHA: RefCell<String> = const { RefCell::new(String::new()) };
}

fn pipes() -> Rc<(GkrPipeline, LeafPipeline)> {
    PIPES.with(|p| {
        let mut b = p.borrow_mut();
        if b.is_none() {
            let c = ctx().expect("initWgpu first");
            *b = Some(Rc::new((GkrPipeline::new(&c.dev), LeafPipeline::new(&c.dev))));
        }
        b.clone().unwrap()
    })
}

/// Start ONE leaf-build pass: parse the world, create pipelines, allocate +
/// default-fill the 2 GiB leaf, and return the per-air input manifest
/// (JSON) the page must feed back via air_feed_tables/air_feed_trace.
#[wasm_bindgen(js_name = "w2bPlan")]
pub fn w2b_plan(world_json: String) -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let world: LeafWorld = serde_json::from_str(&world_json)
        .map_err(|e| JsValue::from_str(&format!("leafworld parse: {e}")))?;
    let world = Rc::new(world);
    WORLD.with(|w| *w.borrow_mut() = Some(world.clone()));
    let pipes = pipes();
    let st = logup_leaf::leaf_begin(&c.dev, &pipes.1, &world, leaf_chunks(&c.dev, world.log_stack).1);
    BUILD.with(|b| *b.borrow_mut() = Some(st));
    let mut manifest = String::from("[");
    for (i, air) in world.airs.iter().enumerate() {
        if i > 0 {
            manifest.push(',');
        }
        manifest.push_str(&format!(
            "{{\"i\":{i},\"air\":{},\"tables\":\"{}\",\"traces\":[{}]}}",
            air.air_id,
            air.tables,
            air.trace_files
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    manifest.push(']');
    Ok(manifest)
}

#[wasm_bindgen(js_name = "w2bFeedTables")]
pub fn w2b_feed_tables(air_idx: u32, tables: Box<[u8]>) -> Result<(), JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let world = WORLD.with(|w| w.borrow().clone()).ok_or_else(|| JsValue::from_str("w2bPlan first"))?;
    BUILD.with(|bb| {
        let mut b = bb.borrow_mut();
        let st = b.as_mut().ok_or_else(|| JsValue::from_str("w2bPlan first"))?;
        air_start(st, &c.dev, &world, air_idx as usize);
        air_feed_tables(st, &c.dev, &world, tables.into_vec());
        Ok(())
    })
}

#[wasm_bindgen(js_name = "w2bFeedTrace")]
pub fn w2b_feed_trace(air_idx: u32, raw: Box<[u8]>) -> Result<(), JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let world = WORLD.with(|w| w.borrow().clone()).ok_or_else(|| JsValue::from_str("w2bPlan first"))?;
    BUILD.with(|bb| {
        let mut b = bb.borrow_mut();
        let st = b.as_mut().ok_or_else(|| JsValue::from_str("w2bPlan first"))?;
        let _ = air_idx;
        air_feed_trace(st, &c.dev, &world, raw.into_vec());
        Ok(())
    })
}

#[wasm_bindgen(js_name = "w2bAirRun")]
pub fn w2b_air_run(air_idx: u32) -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let world = WORLD.with(|w| w.borrow().clone()).ok_or_else(|| JsValue::from_str("w2bPlan first"))?;
    let pipes = pipes();
    BUILD.with(|bb| {
        let mut b = bb.borrow_mut();
        let st = b.as_mut().ok_or_else(|| JsValue::from_str("w2bPlan first"))?;
        air_run(st, &c.dev, &pipes.1, &world);
        Ok(format!("W2B_AIR {air_idx} done\n"))
    })
}

/// Canonical sha256 of the built leaf (awaited mapAsync readback).
#[wasm_bindgen(js_name = "w2bLeafSha")]
pub async fn w2b_leaf_sha() -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let st = BUILD.with(|bb| bb.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("w2bPlan first"))?;
    let sha = logup_leaf::leaf_sha_async(&c.dev, &st).await;
    BUILD.with(|bb| *bb.borrow_mut() = Some(st));
    LEAF_SHA.with(|s| *s.borrow_mut() = sha.clone());
    Ok(sha)
}

/// Run the GKR rounds on the built leaf and compare the per-layer /
/// concat golden streams (same table as the native w2b_logup bin).
#[wasm_bindgen(js_name = "w2bGkr")]
pub async fn w2b_gkr(
    run_no: u32,
    gkr_json: String,
    shas_json: String,
) -> Result<String, JsValue> {
    let c = ctx().map_err(|e| JsValue::from_str(&e))?;
    let pipes = pipes();
    let leaf_bufs = BUILD.with(|bb| bb.borrow_mut().as_mut().map(|st| std::mem::take(&mut st.leaf)))
        .ok_or_else(|| JsValue::from_str("w2bPlan first"))?;
    if leaf_bufs.is_empty() {
        return Err(JsValue::from_str("leaf already consumed (re-plan)"));
    }
    let leaf_sha = LEAF_SHA.with(|s| s.borrow().clone());
    let inputs: GkrInputs = serde_json::from_str(&gkr_json)
        .map_err(|e| JsValue::from_str(&format!("gkr-inputs parse: {e}")))?;
    let shas: serde_json::Value = serde_json::from_str(&shas_json)
        .map_err(|e| JsValue::from_str(&format!("layer-shas parse: {e}")))?;
    if inputs.instances.len() != 1 {
        return Err(JsValue::from_str(&format!(
            "expected 1 instance, got {}",
            inputs.instances.len()
        )));
    }
    let inst = &inputs.instances[0];
    let gold_layers = shas["layers"].as_array().unwrap().clone();
    let gold_concat = shas["concat_sha256"].as_str().unwrap_or("").to_string();

    let t0 = js_sys::Date::now();
    let mut snaps = Vec::new();
    let out = run_instance_gpu_async(
        &c.dev, &pipes.0, inst, leaf_bufs, leaf_sha.clone(), std::path::Path::new("."), &mut snaps,
    )
    .await;
    let ms = js_sys::Date::now() - t0;
    let mismatches = check_expected(inst, &out);

    let mut layer_ok = true;
    let mut bad_layer = String::new();
    let mut concat_bytes: Vec<u8> = Vec::new();
    for l in 0..=(inst.t - 1) {
        let groups: Vec<&[u32]> = if l == 0 {
            vec![&out.root_q, &out.claims0]
        } else {
            let r = &out.rounds[l as usize - 1];
            let mut g: Vec<&[u32]> = Vec::with_capacity(r.subs.len() + 1);
            for s in &r.subs {
                g.push(s.as_slice());
            }
            g.push(r.claims.as_slice());
            g
        };
        let bytes = quads_bytes(&groups);
        concat_bytes.extend_from_slice(&bytes);
        let sha = sha256_hex(&bytes);
        let exp = gold_layers[l as usize]["sha"].as_str().unwrap_or("");
        let exp_n = gold_layers[l as usize]["n"].as_u64().unwrap_or(0) as usize;
        if &sha[..16] != exp || exp_n != bytes.len() / 16 {
            layer_ok = false;
            if bad_layer.is_empty() {
                bad_layer = format!("layer {l} gpu={} exp={exp} n={}", &sha[..16], bytes.len() / 16);
            }
        }
    }
    let concat_gpu = sha256_hex(&concat_bytes);
    let mut rep = String::new();
    rep.push_str(&format!(
        "W2B_RUN {run_no} LEAF_SHA256 {leaf_sha} OUT_SHA256 {}\n",
        out.out_sha256
    ));
    rep.push_str(&format!(
        "W2B_RUN {run_no} ROUNDS {}/{} MISMATCHES {}\n",
        out.rounds.len(),
        inst.rounds.len(),
        mismatches.len()
    ));
    for m in mismatches.iter().take(4) {
        rep.push_str(&format!("W2B_RUN {run_no} MISMATCH {m}\n"));
    }
    rep.push_str(&format!(
        "W2B_RUN {run_no} LAYERS {} BAD {bad_layer}\n",
        if layer_ok { "OK" } else { "FAIL" }
    ));
    rep.push_str(&format!(
        "W2B_RUN {run_no} CONCAT {concat_gpu} CONCAT_OK {}\n",
        concat_gpu == gold_concat
    ));
    rep.push_str(&format!("W2B_RUN {run_no} GKR_MS {ms:.0}\n"));
    Ok(rep)
}

fn quads_bytes(groups: &[&[u32]]) -> Vec<u8> {
    let mut v = Vec::new();
    for g in groups {
        assert_eq!(g.len() % 4, 0);
        for w in *g {
            v.extend_from_slice(&w.to_le_bytes());
        }
    }
    v
}

/// W2D: hand the built leaf buffers to the w2d module (one-shot move).
pub fn take_leaf() -> Vec<wgpu::Buffer> {
    BUILD.with(|bb| {
        bb.borrow_mut()
            .as_mut()
            .map(|st| std::mem::take(&mut st.leaf))
            .unwrap_or_default()
    })
}

pub fn last_leaf_sha() -> String {
    LEAF_SHA.with(|s| s.borrow().clone())
}
