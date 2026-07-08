#!/usr/bin/env python3
"""READ-ONLY degeneracy check for the plain stage-1 scene-flow checkpoint.

Question answered:
  Did the plain stage-1 checkpoint (VLM + cam_branch + scene_predictor flow-matching
  DiT, action head frozen/random) actually learn to PREDICT scene flow, or is its low
  training flow_loss (~0.005) a degenerate "predict ~zero flow" solution?

Decisive test = DEGENERACY BASELINE COMPARISON on dynamic-and-valid grid cells:
  - Model EPE          = mean ||pred_flow - gt_flow||_2   (sampled via Euler flow-matching ODE)
  - Zero-flow EPE      = mean ||gt_flow||_2                (error of predicting all-zeros)
  - Relative EPE       = Model / Zero   (<<1 = learned real motion; ~1 = degenerate)
  - Cosine(pred, gt)   on dynamic cells (~0 = no direction learned; >0.5 = tracks motion)

Faithfully replicates the training paths:
  - VLM encode under bf16 autocast -> last_hidden (== vl_embs for scene_predictor)
  - _collect_scene_targets + head._pool_target / _supervision_mask gridding (16x16)
  - scene DiT forward under fp32 autocast (training's fp32 island), torch.no_grad
Sampling = Euler-integrate dx/dt = pred_v(x,t) from x=noise (t=0) to t=1, N uniform steps.

Nothing is written except a matplotlib PNG + a small metrics JSON into --out-dir.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf  # noqa: E402
from starVLA.model.framework.share_tools import apply_config_compat  # noqa: E402


def load_ckpt_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint is not a state_dict-like dict: {type(ckpt).__name__}")
    return ckpt


def build_model(cfg, device):
    from starVLA.model.framework.base_framework import build_framework

    model = build_framework(cfg)
    model.to(device)
    model.eval()
    return model


def encode_vl(model, examples, device):
    """Replicate QwenGR00T.forward's encode block EXACTLY: bf16 autocast -> hidden_states[-1]."""
    images = [e["image"] for e in examples]
    instructions = [e["lang"] for e in examples]
    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
    backbone_attention_mask = qwen_inputs.get("attention_mask", None)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.qwen_vl_interface(
            **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
        )
        last_hidden = out.hidden_states[-1]
    enc_mask = backbone_attention_mask.to(torch.bool) if backbone_attention_mask is not None else None
    return last_hidden, enc_mask


def grid_gt(head, flow_batch, device):
    """Replicate head.forward's gridding: pooled target (B,256,3) + sup / valid masks (B,256)."""
    field_gt = flow_batch[head.target_key].to(device)  # (B,3,Hgt,Wgt) fp32
    valid = flow_batch.get(head.valid_key, None)
    dynamic = flow_batch.get(head.dynamic_key, None)
    if valid is None:
        valid = torch.ones(field_gt.shape[0], 1, field_gt.shape[2], field_gt.shape[3], device=device)
    valid = valid.to(device)
    if valid.dim() == 3:
        valid = valid.unsqueeze(1)
    if dynamic is not None:
        dynamic = dynamic.to(device)
        if dynamic.dim() == 3:
            dynamic = dynamic.unsqueeze(1)

    target_grid, valid_grid = head._pool_target(field_gt, valid)  # (B,3,g,g),(B,1,g,g)
    dyn_grid = None
    if dynamic is not None:
        dyn_grid = F.adaptive_avg_pool2d(dynamic.float(), (head.grid_size, head.grid_size)) > 0.5
    sup = head._supervision_mask(valid_grid, dyn_grid)  # (B,1,g,g) bool
    if "flow_has_gt" in flow_batch:
        sup = sup & flow_batch["flow_has_gt"].to(sup.device).view(-1, 1, 1, 1)

    target = target_grid.flatten(2).permute(0, 2, 1).contiguous()  # (B,256,3)
    sup_flat = sup.reshape(sup.shape[0], -1)  # (B,256) bool
    valid_flat = valid_grid.reshape(valid_grid.shape[0], -1)  # (B,256) bool
    return target, sup_flat, valid_flat


@torch.no_grad()
def sample_flow(head, vl_embs, enc_mask, n_steps):
    """Euler-integrate dx/dt = pred_v(x,t), x0=noise(t=0) -> x1(t=1). Returns (B,256,3) fp32."""
    B = vl_embs.shape[0]
    device = vl_embs.device
    x = torch.randn(B, head.num_tokens, head.field_dim, device=device, dtype=torch.float32)
    dt = 1.0 / n_steps
    with torch.autocast("cuda", dtype=torch.float32):
        for i in range(n_steps):
            t_val = i * dt
            t_cont = torch.full((B,), t_val, device=device, dtype=torch.float32)
            t_disc = (t_cont * head.num_timestep_buckets).long().clamp_(max=head.num_timestep_buckets - 1)
            feats = head._encode(x, t_disc)
            out = head.model(
                hidden_states=feats,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=enc_mask,
                timestep=t_disc,
            )
            pred_v = head.field_decoder(out).float()
            x = x + dt * pred_v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="qwen0p8_groot_cascade_motion_cambranch_learn_scene_flow_fromscratch_leftprimary")
    ap.add_argument("--ckpt-step", type=int, default=30000)
    ap.add_argument("--n-samples", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-euler-steps", type=int, default=10)
    ap.add_argument("--out-dir", default="docs/experiments/stage1_flow_prediction_check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FATAL] CUDA required", flush=True)
        sys.exit(2)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_root = ROOT / "playground" / "Checkpoints" / args.run_id
    cfg_path = ckpt_root / "config.full.yaml"
    ckpt_path = ckpt_root / "checkpoints" / f"steps_{args.ckpt_step}_pytorch_model.pt"
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cfg]  {cfg_path}", flush=True)
    print(f"[ckpt] {ckpt_path}", flush=True)
    print(f"[out]  {out_dir}", flush=True)

    cfg = OmegaConf.load(cfg_path)
    _compat = apply_config_compat(cfg)
    if _compat is not None:
        cfg = _compat
    # be defensive: some dataset code paths read task_id (train __main__ sets it)
    OmegaConf.update(cfg, "datasets.vla_data.task_id", "all", force_add=True)
    OmegaConf.update(cfg, "output_dir", str(out_dir), force_add=True)

    print("[build] constructing QwenGR00T + scene_predictor (loads Qwen VLM)...", flush=True)
    model = build_model(cfg, device)
    sp = getattr(model, "scene_predictor", None)
    if sp is None:
        print("[FATAL] scene_predictor not built (framework.scene_predictor.enabled?)", flush=True)
        sys.exit(1)

    # ---- load the stage-1 checkpoint (bypass the custom warm-start audit; we WANT the
    #      trained scene_predictor + cam_branch + VLM keys loaded, all present in ckpt).
    state = load_ckpt_state(str(ckpt_path))
    ret = torch.nn.Module.load_state_dict(model, state, strict=False)
    missing = list(ret.missing_keys)
    unexpected = list(ret.unexpected_keys)
    sp_missing = [k for k in missing if k.startswith("scene_predictor.")]
    vlm_missing = [k for k in missing if k.startswith("qwen_vl_interface.")]
    cam_missing = [k for k in missing if k.startswith("stereo_cam_branch_layers_modules.") or ".stereo_cam_layer." in k]
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    print(f"[load] scene_predictor missing={len(sp_missing)}  qwen_vl missing={len(vlm_missing)}  cam_branch missing={len(cam_missing)}", flush=True)
    if sp_missing:
        print(f"[FATAL] scene_predictor keys MISSING from checkpoint (would be random!): {sp_missing[:8]}", flush=True)
        sys.exit(1)
    if vlm_missing:
        print(f"[WARN] {len(vlm_missing)} qwen_vl keys missing e.g. {vlm_missing[:4]}", flush=True)
    if cam_missing:
        print(f"[WARN] {len(cam_missing)} cam_branch keys missing e.g. {cam_missing[:4]}", flush=True)
    model.eval()

    # ---- build the real training dataset (gt_only_sampler -> only GT frames) ----
    from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    from torch.utils.data import DataLoader

    print(f"[data] building dataset mix={cfg.datasets.vla_data.data_mix} (gt_only_sampler)...", flush=True)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)

    print(f"[head] grid={sp.grid_size} field_dim={sp.field_dim} num_tokens={sp.num_tokens} "
          f"num_timestep_buckets={sp.num_timestep_buckets} dynamic_fallback_to_valid={sp.dynamic_fallback_to_valid}", flush=True)

    # accumulators
    P_sup, G_sup = [], []          # (Ncells, 3) over dynamic-and-valid cells
    P_val, G_val = [], []          # (Ncells, 3) over all-valid cells
    static_mag, dynamic_mag = [], []   # ||G|| on static(valid & !sup) vs dynamic(sup) cells
    n_total_cells = 0
    n_valid_cells = 0
    n_sup_cells = 0
    n_samples = 0
    n_fell_to_zero_sup = 0         # samples with zero dynamic-and-valid cells
    n_imgs_per_sample = None
    per_sample_viz = []            # (target16, pred16, supmask16, epe, cos, magmean)

    for examples in loader:
        if n_samples >= args.n_samples:
            break
        # drop samples with no GT (shouldn't happen under gt_only, but be safe for _collect_scene_targets all-or-none)
        examples = [e for e in examples if bool(e.get("flow_has_gt", False))]
        if not examples:
            continue
        if n_imgs_per_sample is None:
            n_imgs_per_sample = len(examples[0]["image"])
        B = len(examples)

        vl_embs, enc_mask = encode_vl(model, examples, device)
        flow_batch = model._collect_scene_targets(examples, device)
        if flow_batch is None:
            continue
        target, sup_flat, valid_flat = grid_gt(sp, flow_batch, device)  # (B,256,3),(B,256),(B,256)
        pred = sample_flow(sp, vl_embs, enc_mask, args.n_euler_steps)   # (B,256,3)

        target = target.float()
        pred = pred.float()

        for b in range(B):
            g = target[b]           # (256,3)
            p = pred[b]             # (256,3)
            sup_b = sup_flat[b]     # (256,)
            val_b = valid_flat[b]   # (256,)
            gmag = g.norm(dim=-1)   # (256,)

            n_total_cells += 256
            n_valid_cells += int(val_b.sum().item())
            ns = int(sup_b.sum().item())
            n_sup_cells += ns
            if ns == 0:
                n_fell_to_zero_sup += 1

            # magnitude distributions
            static_sel = val_b & (~sup_b)
            if static_sel.any():
                static_mag.append(gmag[static_sel].detach().cpu())
            if sup_b.any():
                dynamic_mag.append(gmag[sup_b].detach().cpu())

            # metrics accumulators
            if sup_b.any():
                P_sup.append(p[sup_b].detach().cpu())
                G_sup.append(g[sup_b].detach().cpu())
            if val_b.any():
                P_val.append(p[val_b].detach().cpu())
                G_val.append(g[val_b].detach().cpu())

            # per-sample record for visualization (only samples with motion)
            if ns > 0:
                epe = (p[sup_b] - g[sup_b]).norm(dim=-1).mean().item()
                pc = p[sup_b]
                gc = g[sup_b]
                cos = F.cosine_similarity(pc, gc, dim=-1).mean().item()
                per_sample_viz.append({
                    "target16": g.reshape(16, 16, 3).detach().cpu().numpy(),
                    "pred16": p.reshape(16, 16, 3).detach().cpu().numpy(),
                    "sup16": sup_b.reshape(16, 16).detach().cpu().numpy(),
                    "epe": epe, "cos": cos, "ncells": ns,
                    "magmean": gmag[sup_b].mean().item(),
                })
            n_samples += 1

        print(f"  collected {n_samples} samples | sup_cells so far={n_sup_cells}", flush=True)

    # ---------------- METRICS ----------------
    def epe(P, G):
        return (P - G).norm(dim=-1).mean().item()

    P_sup_t = torch.cat(P_sup, 0) if P_sup else torch.zeros(0, 3)
    G_sup_t = torch.cat(G_sup, 0) if G_sup else torch.zeros(0, 3)
    P_val_t = torch.cat(P_val, 0) if P_val else torch.zeros(0, 3)
    G_val_t = torch.cat(G_val, 0) if G_val else torch.zeros(0, 3)

    results = {}
    if G_sup_t.shape[0] > 0:
        model_epe_sup = epe(P_sup_t, G_sup_t)
        zero_epe_sup = G_sup_t.norm(dim=-1).mean().item()
        meanvec_sup = G_sup_t.mean(dim=0, keepdim=True)               # (1,3)
        mean_epe_sup = (meanvec_sup - G_sup_t).norm(dim=-1).mean().item()
        cos_sup = F.cosine_similarity(P_sup_t, G_sup_t, dim=-1).mean().item()
        rel_epe_sup = model_epe_sup / zero_epe_sup if zero_epe_sup > 0 else float("nan")
    else:
        model_epe_sup = zero_epe_sup = mean_epe_sup = cos_sup = rel_epe_sup = float("nan")

    if G_val_t.shape[0] > 0:
        model_epe_val = epe(P_val_t, G_val_t)
        zero_epe_val = G_val_t.norm(dim=-1).mean().item()
        meanvec_val = G_val_t.mean(dim=0, keepdim=True)
        mean_epe_val = (meanvec_val - G_val_t).norm(dim=-1).mean().item()
        cos_val = F.cosine_similarity(P_val_t, G_val_t, dim=-1).mean().item()
        rel_epe_val = model_epe_val / zero_epe_val if zero_epe_val > 0 else float("nan")
    else:
        model_epe_val = zero_epe_val = mean_epe_val = cos_val = rel_epe_val = float("nan")

    static_mag_t = torch.cat(static_mag, 0) if static_mag else torch.zeros(0)
    dynamic_mag_t = torch.cat(dynamic_mag, 0) if dynamic_mag else torch.zeros(0)

    frac_dyn_of_all = n_sup_cells / n_total_cells if n_total_cells else float("nan")
    frac_dyn_of_valid = n_sup_cells / n_valid_cells if n_valid_cells else float("nan")
    frac_valid_of_all = n_valid_cells / n_total_cells if n_total_cells else float("nan")

    def _stat(t):
        if t.numel() == 0:
            return {"mean": float("nan"), "median": float("nan"), "n": 0}
        return {"mean": float(t.mean()), "median": float(t.median()), "n": int(t.numel())}

    results = {
        "run_id": args.run_id, "ckpt_step": args.ckpt_step,
        "n_samples": n_samples, "n_euler_steps": args.n_euler_steps,
        "n_imgs_per_sample": n_imgs_per_sample,
        "dynamic_and_valid": {
            "model_epe": model_epe_sup, "zero_baseline_epe": zero_epe_sup,
            "mean_baseline_epe": mean_epe_sup, "relative_epe": rel_epe_sup,
            "cosine": cos_sup, "n_cells": int(G_sup_t.shape[0]),
        },
        "all_valid": {
            "model_epe": model_epe_val, "zero_baseline_epe": zero_epe_val,
            "mean_baseline_epe": mean_epe_val, "relative_epe": rel_epe_val,
            "cosine": cos_val, "n_cells": int(G_val_t.shape[0]),
        },
        "coverage": {
            "frac_dynvalid_of_all_cells": frac_dyn_of_all,
            "frac_dynvalid_of_valid_cells": frac_dyn_of_valid,
            "frac_valid_of_all_cells": frac_valid_of_all,
            "n_total_cells": n_total_cells, "n_valid_cells": n_valid_cells, "n_sup_cells": n_sup_cells,
            "n_samples_zero_dynamic": n_fell_to_zero_sup,
            "dynamic_fallback_to_valid": bool(sp.dynamic_fallback_to_valid),
        },
        "gt_flow_magnitude_m": {
            "dynamic_cells": _stat(dynamic_mag_t),
            "static_cells": _stat(static_mag_t),
        },
    }

    # ---------------- VERDICT ----------------
    verdict = "UNDETERMINED"
    reason = ""
    r = rel_epe_sup
    c = cos_sup
    if not np.isnan(r) and not np.isnan(c):
        if r < 0.5 and c > 0.5:
            verdict = "PREDICTIONS MEANINGFUL: model tracks real scene flow, not degenerate."
        elif r >= 0.9 and abs(c) < 0.15:
            verdict = "DEGENERATE: model predicts ~zero flow; low train loss is trivial."
        else:
            verdict = "PARTIAL / IN-BETWEEN: see numbers."
        reason = f"RelativeEPE(dyn)={r:.3f}, cosine(dyn)={c:.3f}"
    results["verdict"] = verdict
    results["verdict_reason"] = reason

    # ---------------- PRINT ----------------
    def f(x):
        return "nan" if (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"

    print("\n" + "=" * 78, flush=True)
    print("SCENE-FLOW PREDICTION DEGENERACY CHECK", flush=True)
    print("=" * 78, flush=True)
    print(f"run_id            : {args.run_id}", flush=True)
    print(f"ckpt_step         : {args.ckpt_step}", flush=True)
    print(f"n_samples         : {n_samples}   (imgs/sample={n_imgs_per_sample}, euler_steps={args.n_euler_steps})", flush=True)
    print("", flush=True)
    print("--- DYNAMIC ∩ VALID cells (the supervised set; the decisive test) ---", flush=True)
    print(f"  Model EPE           : {f(model_epe_sup)} m", flush=True)
    print(f"  Zero-flow EPE (base): {f(zero_epe_sup)} m   <-- degeneracy baseline", flush=True)
    print(f"  Mean-flow EPE (base): {f(mean_epe_sup)} m", flush=True)
    print(f"  RELATIVE EPE        : {f(rel_epe_sup)}      (Model/Zero; <<1 good, ~1 degenerate)", flush=True)
    print(f"  Cosine(pred,gt)     : {f(cos_sup)}          (dynamic cells)", flush=True)
    print(f"  n_cells             : {G_sup_t.shape[0]}", flush=True)
    print("", flush=True)
    print("--- ALL-VALID cells (mostly static; sanity) ---", flush=True)
    print(f"  Model EPE           : {f(model_epe_val)} m", flush=True)
    print(f"  Zero-flow EPE (base): {f(zero_epe_val)} m", flush=True)
    print(f"  Mean-flow EPE (base): {f(mean_epe_val)} m", flush=True)
    print(f"  RELATIVE EPE        : {f(rel_epe_val)}", flush=True)
    print(f"  Cosine(pred,gt)     : {f(cos_val)}", flush=True)
    print(f"  n_cells             : {G_val_t.shape[0]}", flush=True)
    print("", flush=True)
    print("--- coverage / diagnostics ---", flush=True)
    print(f"  frac dyn∩valid of ALL cells   : {f(frac_dyn_of_all)}", flush=True)
    print(f"  frac dyn∩valid of VALID cells : {f(frac_dyn_of_valid)}", flush=True)
    print(f"  frac valid of ALL cells       : {f(frac_valid_of_all)}", flush=True)
    print(f"  samples w/ 0 dynamic cells    : {n_fell_to_zero_sup} / {n_samples}", flush=True)
    print(f"  dynamic_fallback_to_valid     : {sp.dynamic_fallback_to_valid}", flush=True)
    print(f"  ||GT|| dynamic cells (m)      : mean={f(results['gt_flow_magnitude_m']['dynamic_cells']['mean'])} "
          f"median={f(results['gt_flow_magnitude_m']['dynamic_cells']['median'])}", flush=True)
    print(f"  ||GT|| static cells  (m)      : mean={f(results['gt_flow_magnitude_m']['static_cells']['mean'])} "
          f"median={f(results['gt_flow_magnitude_m']['static_cells']['median'])}", flush=True)
    print("", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"         {reason}", flush=True)
    print("=" * 78, flush=True)

    # ---------------- SAVE JSON ----------------
    json_path = out_dir / "flow_pred_metrics.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[saved] {json_path}", flush=True)

    # ---------------- VISUALIZATION ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_sample_viz.sort(key=lambda d: d["ncells"], reverse=True)
    picks = per_sample_viz[:6]
    if picks:
        ncol = 2
        nrow = len(picks)
        fig, axes = plt.subplots(nrow, ncol, figsize=(10, 4.2 * nrow), squeeze=False)
        gs = 16
        ys, xs = np.mgrid[0:gs, 0:gs]
        for r_i, rec in enumerate(picks):
            g16 = rec["target16"]; p16 = rec["pred16"]; m16 = rec["sup16"].astype(bool)
            zmax = max(1e-6, np.abs(np.concatenate([g16[..., 2][m16], p16[..., 2][m16]])).max()) if m16.any() else 1e-6
            for c_i, (fld, ttl, arrow_c) in enumerate([(g16, "GT", "tab:blue"), (p16, "PRED", "tab:red")]):
                ax = axes[r_i][c_i]
                # background = Z (depth) flow component
                bg = np.where(m16, fld[..., 2], np.nan)
                ax.imshow(bg, origin="upper", cmap="coolwarm", vmin=-zmax, vmax=zmax, alpha=0.65,
                          extent=[-0.5, gs - 0.5, gs - 0.5, -0.5])
                U = np.where(m16, fld[..., 0], np.nan)
                V = np.where(m16, fld[..., 1], np.nan)
                ax.quiver(xs, ys, U, -V, color=arrow_c, angles="xy", scale_units="xy",
                          scale=None, width=0.006)
                ax.set_title(f"{ttl}  (cells={rec['ncells']}, EPE={rec['epe']:.3f}m, cos={rec['cos']:.2f})", fontsize=9)
                ax.set_xlim(-0.5, gs - 0.5); ax.set_ylim(gs - 0.5, -0.5)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            f"Stage-1 scene-flow: GT vs sampled prediction (XY quiver, Z=bg color)\n"
            f"dyn∩valid RelEPE={f(rel_epe_sup)} cos={f(cos_sup)}  |  {verdict.split(':')[0]}",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        png_path = out_dir / "flow_pred_vs_gt.png"
        fig.savefig(png_path, dpi=130)
        print(f"[saved] {png_path}", flush=True)
    else:
        print("[WARN] no samples with dynamic cells -> no quiver figure", flush=True)


if __name__ == "__main__":
    main()
