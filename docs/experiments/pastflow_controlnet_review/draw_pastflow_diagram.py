#!/usr/bin/env python3
"""Render the past-flow ControlNet design flowchart (train vs inference) to PNG.
Matplotlib manual layout (no graphviz available). English labels (no CJK font)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 18))
ax.set_xlim(0, 15); ax.set_ylim(0, 18); ax.axis("off")

C = dict(shared="#e8eef7", vlm="#cfe0f3", scene="#f7e0c0", action="#d8ead0",
         train="#cdeccd", infer="#cfe3f5", note="#fbe7a2", stroke="#33475b")

def box(x, y, w, h, text, fc, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                linewidth=1.4, edgecolor=C["stroke"], facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True,
            fontweight=("bold" if bold else "normal"), color="#10202e")

def arrow(x1, y1, x2, y2, label="", color="#33475b", ls="-", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2, label, ha="center", va="center", fontsize=8.5,
                color=color, style="italic", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

ax.text(7.5, 17.5, "Past-Flow ControlNet into the Scene-Flow DiT  —  train vs inference",
        ha="center", fontsize=15, fontweight="bold", color="#10202e")
ax.text(7.5, 17.0, "Fix reversed scene-flow direction (single-frame motion ambiguity) by feeding past flow fields; current frame stays the PRIMARY, always-accurate signal.",
        ha="center", fontsize=9.5, color="#44586a")

# ---- main spine (center-right) ----
box(10.3, 16.0, 5.0, 0.9, "Current stereo frame (leftprimary)", C["shared"], 10.5, True)
box(10.3, 14.4, 5.6, 1.2, "VLM  (Qwen3.5-0.8B)  ->  last_hidden\n[CURRENT observation — ALWAYS accurate,\nPRIMARY conditioning; anchors everything]", C["vlm"], 9.5)
arrow(10.3, 15.55, 10.3, 15.0)

box(10.3, 10.9, 6.2, 2.0,
    "Scene-flow DiT  (DiT-B, 16 layers)\n\ncross-attn  <-  last_hidden  (current)\n+ per-layer ZERO-INIT MotionCoupler\n   <-  past_flow_hidden  x  gate(has_past_flow)\n[reuses the DiT's built-in coupler; step-0 == baseline]",
    C["scene"], 9.2)
arrow(10.3, 13.8, 10.3, 11.9, "current-frame conditioning")

# scene DiT two outputs
box(6.7, 7.7, 4.2, 1.5, "forward()  (supervised)\n-> predicted flow field", C["scene"], 9.2)
box(13.2, 8.4, 3.4, 1.5, "extract_conditioning_hidden()\n(GT-FREE)  ->  motion_hidden", C["scene"], 8.6)
arrow(9.4, 10.1, 7.2, 8.45, "", rad=-0.15)
arrow(11.6, 10.1, 12.9, 9.15, "", rad=0.12)
ax.text(7.6, 9.2, "past-flow branch feeds\nBOTH forward paths (codex #3)",
        ha="center", fontsize=8.2, color="#a0392b", style="italic", fontweight="bold")

# train vs infer targets under forward()
box(4.6, 5.7, 3.6, 1.1, "TRAIN:\nflow_loss  vs  GT current flow", C["train"], 9)
box(8.9, 5.7, 3.9, 1.1, "INFER:\nEuler-sample field  ->  push to FIFO", C["infer"], 9)
arrow(6.0, 7.0, 5.0, 6.3, color=C["train"] and "#2e7d32")
arrow(7.4, 7.0, 8.4, 6.3, color="#1565c0")

# action path
box(13.2, 6.3, 3.4, 1.0, "Action DiT\n(zero-init coupler <- motion_hidden)", C["action"], 8.8)
arrow(13.2, 7.65, 13.2, 6.8)
box(13.2, 4.6, 3.4, 0.9, "Action chunk", C["action"], 9.5, True)
arrow(13.2, 5.8, 13.2, 5.05)
box(13.2, 2.9, 3.9, 1.5,
    "INFER: execute chunk.\nIf an action FAILS -> the NEXT current\nframe reflects the true (bad) state\n-> re-plan from truth (anchor);\npast-flow is down-weighted, no cascade.",
    C["infer"], 8.3)
arrow(13.2, 4.15, 13.2, 3.65, color="#1565c0")

# ---- left branch: past-flow source ----
box(3.0, 13.6, 5.4, 1.7,
    "TRAIN past-flow source:\nGT 1-step flow[t-k*delta]  (sidecar,\nsame _apply_spatial_alignment as flow_gt)\n+ dropout p  -> has_past_flow=0  (learn no-past)\n+ noise-aug  (match self-pred error 0.80/0.70)",
    C["train"], 8.6)
box(3.0, 11.3, 5.4, 1.6,
    "INFER past-flow source:\nFIFO of last K SELF-predicted fields\n(per action-chunk; server reset per episode)\ncold-start: empty -> has_past_flow=0\n-> gate=0 -> exact single-frame baseline",
    C["infer"], 8.6)
box(3.0, 9.2, 5.0, 1.2,
    "PastFlowEncoder\n(K fields: tokenize + time-position emb\n+ cross-time attention -> velocity/accel)", C["shared"], 8.8)
box(3.0, 7.6, 3.4, 0.8, "past_flow_hidden", C["shared"], 9.5, True)
arrow(4.9, 12.9, 4.4, 9.7, color="#2e7d32", rad=-0.3)   # TRAIN source -> encoder (routed right)
arrow(3.0, 10.5, 3.0, 9.8, color="#1565c0")             # INFER source -> encoder
arrow(3.0, 8.6, 3.0, 8.0)
arrow(4.7, 7.6, 7.5, 10.2, "motion_hidden  (x per-sample gate)", rad=-0.25)

# legend
lx, ly = 0.5, 1.4
ax.add_patch(FancyBboxPatch((lx-0.2, ly-1.0), 6.7, 1.9, boxstyle="round,pad=0.1", lw=1, ec="#aab6c2", fc="#fbfcfe"))
ax.text(lx, ly+0.65, "Legend / key ideas", fontsize=9.5, fontweight="bold", color="#10202e")
for i,(c,t) in enumerate([(C["train"],"train-only"),(C["infer"],"infer-only"),(C["scene"],"scene DiT"),(C["vlm"],"VLM"),(C["action"],"action")]):
    ax.add_patch(FancyBboxPatch((lx+0.0+i*1.25, ly+0.15), 0.35, 0.28, boxstyle="round,pad=0.02", lw=0.8, ec=C["stroke"], fc=c))
    ax.text(lx+0.42+i*1.25, ly+0.29, t, fontsize=7.6, va="center")
ax.text(lx, ly-0.25, "Zero-init coupler + has_past_flow gate -> step-0 == baseline. Current frame is the anchor;\npast-flow is a DOWN-WEIGHTABLE hint (dropout + noise-aug) -> robust to cold-start & failed actions.",
        fontsize=8.2, color="#44586a")

plt.tight_layout()
out = "docs/experiments/pastflow_controlnet_review/pastflow_design_flow.png"
plt.savefig(out, dpi=145, bbox_inches="tight", facecolor="white")
print("SAVED", out)
