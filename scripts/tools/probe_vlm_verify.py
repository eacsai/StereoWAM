#!/usr/bin/env python3
"""定性 probe: 验 Qwen3.5-0.8B 能不能从正交田字格读出物体。

机制: 从 ortho cache sqlite 直读 grid(不重渲染, 跟生产一致) + 从 dataset 读 primary 相机图/lang,
调 Qwen3.5-0.8B 问物体 → positive control(primary 相机图 vs grid 田字格) → 粗略命中率(GT 物体名从 lang 正则抽)。

⚠️ 边界(codex Stage1.5):
  - High4: 不算精确 IoU(无 GT), 只做 "VLM 答案 contains GT 物体名" 的粗略命中率。
  - Med7: zero-shot detection ≠ full-ft action 能力。pass/fail 只判 "VLM 读得出物体", 不外推 "ortho 对训练有帮助"。
  - origin=点云质心(每帧漂), 只能定性比较, 不做绝对位置。
跑法(4090d 项目 venv):
  cd /data/wangqiwei/ICLR2026/starVLA && .venv/bin/python scripts/tools/probe_vlm_verify.py \
    --n-frames 30 --view grid --out-dir playground/probe_vlm_verify
"""
from __future__ import annotations
import sys, json, sqlite3, io, argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # starVLA/
sys.path.insert(0, str(ROOT))

from PIL import Image


# ---------- dataset + cache ----------
def load_cfg_and_mixture(config_yaml, data_root, data_mix, video_backend):
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat
    cfg = OmegaConf.load(config_yaml)
    override = OmegaConf.create({"datasets": {"vla_data": {
        "data_root_dir": data_root, "data_mix": data_mix, "video_backend": video_backend}}})
    cfg = OmegaConf.merge(cfg, override)
    return apply_config_compat(cfg)


def get_single_suite(cfg, suite_name):
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    mixture = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train", seed=42)
    return next(d for d in mixture.datasets if d.dataset_name == suite_name)


def read_cache_view(cache_dir, suite_name, row, view="grid"):
    """复用 datasets.py:761 _load_ortho_cache_image 逻辑, 直读 sqlite blob。"""
    suite_dir = Path(cache_dir) / suite_name
    db = suite_dir / "ortho_views.sqlite"
    if not db.exists():
        return None
    conn = sqlite3.connect(f"file:{db}?immutable=1", uri=True)  # immutable: cache 预计算在写同 db, 不加锁不读journal, 读 committed 快照(done rows)
    col = "grid_png" if view == "grid" else "axo_png"
    blob = conn.execute(f"SELECT {col} FROM ortho_views WHERE row=?", (int(row),)).fetchone()
    conn.close()
    if not blob or blob[0] is None:
        return None
    return Image.open(io.BytesIO(blob[0])).convert("RGB")


# ---------- VLM (Qwen3.5-0.8B 是 VL, 无 blocker) ----------
def load_vlm(model_path):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_path)
    proc.tokenizer.padding_side = "left"   # 陷阱: starVLA 用左 pad
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, proc


def ask_vlm(model, proc, pil_img, prompt, max_new=64):
    """图文问: structured content。repetition_penalty 抑制 0.8B greedy 重复退化, 短输出。"""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": prompt},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[pil_img], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                         repetition_penalty=1.3, no_repeat_ngram_size=4)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return proc.decode(gen, skip_special_tokens=True).strip()


# ---------- GT 物体名 (从 task 文本粗略抽) ----------
def extract_gt_object(lang):
    """LIBERO task 如 'pick up the red mug and place it in the basket' → 'red mug'。粗略, 仅作命中率分母。"""
    m = re.search(r"(?:pick up|grab|take|get)\s+(?:the\s+)?(.+?)(?:\s+and\s+|\s+from\s+|\s+to\s+|\s+in\s+the\s+|$)",
                  lang, re.I)
    return m.group(1).strip() if m else None


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-yaml", default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    ap.add_argument("--data-root", default="playground/Datasets/LEROBOT_LIBERO_OURRENDER_PW")
    ap.add_argument("--data-mix", default="libero_all_sfstereo_leftprimary")   # 必须
    ap.add_argument("--video-backend", default="torchvision_av")               # yaml default
    ap.add_argument("--suite", default="libero_object_no_noops_1.0.0_lerobot")
    ap.add_argument("--cache-dir", default="playground/Caches/ortho_views_leftprimary_probe_v1")
    ap.add_argument("--vlm", default="playground/Pretrained_models/Qwen3.5-0.8B")
    ap.add_argument("--view", default="grid", choices=["grid", "axo"])
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="cache", choices=["cache", "render"],
                    help="cache=读sqlite(旧axo); render=独立渲调OrthoRenderer(用改后render_axonometric)")
    ap.add_argument("--ffs-model", default="/data/wangqiwei/ICLR2026/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg_and_mixture(args.config_yaml, args.data_root, args.data_mix, args.video_backend)
    single = get_single_suite(cfg, args.suite)
    model, proc = load_vlm(args.vlm)
    renderer = None
    if args.source == "render":
        sys.path.insert(0, str(ROOT / "scripts" / "4090d"))
        from precompute_ortho_view_cache import OrthoRenderer
        from argparse import Namespace
        renderer = OrthoRenderer(Namespace(
            ffs_repo_dir="/data/wangqiwei/ICLR2026/Fast-FoundationStereo",
            ffs_model_path=args.ffs_model, valid_iters=8, max_disp=192,
            axo_size=448, level_table="auto"))
        print("[probe] source=render, 用改后 render_axonometric 独立渲")

    PROMPT_PRIMARY = "Look at this robot camera image. In one short sentence, name the main objects you see."
    if args.view == "grid":
        PROMPT_VIEW = ("This image is a 2x2 grid: top-view, front-view, side-view, axonometric of a tabletop "
                       "rendered from a stereo point cloud. In one short sentence, name the objects you can see.")
    else:
        PROMPT_VIEW = ("This image is an axonometric (3D) view of a tabletop rendered from a stereo point cloud. "
                       "In one short sentence, name the objects you can see.")

    results = []
    for r in range(min(args.n_frames, len(single.all_steps))):
        s = single[r]
        if args.source == "render":
            s["image"] = s["image"][:2]   # 砍可能的 ortho 第3张, OrthoRenderer 要 2 张
            _grid, _axo, _info = renderer.render(s)
            _arr = _axo if args.view == "axo" else _grid
            grid = Image.fromarray(_arr)   # renderer 返回 numpy ndarray → PIL
        else:
            grid = read_cache_view(args.cache_dir, args.suite, r, args.view)
            if grid is None:
                print(f"[skip] row {r}: cache not done yet (precompute 还在跑)")
                continue
        primary = s["image"][0]   # PIL, primary=几何右眼
        lang = str(s["lang"])
        gt = extract_gt_object(lang)

        ans_p = ask_vlm(model, proc, primary, PROMPT_PRIMARY)
        ans_g = ask_vlm(model, proc, grid, PROMPT_VIEW)
        hit_p = (gt.lower() in ans_p.lower()) if gt else None
        hit_g = (gt.lower() in ans_g.lower()) if gt else None

        rec = {"row": r, "lang": lang, "gt_object": gt,
               "ans_primary": ans_p, "hit_primary": hit_p,
               "ans_grid": ans_g, "hit_grid": hit_g}
        results.append(rec)
        grid.save(out / f"row{r:02d}_{args.view}.png")
        primary.save(out / f"row{r:02d}_primary.png")
        print(f"[{len(results)}] row{r} gt={gt!r:30} primary={hit_p}  grid={hit_g}")

    n_gt = sum(1 for x in results if x["gt_object"])
    hp = sum(1 for x in results if x["hit_primary"])
    hg = sum(1 for x in results if x["hit_grid"])
    print(f"\n=== 定性统计 (n={len(results)}, 有GT={n_gt}) ===")
    print(f"  primary 相机图命中: {hp}/{n_gt} = {hp/max(n_gt,1):.0%}  (positive control)")
    print(f"  grid 田字格命中:    {hg}/{n_gt} = {hg/max(n_gt,1):.0%}  (idea 验证)")
    print(f"  ⚠️ zero-shot detection≠action能力, 仅判 VLM 读不读得出, 不外推训练效果")

    with open(out / "results.jsonl", "w") as fh:
        for x in results:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"wrote {out}/results.jsonl + 图(供人看)")


if __name__ == "__main__":
    main()
