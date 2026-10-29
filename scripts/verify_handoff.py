#!/usr/bin/env python3
"""Read-only source/metadata checks for a fresh research-branch checkout.

Does not import torch, load checkpoints/pickle files, install packages, fetch,
download, or launch a GPU job. Asset checks are existence/schema checks only.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "AGENTS.md",
    "docs/TAKEOVER.md",
    "docs/CODEX_PROJECT_RECOVERY.md",
    "docs/ASSET_MANIFEST.md",
    "docs/experiments/20261029_stereo_wrist_utonia_plan.md",
    "docs/experiments/20261029_l40_results.md",
    "docs/handoff/environment.json",
    "docs/handoff/experiment_results.json",
    "scripts/codex_restore_context.sh",
    "starVLA/model/framework/VLM4A/QwenGR00T.py",
    "starVLA/model/framework/VLM4A/QwenGR00T_UtoniaPromptTokenFFS.py",
    "starVLA/model/modules/stereo/cam_branch_attention.py",
    "starVLA/model/modules/stereo/visual_attention.py",
    "scripts/a800/test_visual_attention_ablation.py",
    "starVLA/config/deepseeds/deepspeed_zero2_ga4.yaml",
    "starVLA/config/deepseeds/ds_config_ga4.yaml",
)


def check_sources() -> list[str]:
    errors = []
    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            errors.append(f"missing source: {rel}")
    if errors:
        return errors
    for rel in REQUIRED:
        path = ROOT / rel
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"invalid Python {rel}: {exc}")
    for rel in (
        "docs/TAKEOVER.md",
        "docs/CODEX_PROJECT_RECOVERY.md",
        "docs/ASSET_MANIFEST.md",
        "docs/experiments/20261029_l40_results.md",
        "docs/experiments/20261029_stereo_wrist_utonia_plan.md",
    ):
        path = ROOT / rel
        for target in re.findall(r"\[[^\]]+\]\(([^)\s]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            linked = (path.parent / target.split("#", 1)[0]).resolve()
            if not linked.exists():
                errors.append(f"broken local link: {rel} -> {target}")
    try:
        data = json.loads((ROOT / "docs/handoff/experiment_results.json").read_text(encoding="utf-8"))
        expected_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        if data["suite_order"] != expected_suites:
            errors.append("suite order changed in historical results")
        seen = set()
        for record in data["records"]:
            if record["id"] in seen:
                errors.append(f"duplicate experiment: {record['id']}")
            seen.add(record["id"])
            rates = record["rates"]
            if len(rates) != 4 or any(not 0 <= x <= 1 for x in rates):
                errors.append(f"invalid rates: {record['id']}")
            elif abs(sum(rates) / 4 - record["mean_rate"]) > 1e-10:
                errors.append(f"incorrect mean: {record['id']}")
        json.loads((ROOT / "docs/handoff/environment.json").read_text(encoding="utf-8"))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid handoff JSON: {exc}")
    return errors


def check_assets(asset_root: Path) -> list[str]:
    errors = []
    for rel in (
        "Pretrained_models/Qwen3.5-0.8B/config.json",
        "Pretrained_models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors",
        "Pretrained_models/Utonia/utonia.pth",
    ):
        path = asset_root / rel
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing/empty asset: {path}")
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        root = asset_root / "Datasets/LEROBOT_LIBERO_STEREO_LEFTPRIMARY" / f"{suite}_no_noops_1.0.0_lerobot"
        info = root / "meta/info.json"
        if not info.is_file():
            errors.append(f"missing dataset metadata: {info}")
            continue
        try:
            metadata = json.loads(info.read_text(encoding="utf-8"))
            for key in ("observation.images.image", "observation.images.right_view", "observation.images.wrist_image"):
                if key not in metadata.get("features", {}):
                    errors.append(f"missing image feature {key}: {info}")
        except (ValueError, OSError) as exc:
            errors.append(f"invalid metadata {info}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true", help="validate portable source handoff only")
    parser.add_argument("--asset-root", type=Path, help="optional playground root; existence/metadata only")
    args = parser.parse_args()
    if args.source_only and args.asset_root is not None:
        parser.error("--source-only and --asset-root are mutually exclusive")
    errors = check_sources()
    if args.asset_root is not None:
        errors.extend(check_assets(args.asset_root.expanduser().resolve()))
    for error in errors:
        print(f"FAIL {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"PASS source handoff ({len(REQUIRED)} required files, links, Python syntax, historical-result arithmetic)")
    if args.asset_root is not None:
        print("PASS core asset existence/metadata; weights, dataset contents and camera K/T not certified")
    else:
        print("Heavy assets and new three-camera implementation are not validated by source-only recovery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
