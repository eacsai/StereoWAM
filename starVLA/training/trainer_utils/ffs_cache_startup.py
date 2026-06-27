"""Shared trainer hook for FFS net0 cache startup validation."""
from __future__ import annotations


def maybe_validate_ffs_cache_startup(*, accelerator, model, vla_train_dataloader, raw_dataset=None) -> None:
    unwrapped = accelerator.unwrap_model(model)
    attach = getattr(unwrapped, "set_ffs_cache_dataset", None)
    validate = getattr(unwrapped, "maybe_validate_ffs_cache_startup", None)
    if not callable(attach) and not callable(validate):
        return
    dataset = getattr(vla_train_dataloader, "dataset", None)
    if dataset is None:
        dataset = raw_dataset
    if callable(attach):
        attach(dataset)
    if callable(validate):
        validate()
