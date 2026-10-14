"""Deterministic identity-disjoint metadata splitting."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class IdentitySplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    person_col: str = "person_id"

    def ratios(self) -> dict[str, float]:
        ratios = {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": self.test_ratio,
        }
        if any(value < 0.0 for value in ratios.values()):
            raise ValueError("Identity split ratios cannot be negative")
        if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Identity split ratios must sum to 1.0")
        return ratios


def _allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw_counts = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: int(math.floor(value)) for split, value in raw_counts.items()}
    remaining = total - sum(counts.values())
    priority = {split: pos for pos, split in enumerate(ratios)}
    ranked = sorted(
        ratios,
        key=lambda split: (-(raw_counts[split] - counts[split]), priority[split]),
    )
    for split in ranked[:remaining]:
        counts[split] += 1
    return counts


def split_metadata_by_identity(
    metadata: pd.DataFrame,
    config: IdentitySplitConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Assign every identity and all of its images to exactly one split."""

    cfg = config or IdentitySplitConfig()
    ratios = cfg.ratios()
    if cfg.person_col not in metadata.columns:
        raise ValueError(f"Metadata does not contain {cfg.person_col!r}")
    if metadata.empty:
        raise ValueError("Cannot split empty metadata")

    person_ids = sorted(metadata[cfg.person_col].astype(str).unique().tolist())
    random.Random(cfg.random_seed).shuffle(person_ids)
    counts = _allocate_counts(len(person_ids), ratios)

    assignments: dict[str, str] = {}
    start = 0
    for split, count in counts.items():
        for person_id in person_ids[start : start + count]:
            assignments[person_id] = split
        start += count

    assigned = metadata.copy()
    assigned[cfg.person_col] = assigned[cfg.person_col].astype(str)
    assigned["split"] = assigned[cfg.person_col].map(assignments)
    if assigned["split"].isna().any():
        raise RuntimeError("Some identities were not assigned to a split")

    metadata_by_split: dict[str, pd.DataFrame] = {}
    for split in ratios:
        frame = assigned[assigned["split"] == split].copy()
        frame = frame.drop(columns=["index"], errors="ignore").reset_index(drop=True)
        frame.insert(0, "index", range(len(frame)))
        metadata_by_split[split] = frame

    all_metadata = pd.concat(metadata_by_split.values(), ignore_index=True)
    all_metadata = all_metadata.drop(columns=["index"], errors="ignore")
    all_metadata.insert(0, "index", range(len(all_metadata)))

    manifest = pd.DataFrame(
        [{"person_id": person_id, "split": split} for person_id, split in assignments.items()]
    ).sort_values(["split", "person_id"]).reset_index(drop=True)
    if manifest["person_id"].duplicated().any():
        raise RuntimeError("Identity leakage detected in split manifest")

    return metadata_by_split, all_metadata, manifest
