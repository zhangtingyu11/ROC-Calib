from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class StoragePaths:
    """Resolved runtime storage paths.

    New deployments use one data root with stable child directories.  The
    legacy layout is detected instead of being moved so an existing mine-site
    installation keeps working without a destructive migration.
    """

    data_root: Path | None
    group_root: Path
    bag_root: Path
    paired_root: Path
    prepared_root: Path
    export_root: Path
    layout: str


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _looks_like_legacy_root(root: Path) -> bool:
    if (root / "rosbags").exists() or (root / "prepared-cache").exists():
        return True
    try:
        return any((child / "group.json").is_file() for child in root.iterdir() if child.is_dir())
    except OSError:
        return False


def resolve_storage_paths(environ: Mapping[str, str] | None = None) -> StoragePaths:
    env = os.environ if environ is None else environ
    configured_root = env.get("AUTOCALIB_DATA_ROOT", "").strip()

    if configured_root:
        data_root = _resolved(configured_root)
        requested_layout = env.get("AUTOCALIB_STORAGE_LAYOUT", "auto").strip().lower()
        if requested_layout not in {"auto", "legacy", "v2"}:
            raise ValueError("AUTOCALIB_STORAGE_LAYOUT must be auto, legacy, or v2")
        layout = (
            "legacy"
            if requested_layout == "legacy" or (requested_layout == "auto" and _looks_like_legacy_root(data_root))
            else "v2"
        )
        defaults = {
            "group": data_root if layout == "legacy" else data_root / "groups",
            "bag": data_root / ("rosbags" if layout == "legacy" else "bags"),
            "paired": data_root / "paired",
            "prepared": data_root / ("prepared-cache" if layout == "legacy" else "cache"),
            "export": data_root / "exports",
        }
    else:
        # Keep direct service launches and older compose files compatible.
        data_root = None
        layout = "legacy-env"
        group_default = _resolved("/calibration-groups")
        defaults = {
            "group": group_default,
            "bag": _resolved("/bag-source"),
            "paired": _resolved("/paired-source"),
            "prepared": group_default / "prepared-cache",
            "export": group_default / "exports",
        }

    return StoragePaths(
        data_root=data_root,
        group_root=_resolved(env.get("AUTOCALIB_GROUP_ROOT", str(defaults["group"]))),
        bag_root=_resolved(env.get("AUTOCALIB_BAG_ROOT", str(defaults["bag"]))),
        paired_root=_resolved(env.get("AUTOCALIB_PAIRED_ROOT", str(defaults["paired"]))),
        prepared_root=_resolved(env.get("AUTOCALIB_PREPARED_ROOT", str(defaults["prepared"]))),
        export_root=_resolved(env.get("AUTOCALIB_EXPORT_ROOT", str(defaults["export"]))),
        layout=layout,
    )


STORAGE = resolve_storage_paths()
