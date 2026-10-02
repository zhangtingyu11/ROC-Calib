"""Download and verify the full-density CARLA example."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request
import os
import shutil
import threading

_IMPORT_LOCK = threading.Lock()

URL = "https://github.com/zhangtingyu11/ROC-Calib/releases/download/v0.1.0/carla-demo.tar.gz"
SHA256 = "1358b17a03338006fe6623e7b1f3aa02776622642d3765e49fbea3d1cb07b764"


def download(destination):
    checksums = json.loads((destination / "checksums.json").read_text())
    assets = {name: digest for name, digest in checksums.items() if name.startswith(("clouds/", "images/"))}
    if all((destination / name).is_file() and hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest for name, digest in assets.items()):
        print("CARLA example is ready.")
        return
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "carla-demo.tar.gz"
        print("Downloading CARLA example (56 MiB)…")
        urllib.request.urlretrieve(URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise ValueError("CARLA archive checksum mismatch")
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if set(member.name for member in members) != set(assets):
                raise ValueError("Unexpected archive contents")
            for member in members:
                if not member.isfile() or not (destination / member.name).resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path")
            tar.extractall(destination, members=members, filter="data")
    for name, digest in assets.items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    print("CARLA example is ready.")


def install():
    from .groups import (PAIRED_ROOT, GROUP_ROOT, GroupCreate, GroupImport,
                         CameraGroupState, create_group, import_datasets,
                         write_camera_state, _now)
    from .paired_datasets import prepare_paired_dataset

    source = Path(os.environ.get("ROC_CALIB_DEMO_DIR", Path(__file__).resolve().parents[2] / "demo/carla"))
    if not source.exists():
        source = Path("/demo/carla")
    with _IMPORT_LOCK:
        existing = GROUP_ROOT / "CARLA-demo" / "group.json"
        if existing.exists():
            return json.loads(existing.read_text())
        target = PAIRED_ROOT / "carla-demo"
        target.mkdir(parents=True, exist_ok=True)
        for name in ["autocalib.json", "checksums.json", "request.json"]:
            shutil.copy2(source / name, target / name)
        for folder in ["images", "clouds"]:
            if (source / folder).exists():
                shutil.copytree(source / folder, target / folder, dirs_exist_ok=True)
        download(target)
        dataset, _ = prepare_paired_dataset("carla-demo", lambda *args: None)
        create_group(GroupCreate(name="CARLA demo", rig_id="carla-demo"), group_id="CARLA-demo")
        group = import_datasets("CARLA-demo", GroupImport(dataset_ids=[dataset["id"]], paired_source_paths=["carla-demo"]))
        request = json.loads((source / "request.json").read_text())
        timestamps = {str(frame["label"]): frame["timestampNs"] for frame in dataset["lidars"][0]["frames"]}
        annotations = []
        for pair in request["pairs"]:
            label = Path(pair["cloud_url"]).stem
            annotations.append({
                "id": pair["annotation_id"], "datasetId": dataset["id"], "frame": label,
                "lidarId": dataset["lidars"][0]["id"], "cameraId": dataset["cameras"][0]["id"],
                "taskGroupId": "CARLA-demo", "status": "paired", "useForOptimization": True,
                "pointIndices": pair["point_indices"], "pointCount": len(pair["point_indices"]),
                "maskPixelCount": sum(pair["mask_rle"][1::2]),
                "imageMask": {"width": pair["mask_width"], "height": pair["mask_height"], "rle": pair["mask_rle"]},
                "lidarTimestampNs": timestamps[label], "cameraTimestampNs": timestamps[label],
                "calibrationSource": "carla-demo:intrinsics-only",
                "imageCoordinateSystem": "raw-distorted-v1", "savedAt": _now(),
            })
        write_camera_state("CARLA-demo", CameraGroupState(annotations=annotations))
        return group
