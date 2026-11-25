"""Bridge saved workbench annotations to the standalone paper solver."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .calibration import _matrix_delta, _resolve_cloud
from .nuscenes_fusion import is_virtual_cloud, read_cloud_bytes


ALGORITHM_VERSION = "roc-calib-region-contour-20261026"


def _write_request(request, directory):
    payload = request.model_dump()
    clouds = {}
    for pair in payload["pairs"]:
        if not pair["point_indices"]:
            raise ValueError("Each correspondence needs selected LiDAR points.")
        if pair["distortion_model"] not in {"rational", "radtan", "fisheye"}:
            raise ValueError("Unsupported camera distortion model.")
        runs = pair["mask_rle"]
        if any(run < 0 for run in runs) or sum(runs) != pair["mask_width"] * pair["mask_height"]:
            raise ValueError("Mask runs must cover the complete image.")
        path = _resolve_cloud(pair["cloud_url"])
        if path not in clouds:
            asset = directory / f"cloud-{len(clouds)}.bin"
            if is_virtual_cloud(path):
                asset.write_bytes(read_cloud_bytes(path))
            else:
                asset.symlink_to(path)
            clouds[path] = asset.name
        pair["cloud_url"] = clouds[path]
    (directory / "request.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


def optimize(request, progress=None, cancelled=None):
    executable = os.environ.get("ROC_CALIB_NATIVE_BIN") or shutil.which("autocalib_native")
    if not executable:
        raise RuntimeError("Paper solver missing. Rebuild the workbench or set ROC_CALIB_NATIVE_BIN.")
    report = progress or (lambda *args: None)
    timeout = float(os.environ.get("ROC_CALIB_SOLVER_TIMEOUT", "1800"))
    with tempfile.TemporaryDirectory(prefix="roc-calib-") as scratch:
        directory = Path(scratch)
        report(5, "正在读取已确认的配对", None)
        _write_request(request, directory)
        output = directory / "result.json"
        report(15, "正在运行 C++ 区域与轮廓求解器", None)
        with (directory / "stderr.log").open("w+") as errors:
            process = subprocess.Popen(
                [executable, "--root", str(directory), "--output", str(output)],
                stdout=subprocess.DEVNULL, stderr=errors,
            )
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if cancelled is not None and cancelled.is_set():
                        raise RuntimeError("任务已取消")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Solver timed out")
                    try:
                        process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process.poll() is None:
                    process.terminate()  # The native supervisor forwards SIGTERM to its worker.
                    process.wait()
            if process.returncode:
                errors.seek(0)
                raise RuntimeError(errors.read().strip() or f"Native solver exited with {process.returncode}")
        result = json.loads(output.read_text())
    solve = result["solve"]
    if not (solve["completed"] and solve["finalAuditCompleted"]):
        raise RuntimeError("Native solver did not finish verification.")
    matrix = np.asarray(solve["matrix"])
    initial = np.asarray(result["initialMatrix"])
    return {
        "algorithmVersion": ALGORITHM_VERSION,
        "initializationMode": request.initialization_mode,
        "matrix": matrix.ravel().tolist(), "initialMatrix": initial.ravel().tolist(),
        "delta": _matrix_delta(matrix, initial).tolist(),
        "originalError": solve["initialLoss"], "optimizedError": solve["loss"],
        "pairCount": result["pairs"], "pointCount": result["points"],
        "finalAuditCompleted": solve["finalAuditCompleted"], "qualityStatus": "unverified",
    }
