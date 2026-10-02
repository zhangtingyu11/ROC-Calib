"""Calibrate a saved set of object pairs without starting the web service."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON object pairs")
    parser.add_argument("--cloud-root", type=Path, help="Root for /data/cache/ URLs; defaults to the request directory")
    parser.add_argument("--initial", type=Path, help="Optional JSON matrix from a prior observation-based initialization")
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration.json"))
    parser.add_argument("--verify", type=Path, help="Compare the computed pose with a saved result, after solving")
    args = parser.parse_args()
    root = (args.cloud_root or args.request.parent).resolve()
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[name] = "1"
    os.environ.update(AUTOCALIB_DATA_ROOT=str(args.output.resolve().parent / ".cache"),
                      AUTOCALIB_PREPARED_ROOT=str(root), AUTOCALIB_PUBLIC_ROOT=str(root))
    import numpy as np
    from .calibration import CalibrationRequest
    from .region_solver import optimize

    payload = json.loads(args.request.read_text())
    payload["initialization_mode"] = "forward"
    if args.initial:
        payload["base_matrix"] = np.asarray(json.loads(args.initial.read_text())["matrix"]).reshape(-1).tolist()
        payload["initialization_mode"] = "existing"
    result = optimize(CalibrationRequest.model_validate(payload))
    result.update(transform="camera_from_lidar", translationUnit="meter", matrixOrder="row-major")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['pairCount']} pairs · {result['pointCount']} points · {result['loss']:.6f} px")
    print(f"Saved {args.output}")
    if args.verify:
        expected = json.loads(args.verify.read_text())
        difference = float(np.max(np.abs(np.asarray(result["matrix"]).reshape(4, 4) - np.asarray(expected["matrix"]).reshape(4, 4))))
        if difference > 1e-7:
            parser.exit(1, f"Saved-pose verification failed: max difference {difference:.3g}\n")
        print(f"Saved-pose verification passed ({difference:.3g}).")


if __name__ == "__main__":
    main()
