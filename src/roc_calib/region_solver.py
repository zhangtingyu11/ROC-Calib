"""ROC-Calib: equally weighted object regions, with verified continuation."""
import time

import numpy as np
from scipy.optimize import minimize

from . import calibration as c
from .accepted_solver import solve as accepted_solve
from .pixel_region import PixelRegion

ALGORITHM_VERSION = "roc-calib-region-v1"
OPTIONS = {"xtol": 1e-5, "ftol": 1e-5, "maxiter": 260}


def cost(pairs, matrix):
    values = []
    for pair in pairs:
        uv = c._project_all_points(pair, pair["points"], matrix)
        if not np.isfinite(uv).all():
            return float("inf")
        values.append(float(np.mean(pair["region"].outside_distance(uv))))
    return float(np.mean(values))


def fit(pairs, seed, progress=None):
    report = progress or (lambda *args: None)
    started = time.monotonic()
    initial_loss = cost(pairs, seed)
    if not np.isfinite(initial_loss):
        raise ValueError("Initial pose has invalid projections. Check object matches and camera intrinsics.")
    cache = {}

    def objective(delta):
        key = np.asarray(delta, dtype=np.float64).tobytes()
        if key not in cache:
            cache[key] = cost(pairs, c._delta_matrix(delta) @ seed)
        return cache[key]

    report(30, "Region fit", None)
    initial = minimize(objective, np.zeros(6), method="Powell", options=OPTIONS)
    first = c._delta_matrix(initial.x) @ seed
    if not np.isfinite(cost(pairs, first)):
        raise ValueError("Region fit returned an invalid pose.")
    report(55, "Verifying candidate acceptance", None)
    accepted = accepted_solve(lambda matrix: cost(pairs, matrix), c._delta_matrix, first)
    prior = np.asarray(accepted["matrix"])
    prior_loss = accepted["loss"]
    directions = list(np.eye(6))
    for i in range(6):
        for j in range(i + 1, 6):
            for sign in [-1, 1]:
                directions.append((np.eye(6)[i] + sign * np.eye(6)[j]) / np.sqrt(2))
    units = np.array([np.deg2rad(.01)] * 3 + [.001] * 3)
    best, best_loss = prior.copy(), prior_loss
    report(75, "Checking local directions", None)
    for scale in [1., .1, .01]:
        for direction in directions:
            for sign in [-1, 1]:
                candidate = c._delta_matrix(units * direction * scale * sign) @ prior
                value = cost(pairs, candidate)
                if np.isfinite(value) and value < best_loss:
                    best, best_loss = candidate.copy(), value
    continued = prior_loss - best_loss > 1e-12 + 1e-12 * abs(prior_loss)
    if continued:
        report(85, "Final continuation", None)
        accepted = accepted_solve(lambda matrix: cost(pairs, matrix), c._delta_matrix, best)
    matrix = np.asarray(accepted["matrix"])
    final_loss = cost(pairs, matrix)
    if not np.isfinite(final_loss) or final_loss > prior_loss + 1e-12:
        raise ValueError("Final candidate did not pass objective verification.")
    return matrix, {
        "initialLoss": initial_loss, "loss": final_loss, "firstFitLoss": cost(pairs, first),
        "probeCount": 216, "continued": continued, "seconds": time.monotonic() - started,
        "optimizerSuccess": accepted["optimizerSuccess"],
        "optimizerMessage": accepted["optimizerMessage"],
        "iterationCount": int(initial.nit) + accepted["optimizerIterations"],
        "algorithmVersion": ALGORITHM_VERSION,
    }


def prepare(request):
    pairs = []
    for observation in request.pairs:
        pair = c._prepare_pair(observation)
        pair["points"] = pair["selected_points_full"]
        if len(pair["points"]) != len(observation.point_indices):
            raise ValueError("Every selected point must remain in the region objective.")
        pair["region"] = PixelRegion(pair["mask"])
        pairs.append(pair)
    return pairs


def optimize(request, progress=None):
    report = progress or (lambda *args: None)
    report(5, "Loading object pairs", None)
    pairs = prepare(request)
    if request.initialization_mode == "forward":
        seed, used, reprojection = c._pnp_initialize(pairs)
    else:
        seed = np.asarray(request.base_matrix, dtype=float).reshape(4, 4)
        used, reprojection = None, None
    matrix, result = fit(pairs, seed, report)
    inside = lambda pose: float(np.mean([
        np.mean(pair["region"].outside_distance(c._project_all_points(pair, pair["points"], pose)) == 0)
        for pair in pairs
    ]))
    report(100, "Calibration complete", None)
    return {
        **result, "matrix": matrix.reshape(-1).tolist(), "initialMatrix": seed.reshape(-1).tolist(),
        "fullMatchMatrix": matrix.reshape(-1).tolist(), "delta": c._matrix_delta(matrix, seed).tolist(),
        "originalError": result["initialLoss"], "optimizedError": result["loss"],
        "fullMatchError": result["loss"], "originalContainmentError": result["initialLoss"],
        "optimizedContainmentError": result["loss"], "fullMatchContainmentError": result["loss"],
        "originalInsideRatio": inside(seed), "optimizedInsideRatio": inside(matrix),
        "fullMatchInsideRatio": inside(matrix), "pairCount": len(pairs),
        "pointCount": sum(len(pair["points"]) for pair in pairs),
        "pnpInlierCount": used, "pnpReprojectionError": reprojection,
        "converged": result["optimizerSuccess"], "qualityStatus": "unverified",
        "qualityWarnings": ["Inspect projections before using this transform. Region fit does not establish physical accuracy."],
        "pairDiagnostics": [], "suggestedReviewAnnotationIds": [],
    }
