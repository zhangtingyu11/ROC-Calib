"""Deterministic, full-density nuScenes fusion with a bounded RAM cache.

Only prebuilt descriptors are accepted. Camera extrinsics are never read here.
No generated point clouds are persisted, so annotation indices must retain the
frozen descriptor's source order and algorithm version.
"""
from collections import OrderedDict
from functools import lru_cache
import json
from pathlib import Path
import struct
import threading

import numpy as np
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

from .groups import PREPARED_ROOT

_LOCK = threading.RLock()
_CACHE = OrderedDict()
_CACHE_BYTES = 0
CACHE_LIMIT_BYTES = 128 * 1024 * 1024
ALGORITHM = 'nuscenes-keyframe-seven-sweeps-camera-time-v1'
VISIBILITY_POLICY = 'source-ego-footprint-excluded-v1'
EXCLUDED_SELF_RETURN = 2  # ACP1 byte 12; 0/1 retain their existing meanings.


@lru_cache(maxsize=4)
def _index(directory: str):
    path = PREPARED_ROOT / directory / 'fusion-index.json'
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if value['algorithm'] != ALGORITHM:
        raise ValueError('Unsupported fusion algorithm')
    return value


def descriptor(path: Path):
    try:
        parts = path.relative_to(PREPARED_ROOT).parts
    except ValueError:
        return None
    if len(parts) != 2 or not parts[0].startswith('nuscenes-test-fusion-v1-') or not parts[1].endswith('.bin'):
        return None
    index = _index(parts[0])
    frame = index.get('frames', {}).get(parts[1])
    return (index, frame) if frame else None


def is_virtual_cloud(path: Path) -> bool:
    return descriptor(path) is not None


def read_cloud_bytes(path: Path) -> bytes:
    global _CACHE_BYTES
    spec = descriptor(path)
    if spec is None:
        return path.read_bytes()
    # Serialize generation, bounding temporary memory as well as cached bytes.
    with _LOCK:
        key = str(path)
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
        index, frame = spec
        root = Path(index['rawRoot']).resolve()
        chunks = []
        for sweep_index, sweep in enumerate(frame['sweeps']):
            source = (root / sweep['path']).resolve()
            if root not in source.parents:
                raise ValueError('Sweep escapes raw data root')
            if 'sourceBytes' in sweep and source.stat().st_size != sweep['sourceBytes']:
                raise ValueError('Frozen sweep size changed')
            raw = np.fromfile(source, dtype='<f4').reshape(-1, 5)
            raw = raw[np.isfinite(raw[:, :3]).all(axis=1)]
            transform = np.asarray(sweep['transform'], dtype=np.float64)
            xyz = raw[:, :3].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
            records = np.zeros(len(raw), dtype=np.dtype({'names': ['xyz', 'stable', 'scanline'],
                'formats': [('<f4', 3), 'u1', 'u1'], 'offsets': [0, 12, 15], 'itemsize': 16}))
            records['xyz'] = xyz
            # Same full-density eligibility flag as paired imports; this is NOT
            # a claim that ego-pose compensation corrects moving objects.
            records['stable'] = 1
            # Source-frame footprint, BEFORE motion alignment. Keep XYZ/order
            # intact so existing point indices remain valid. This is not thinning.
            self_return = (np.abs(raw[:, 0]) < 0.8) & (np.abs(raw[:, 1]) < 2.7)
            records['stable'][self_return] = EXCLUDED_SELF_RETURN
            rings = raw[:, 4]
            if not np.isfinite(rings).all() or np.any(rings < 0) or np.any(rings > 31):
                raise ValueError('Invalid HDL-32E ring')
            records['scanline'] = rings.astype(np.uint8) + 1 + sweep_index * 32
            chunks.append(records.tobytes())
        count = sum(len(chunk) // 16 for chunk in chunks)
        result = b'ACP1' + struct.pack('<Iq', count, int(frame['timestampNs'])) + b''.join(chunks)
        while _CACHE and _CACHE_BYTES + len(result) > CACHE_LIMIT_BYTES:
            _, old = _CACHE.popitem(last=False)
            _CACHE_BYTES -= len(old)
        if len(result) <= CACHE_LIMIT_BYTES:
            _CACHE[key] = result
            _CACHE_BYTES += len(result)
        return result


class PreparedFiles(StaticFiles):
    async def get_response(self, path, scope):
        candidate = (PREPARED_ROOT / path).resolve()
        if not candidate.is_relative_to(PREPARED_ROOT.resolve()):
            return Response(status_code=404)
        if await run_in_threadpool(is_virtual_cloud, candidate):
            if scope['method'] not in ('GET', 'HEAD'):
                return Response(status_code=405)
            data = await run_in_threadpool(read_cloud_bytes, candidate)
            return Response(data if scope['method'] == 'GET' else b'', media_type='application/octet-stream',
                            headers={'Content-Length': str(len(data)), 'Cache-Control': 'no-store', 'X-Point-Visibility-Policy': VISIBILITY_POLICY})
        return await super().get_response(path, scope)
